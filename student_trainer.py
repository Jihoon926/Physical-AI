import argparse
import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from student_policy import POLICY
from waypoint import build_waypoints


EPS = 1e-6
RAW_FEATURE_KEYS = (
    "curr_bpos",
    "curr_bquat",
    "hand_pos",
    "hand_quat",
    "target_pos",
    "waypoint_pos",
)

# Per-stage oversampling weights.
# Grasp (2), place (5), release (6) are least frequent but most critical —
# oversample them so the model doesn’t underfit the hardest transitions.
STAGE_SAMPLE_WEIGHTS = np.array([1.0, 1.5, 4.0, 2.0, 1.5, 4.0, 4.0, 2.0])


def infer_waypoint_index(waypoint_pos, waypoint_grip, block_init_xpos, target_xpos):
    """Infer waypoint stage index (0–7) for every timestep without rerunning the sim.

    The stored waypoint_pos is exactly one of the 8 waypoint target positions
    computed by build_waypoints().  We match each timestep’s stored
    (waypoint_pos, waypoint_grip) against the 8 candidates and return the
    index of the closest match.

    Position alone is ambiguous for 4 pairs that share the same target pos:
      approach(0) vs lift(3)   — both above_block, but grip 255 vs 0
      descend(1) vs grasp(2)   — both on_block,    but grip 255 vs 0
      move(4) vs retreat(7)    — both above_target, but grip 0   vs 255
      place(5) vs release(6)   — both on_target,   but grip 0   vs 255
    Adding a large penalty for grip-side mismatch resolves every ambiguity.

    Args:
        waypoint_pos:    np.ndarray [T, 3] — stored current waypoint positions.
        waypoint_grip:   np.ndarray [T]    — stored current waypoint grip values.
        block_init_xpos: np.ndarray [3]    — block initial position.
        target_xpos:     np.ndarray [3]    — target position.

    Returns:
        np.ndarray [T] int64 — waypoint stage index per timestep.
    """
    waypoints = build_waypoints(block_init_xpos, target_xpos)
    wp_positions = np.array([w.pos for w in waypoints])  # [8, 3]
    wp_grips = np.array([w.grip for w in waypoints])     # [8]

    # Position distance: [T, 8]
    pos_dists = np.linalg.norm(
        waypoint_pos[:, None, :] - wp_positions[None, :, :], axis=-1
    )
    # Large penalty when grip side (open/closed) doesn’t match.
    grip_mismatch = (waypoint_grip[:, None] < 128) != (wp_grips[None, :] < 128)  # [T, 8]
    score = pos_dists + grip_mismatch.astype(np.float64) * 1000.0

    return np.argmin(score, axis=-1).astype(np.int64)


class PickAndPlaceDataset(Dataset):
    def __init__(self, input_data, output_data, penalty_data=None):
        self.input_data = input_data
        self.output_data = output_data
        self.penalty_data = penalty_data

    def __len__(self):
        return len(self.output_data)

    def __getitem__(self, idx):
        input_sample = {}
        for k, v in self.input_data.items():
            if k == "waypoint_idx":
                # Keep as LongTensor — consumed by nn.Embedding, not z-score normalized.
                input_sample[k] = torch.tensor(int(v[idx]), dtype=torch.long)
            else:
                input_sample[k] = torch.from_numpy(v[idx]).float()
        output_sample = torch.from_numpy(self.output_data[idx]).float()
        if self.penalty_data is None:
            return input_sample, output_sample
        penalty_sample = torch.tensor(self.penalty_data[idx]).float()
        return input_sample, output_sample, penalty_sample


def weighted_mse(predictions, targets, weights):
    return ((predictions - targets) ** 2 * weights).mean()


def weighted_huber(predictions, targets, weights, beta=1.0):
    diff = torch.abs(predictions - targets)
    huber = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return (huber * weights).mean()


def waypoint_action_penalty(
    predictions,
    output_mean,
    output_std,
    penalty_weights,
):
    pred_raw = predictions * output_std + output_mean
    joint_delta = pred_raw[:, :7]
    return (joint_delta.pow(2).mean(dim=-1) * penalty_weights).mean()


def add_train_input_noise(batch_inputs, input_stats):
    noisy_keys = {
        "curr_qpos",
        "curr_qvel",
        "curr_bpos",
        "curr_bquat",
        "hand_pos",
        "hand_quat",
        "waypoint_pos",
        "target_pos",   # helps generalise to novel target positions
    }
    for k in noisy_keys:
        if k in batch_inputs:
            batch_inputs[k] = batch_inputs[k] + torch.randn_like(batch_inputs[k]) * torch.from_numpy(0.5*(input_stats[k]["std"])).float().to(batch_inputs[k].device)
    return batch_inputs


def load_split(file_paths, action_penalty_radius):
    input_data = {"curr_qpos": [], "curr_qvel": [], "curr_bpos": [], "curr_bquat": [],
                  "target_pos": [], "block_size": [], "hand_pos": [], "hand_quat": [],
                  "waypoint_pos": [], "waypoint_grip": [], "waypoint_idx": []}
    output_data = []
    penalty_data = []

    for fp in tqdm(file_paths, desc="Loading trajectories"):
        traj_data = np.load(fp, allow_pickle=True)

        block_size = traj_data["block_size"]
        block_pos = traj_data["bpos"]
        block_quat = traj_data["bquat"]
        target_pos = traj_data["target_xpos"]
        ctrl = traj_data["ctrl"]
        qpos = traj_data["qpos"]
        qvel = traj_data["qvel"]
        waypoint_pos = traj_data["waypoint_pos"]
        waypoint_grip = traj_data["waypoint_grip"]
        hand_pos = traj_data["hpos"]
        hand_quat = traj_data["hquat"]

        input_data["curr_qpos"].append(qpos)
        input_data["curr_qvel"].append(qvel)
        input_data["curr_bpos"].append(block_pos)
        input_data["curr_bquat"].append(block_quat)
        input_data["waypoint_pos"].append(waypoint_pos)
        input_data["waypoint_grip"].append(waypoint_grip[:, None].astype(np.float32))
        input_data["block_size"].append(block_size[None, None].repeat(len(ctrl), axis=0))
        input_data["target_pos"].append(target_pos[None, :].repeat(len(ctrl), axis=0))
        input_data["hand_pos"].append(hand_pos)
        input_data["hand_quat"].append(hand_quat)
        output_data.append(ctrl)

        if action_penalty_radius > 0.0:
            dist_to_waypoint = np.linalg.norm(waypoint_pos - hand_pos, axis=-1)
            penalty_weight = np.clip(1.0 - dist_to_waypoint / action_penalty_radius, 0.0, 1.0)
        else:
            penalty_weight = np.zeros(len(ctrl), dtype=np.float32)
        penalty_data.append(penalty_weight.astype(np.float32))

        # Infer waypoint stage index from stored waypoint_pos + grip.
        # No re-simulation needed — see infer_waypoint_index() for the matching logic.
        block_init_xpos = traj_data["block_init_xpos"]
        waypoint_idx = infer_waypoint_index(
            waypoint_pos, waypoint_grip,
            block_init_xpos, target_pos,
        )
        input_data["waypoint_idx"].append(waypoint_idx)

    input_data = {k: np.concatenate(v, axis=0) for k, v in sorted(input_data.items())}
    output_data = np.concatenate(output_data, axis=0)
    penalty_data = np.concatenate(penalty_data, axis=0)
    return input_data, output_data, penalty_data


def normalize_data(train_input, valid_input, train_output, valid_output):
    train_output_dq = train_output.copy()
    valid_output_dq = valid_output.copy()
    train_output_dq[:, :7] = train_output_dq[:, :7] - train_input["curr_qpos"][:, :7]
    valid_output_dq[:, :7] = valid_output_dq[:, :7] - valid_input["curr_qpos"][:, :7]

    input_stats = {}
    train_input_norm = {}
    valid_input_norm = {}

    for k in sorted(train_input.keys()):
        if k == "waypoint_idx":
            # Integer index — passed to nn.Embedding, not z-score normalised.
            train_input_norm[k] = train_input[k]
            valid_input_norm[k] = valid_input[k]
            continue
        mean = train_input[k].mean(axis=0, keepdims=True)
        std = np.maximum(train_input[k].std(axis=0, keepdims=True), EPS)
        input_stats[k] = {"mean": mean, "std": std}
        train_input_norm[k] = (train_input[k] - mean) / std
        valid_input_norm[k] = (valid_input[k] - mean) / std

    for k in RAW_FEATURE_KEYS:
        train_input_norm[f"_raw_{k}"] = train_input[k].astype(np.float32)
        valid_input_norm[f"_raw_{k}"] = valid_input[k].astype(np.float32)

    output_mean = train_output_dq.mean(axis=0, keepdims=True)
    output_std = np.maximum(train_output_dq.std(axis=0, keepdims=True), EPS)

    train_output_norm = (train_output_dq - output_mean) / output_std
    valid_output_norm = (valid_output_dq - output_mean) / output_std

    return train_input_norm, valid_input_norm, train_output_norm, valid_output_norm, input_stats, output_mean, output_std


def save_stats(path, input_stats, output_mean, output_std):
    payload = {
        "output_mean": output_mean.squeeze(0),
        "output_std": output_std.squeeze(0),
    }
    for k, stats in input_stats.items():
        payload[f"in_{k}_mean"] = stats["mean"].squeeze(0)
        payload[f"in_{k}_std"] = stats["std"].squeeze(0)
    np.savez(path, **payload)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_fps = sorted(glob(args.train_glob))
    valid_fps = sorted(glob(args.valid_glob))
    print(f"Found {len(train_fps)} training files and {len(valid_fps)} validation files.")

    if len(train_fps) == 0 or len(valid_fps) == 0:
        raise RuntimeError("Training/validation files not found. Check --train_glob and --valid_glob.")

    print("Loading training split...")
    train_input, train_output, train_penalty = load_split(train_fps, args.action_penalty_radius)
    print("Loading validation split...")
    valid_input, valid_output, valid_penalty = load_split(valid_fps, args.action_penalty_radius)

    print("train_input shapes:")
    for k, v in train_input.items():
        print(f"  {k}: {v.shape}")
    print("train_output shape:", train_output.shape)

    print("valid_input shapes:")
    for k, v in valid_input.items():
        print(f"  {k}: {v.shape}")
    print("valid_output shape:", valid_output.shape)
    if args.action_penalty_weight > 0.0:
        print(
            "action penalty active fraction: "
            f"train={(train_penalty > 0).mean():.4f}, "
            f"valid={(valid_penalty > 0).mean():.4f}"
        )

    (
        train_input_norm,
        valid_input_norm,
        train_output_norm,
        valid_output_norm,
        input_stats,
        output_mean,
        output_std,
    ) = normalize_data(train_input, valid_input, train_output, valid_output)

    os.makedirs(args.save_dir, exist_ok=True)
    stats_path = os.path.join(args.save_dir, args.stats_name)
    save_stats(stats_path, input_stats, output_mean, output_std)
    print(f"Saved normalization stats to {stats_path}")

    penalty_train = train_penalty if args.action_penalty_weight > 0.0 else None
    penalty_valid = valid_penalty if args.action_penalty_weight > 0.0 else None
    train_dataset = PickAndPlaceDataset(train_input_norm, train_output_norm, penalty_train)
    valid_dataset = PickAndPlaceDataset(valid_input_norm, valid_output_norm, penalty_valid)

    # Stage-weighted sampler: oversample grasp/place/release transitions.
    # WeightedRandomSampler replaces shuffle=True — do NOT set shuffle in DataLoader.
    sample_weights = torch.from_numpy(
        STAGE_SAMPLE_WEIGHTS[train_input["waypoint_idx"]]
    ).float()
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    input_dim = sum(
        v.shape[-1]
        for k, v in train_input_norm.items()
        if k != "waypoint_idx" and not k.startswith("_raw_")
    )
    output_dim = train_output_norm.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = POLICY(
        input_dim=input_dim,
        output_dim=output_dim,
        hid_dim=args.hid_dim,
        dropout=args.dropout,
        embed_dim=args.embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=5, min_lr=1e-6
    )
    loss_weights = torch.tensor([1.0] * 8, device=device)
    output_mean_t = torch.from_numpy(output_mean).float().to(device)
    output_std_t = torch.from_numpy(output_std).float().to(device)
    print(f"input_dim: {input_dim}, output_dim: {output_dim}, device: {device}")
    print(
        f"waypoint action penalty: weight={args.action_penalty_weight}, "
        f"radius={args.action_penalty_radius}"
    )

    best_valid_loss = float("inf")
    patience_counter = 0

    best_path = os.path.join(args.save_dir, args.best_ckpt_name)
    last_path = os.path.join(args.save_dir, args.last_ckpt_name)

    for epoch in range(args.max_epoch):
        model.train()
        train_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.max_epoch} - Training"
        ):
            if args.action_penalty_weight > 0.0:
                batch_inputs, batch_targets, batch_penalty = batch
                batch_penalty = batch_penalty.to(device)
            else:
                batch_inputs, batch_targets = batch
                batch_penalty = None

            batch_inputs = add_train_input_noise(batch_inputs, input_stats)
            for k in batch_inputs:
                batch_inputs[k] = batch_inputs[k].to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            if args.loss == "huber":
                loss = weighted_huber(predictions, batch_targets, loss_weights, beta=args.huber_beta)
            else:
                loss = weighted_mse(predictions, batch_targets, loss_weights)
            if args.action_penalty_weight > 0.0:
                loss = loss + args.action_penalty_weight * waypoint_action_penalty(
                    predictions,
                    output_mean_t,
                    output_std_t,
                    batch_penalty,
                )
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            train_loss += loss.item() * batch_targets.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                valid_loader, desc=f"Epoch {epoch + 1}/{args.max_epoch} - Validation"
            ):
                if args.action_penalty_weight > 0.0:
                    batch_inputs, batch_targets, batch_penalty = batch
                    batch_penalty = batch_penalty.to(device)
                else:
                    batch_inputs, batch_targets = batch
                    batch_penalty = None

                for k in batch_inputs:
                    batch_inputs[k] = batch_inputs[k].to(device)
                batch_targets = batch_targets.to(device)

                predictions = model(batch_inputs)
                if args.loss == "huber":
                    loss = weighted_huber(predictions, batch_targets, loss_weights, beta=args.huber_beta)
                else:
                    loss = weighted_mse(predictions, batch_targets, loss_weights)
                if args.action_penalty_weight > 0.0:
                    loss = loss + args.action_penalty_weight * waypoint_action_penalty(
                        predictions,
                        output_mean_t,
                        output_std_t,
                        batch_penalty,
                    )
                valid_loss += loss.item() * batch_targets.size(0)
        valid_loss /= len(valid_loader.dataset)
        scheduler.step(valid_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.max_epoch} - Train Loss: {train_loss:.7f} "
            f"- Valid Loss: {valid_loss:.7f} - LR: {current_lr:.2e}"
        )

        improved = valid_loss < (best_valid_loss - args.min_delta)
        if improved:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        torch.save(model.state_dict(), last_path)
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), 'model_epoch_{}.pth'.format(epoch + 1))

        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}. Best valid loss: {best_valid_loss:.7f}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student trainer script converted from baseline trainer")
    parser.add_argument("--train_glob", type=str, default="./train/pick_and_place_data_seed_*.npz")
    parser.add_argument("--valid_glob", type=str, default="./valid/pick_and_place_data_seed_*.npz")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--stats_name", type=str, default="student_stats.npz")
    parser.add_argument("--best_ckpt_name", type=str, default="student_best.pth")
    parser.add_argument("--last_ckpt_name", type=str, default="student_last.pth")
    parser.add_argument("--max_epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2**15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=1e-8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hid_dim", type=int, default=1024)
    parser.add_argument("--input_noise_std", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--huber_beta", type=float, default=1.0)
    parser.add_argument("--action_penalty_weight", type=float, default=0.0)
    parser.add_argument("--action_penalty_radius", type=float, default=0.08)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
