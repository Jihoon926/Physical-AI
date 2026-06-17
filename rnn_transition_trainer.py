import argparse
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from rnn_transition_policy import POLICY
from student_trainer import infer_waypoint_index, weighted_huber, weighted_mse


EPS = 1e-6
RAW_FEATURE_KEYS = ("curr_bpos", "hand_pos", "target_pos", "waypoint_pos")


class SequenceDataset(Dataset):
    def __init__(
        self,
        input_data,
        output_data,
        waypoint_idx,
        advance_targets,
        seq_len,
        seq_stride,
        max_windows=None,
        seed=42,
    ):
        self.input_data = input_data
        self.output_data = output_data
        self.waypoint_idx = waypoint_idx
        self.advance_targets = advance_targets
        self.seq_len = seq_len

        num_traj, traj_len = output_data.shape[:2]
        windows = []
        for traj_idx in range(num_traj):
            for start in range(0, traj_len - seq_len + 1, seq_stride):
                windows.append((traj_idx, start))

        if max_windows is not None and len(windows) > max_windows:
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(windows), size=max_windows, replace=False)
            windows = [windows[i] for i in sorted(keep)]

        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        traj_idx, start = self.windows[idx]
        end = start + self.seq_len
        inputs = {}
        for k, v in self.input_data.items():
            if k == "waypoint_idx":
                inputs[k] = torch.from_numpy(v[traj_idx, start:end]).long()
            else:
                inputs[k] = torch.from_numpy(v[traj_idx, start:end]).float()
        targets = torch.from_numpy(self.output_data[traj_idx, start:end]).float()
        advance_targets = torch.from_numpy(self.advance_targets[traj_idx, start:end]).float()
        return inputs, targets, advance_targets


def load_split(file_paths):
    input_data = {
        "curr_qpos": [],
        "curr_qvel": [],
        "curr_bpos": [],
        "curr_bquat": [],
        "target_pos": [],
        "block_size": [],
        "hand_pos": [],
        "hand_quat": [],
        "waypoint_pos": [],
        "waypoint_grip": [],
        "waypoint_idx": [],
    }
    output_data = []

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
        input_data["block_size"].append(np.asarray(block_size)[None, None].repeat(len(ctrl), axis=0))
        input_data["target_pos"].append(target_pos[None, :].repeat(len(ctrl), axis=0))
        input_data["hand_pos"].append(hand_pos)
        input_data["hand_quat"].append(hand_quat)
        output_data.append(ctrl)

        waypoint_idx = infer_waypoint_index(
            waypoint_pos,
            waypoint_grip,
            traj_data["block_init_xpos"],
            target_pos,
        )
        input_data["waypoint_idx"].append(waypoint_idx)

    input_data = {k: np.stack(v, axis=0) for k, v in sorted(input_data.items())}
    output_data = np.stack(output_data, axis=0)
    return input_data, output_data


def flatten_time(arr):
    return arr.reshape((-1, arr.shape[-1]))


def make_derived_features(input_data):
    return np.concatenate(
        (
            input_data["target_pos"] - input_data["curr_bpos"],
            input_data["curr_bpos"] - input_data["hand_pos"],
            input_data["waypoint_pos"] - input_data["hand_pos"],
        ),
        axis=-1,
    ).astype(np.float32)


def make_advance_targets(waypoint_idx, positive_window=1):
    targets = np.zeros_like(waypoint_idx, dtype=np.float32)
    transitions = waypoint_idx[:, 1:] > waypoint_idx[:, :-1]
    for traj_idx in range(waypoint_idx.shape[0]):
        transition_steps = np.where(transitions[traj_idx])[0]
        for step in transition_steps:
            start = max(0, step - positive_window + 1)
            targets[traj_idx, start : step + 1] = 1.0
    return targets


def normalize_data(train_input, valid_input, train_output, valid_output):
    train_output_dq = train_output.copy()
    valid_output_dq = valid_output.copy()
    train_output_dq[:, :, :7] = train_output_dq[:, :, :7] - train_input["curr_qpos"][:, :, :7]
    valid_output_dq[:, :, :7] = valid_output_dq[:, :, :7] - valid_input["curr_qpos"][:, :, :7]

    input_stats = {}
    train_input_norm = {}
    valid_input_norm = {}

    for k in sorted(train_input.keys()):
        if k == "waypoint_idx":
            train_input_norm[k] = train_input[k]
            valid_input_norm[k] = valid_input[k]
            continue
        flat = flatten_time(train_input[k])
        mean = flat.mean(axis=0, keepdims=True)
        std = np.maximum(flat.std(axis=0, keepdims=True), EPS)
        input_stats[k] = {"mean": mean, "std": std}
        train_input_norm[k] = (train_input[k] - mean[None, :, :]) / std[None, :, :]
        valid_input_norm[k] = (valid_input[k] - mean[None, :, :]) / std[None, :, :]

    for k in RAW_FEATURE_KEYS:
        train_input_norm[f"_raw_{k}"] = train_input[k].astype(np.float32)
        valid_input_norm[f"_raw_{k}"] = valid_input[k].astype(np.float32)

    train_derived = make_derived_features(train_input)
    derived_flat = flatten_time(train_derived)
    derived_mean = derived_flat.mean(axis=0)
    derived_std = np.maximum(derived_flat.std(axis=0), EPS)

    output_flat = flatten_time(train_output_dq)
    output_mean = output_flat.mean(axis=0, keepdims=True)
    output_std = np.maximum(output_flat.std(axis=0, keepdims=True), EPS)
    train_output_norm = (train_output_dq - output_mean[None, :, :]) / output_std[None, :, :]
    valid_output_norm = (valid_output_dq - output_mean[None, :, :]) / output_std[None, :, :]

    return (
        train_input_norm,
        valid_input_norm,
        train_output_norm,
        valid_output_norm,
        input_stats,
        output_mean,
        output_std,
        derived_mean,
        derived_std,
    )


def save_stats(path, input_stats, output_mean, output_std, derived_mean, derived_std):
    payload = {
        "output_mean": output_mean.squeeze(0),
        "output_std": output_std.squeeze(0),
        "derived_mean": derived_mean,
        "derived_std": derived_std,
    }
    for k, stats in input_stats.items():
        payload[f"in_{k}_mean"] = stats["mean"].squeeze(0)
        payload[f"in_{k}_std"] = stats["std"].squeeze(0)
    np.savez(path, **payload)


def move_batch_to_device(batch_inputs, device):
    for k in batch_inputs:
        batch_inputs[k] = batch_inputs[k].to(device)
    return batch_inputs


def load_matching_tensors(model, checkpoint_path, device):
    source_state = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    loaded = []
    skipped = []
    for key, value in source_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    model.load_state_dict(model_state)
    print(f"Initialized {len(loaded)} matching tensors from {checkpoint_path}.")
    if skipped:
        print(f"Skipped {len(skipped)} non-matching tensors.")


def freeze_action_head(model):
    trainable_prefixes = ("advance_rnn", "advance_head")
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if name.startswith(trainable_prefixes):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    print(f"Frozen action-head parameters: {frozen:,}; trainable transition parameters: {trainable:,}")


def read_path_list(list_path):
    paths = []
    if not list_path:
        return paths
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(line)
    return paths


def resolve_paths(*patterns, list_paths=None):
    paths = []
    for pattern in patterns:
        if not pattern:
            continue
        paths.extend(glob(pattern))
    for list_path in list_paths or []:
        paths.extend(read_path_list(list_path))
    return sorted(set(paths))


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    extra_train_globs = args.extra_train_glob or []
    extra_valid_globs = args.extra_valid_glob or []
    train_lists = args.train_list or []
    valid_lists = args.valid_list or []

    train_fps = resolve_paths(
        args.train_glob,
        *extra_train_globs,
        args.dagger_glob,
        list_paths=train_lists,
    )
    valid_fps = resolve_paths(
        args.valid_glob,
        *extra_valid_globs,
        list_paths=valid_lists,
    )
    dagger_count = len(resolve_paths(args.dagger_glob))
    extra_train_count = len(resolve_paths(*extra_train_globs, list_paths=train_lists))
    extra_valid_count = len(resolve_paths(*extra_valid_globs, list_paths=valid_lists))
    print(
        f"Found {len(train_fps)} training files "
        f"({dagger_count} dagger, {extra_train_count} extra train) and "
        f"{len(valid_fps)} validation files ({extra_valid_count} extra valid)."
    )
    if len(train_fps) == 0 or len(valid_fps) == 0:
        raise RuntimeError("Training/validation files not found. Check globs.")

    print("Loading training split...")
    train_input, train_output = load_split(train_fps)
    print("Loading validation split...")
    valid_input, valid_output = load_split(valid_fps)

    print("train trajectories:", train_output.shape)
    print("valid trajectories:", valid_output.shape)

    train_advance = make_advance_targets(train_input["waypoint_idx"], args.advance_positive_window)
    valid_advance = make_advance_targets(valid_input["waypoint_idx"], args.advance_positive_window)
    train_pos = float(train_advance.sum())
    train_total = float(train_advance.size)
    pos_weight_value = min((train_total - train_pos) / max(train_pos, 1.0), args.max_pos_weight)
    print(
        f"advance positives: {int(train_pos)}/{int(train_total)} "
        f"({train_pos / train_total:.6f}), pos_weight={pos_weight_value:.3f}"
    )

    (
        train_input_norm,
        valid_input_norm,
        train_output_norm,
        valid_output_norm,
        input_stats,
        output_mean,
        output_std,
        derived_mean,
        derived_std,
    ) = normalize_data(train_input, valid_input, train_output, valid_output)

    os.makedirs(args.save_dir, exist_ok=True)
    stats_path = os.path.join(args.save_dir, args.stats_name)
    save_stats(stats_path, input_stats, output_mean, output_std, derived_mean, derived_std)
    print(f"Saved normalization stats to {stats_path}")

    train_dataset = SequenceDataset(
        train_input_norm,
        train_output_norm,
        train_input["waypoint_idx"],
        train_advance,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        max_windows=args.max_train_windows,
        seed=args.seed,
    )
    valid_dataset = SequenceDataset(
        valid_input_norm,
        valid_output_norm,
        valid_input["waypoint_idx"],
        valid_advance,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        max_windows=args.max_valid_windows,
        seed=args.seed + 1,
    )
    print(f"train windows: {len(train_dataset)}, valid windows: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = POLICY(
        hid_dim=args.hid_dim,
        dropout=args.dropout,
        embed_dim=args.embed_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        derived_mean=derived_mean,
        derived_std=derived_std,
    ).to(device)
    if args.init_action_ckpt:
        load_matching_tensors(model, args.init_action_ckpt, device)
    if args.freeze_action_head:
        freeze_action_head(model)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=5, min_lr=1e-6
    )
    loss_weights = torch.tensor([1.0] * 8, device=device)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    print(f"device: {device}, seq_len: {args.seq_len}, batch_size: {args.batch_size}")

    best_valid_loss = float("inf")
    best_path = os.path.join(args.save_dir, args.best_ckpt_name)
    last_path = os.path.join(args.save_dir, args.last_ckpt_name)

    for epoch in range(args.max_epoch):
        model.train()
        train_loss = 0.0
        train_action_loss = 0.0
        train_adv_loss = 0.0
        train_adv_correct = 0
        train_count = 0
        train_pred_pos = 0
        train_true_pos = 0

        for batch_inputs, batch_targets, batch_advance in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.max_epoch} - Training"
        ):
            batch_inputs = move_batch_to_device(batch_inputs, device)
            batch_targets = batch_targets.to(device)
            batch_advance = batch_advance.to(device)

            optimizer.zero_grad()
            action_pred = model(batch_inputs)
            advance_logits, _ = model.advance_logits(batch_inputs)
            action_loss = weighted_huber(
                action_pred.reshape(-1, 8),
                batch_targets.reshape(-1, 8),
                loss_weights,
                beta=args.huber_beta,
            ) if args.loss == "huber" else weighted_mse(
                action_pred.reshape(-1, 8),
                batch_targets.reshape(-1, 8),
                loss_weights,
            )
            adv_loss = F.binary_cross_entropy_with_logits(
                advance_logits.reshape(-1),
                batch_advance.reshape(-1),
                pos_weight=pos_weight,
            )
            loss = args.action_loss_weight * action_loss + args.advance_loss_weight * adv_loss
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            count = batch_targets.shape[0] * batch_targets.shape[1]
            pred_adv = (torch.sigmoid(advance_logits) > 0.5)
            true_adv = batch_advance > 0.5
            train_loss += loss.item() * count
            train_action_loss += action_loss.item() * count
            train_adv_loss += adv_loss.item() * count
            train_adv_correct += (pred_adv == true_adv).sum().item()
            train_pred_pos += pred_adv.sum().item()
            train_true_pos += true_adv.sum().item()
            train_count += count

        train_loss /= train_count
        train_action_loss /= train_count
        train_adv_loss /= train_count
        train_adv_acc = train_adv_correct / train_count

        model.eval()
        valid_loss = 0.0
        valid_action_loss = 0.0
        valid_adv_loss = 0.0
        valid_adv_correct = 0
        valid_count = 0
        valid_pred_pos = 0
        valid_true_pos = 0
        with torch.no_grad():
            for batch_inputs, batch_targets, batch_advance in tqdm(
                valid_loader, desc=f"Epoch {epoch + 1}/{args.max_epoch} - Validation"
            ):
                batch_inputs = move_batch_to_device(batch_inputs, device)
                batch_targets = batch_targets.to(device)
                batch_advance = batch_advance.to(device)

                action_pred = model(batch_inputs)
                advance_logits, _ = model.advance_logits(batch_inputs)
                action_loss = weighted_huber(
                    action_pred.reshape(-1, 8),
                    batch_targets.reshape(-1, 8),
                    loss_weights,
                    beta=args.huber_beta,
                ) if args.loss == "huber" else weighted_mse(
                    action_pred.reshape(-1, 8),
                    batch_targets.reshape(-1, 8),
                    loss_weights,
                )
                adv_loss = F.binary_cross_entropy_with_logits(
                    advance_logits.reshape(-1),
                    batch_advance.reshape(-1),
                    pos_weight=pos_weight,
                )
                loss = args.action_loss_weight * action_loss + args.advance_loss_weight * adv_loss

                count = batch_targets.shape[0] * batch_targets.shape[1]
                pred_adv = (torch.sigmoid(advance_logits) > 0.5)
                true_adv = batch_advance > 0.5
                valid_loss += loss.item() * count
                valid_action_loss += action_loss.item() * count
                valid_adv_loss += adv_loss.item() * count
                valid_adv_correct += (pred_adv == true_adv).sum().item()
                valid_pred_pos += pred_adv.sum().item()
                valid_true_pos += true_adv.sum().item()
                valid_count += count

        valid_loss /= valid_count
        valid_action_loss /= valid_count
        valid_adv_loss /= valid_count
        valid_adv_acc = valid_adv_correct / valid_count
        scheduler.step(valid_loss)

        print(
            f"Epoch {epoch + 1}/{args.max_epoch} - "
            f"Train Loss: {train_loss:.7f} "
            f"(action {train_action_loss:.7f}, advance {train_adv_loss:.7f}, "
            f"acc {train_adv_acc:.4f}, pred+ {train_pred_pos / train_count:.5f}, "
            f"true+ {train_true_pos / train_count:.5f}) "
            f"- Valid Loss: {valid_loss:.7f} "
            f"(action {valid_action_loss:.7f}, advance {valid_adv_loss:.7f}, "
            f"acc {valid_adv_acc:.4f}, pred+ {valid_pred_pos / valid_count:.5f}, "
            f"true+ {valid_true_pos / valid_count:.5f}) "
            f"- LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if valid_loss < best_valid_loss - args.min_delta:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_path)
        torch.save(model.state_dict(), last_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train binary waypoint-transition RNN policy.")
    parser.add_argument("--train_glob", type=str, default="./train/pick_and_place_data_seed_*.npz")
    parser.add_argument("--dagger_glob", type=str, default=None)
    parser.add_argument("--valid_glob", type=str, default="./valid/pick_and_place_data_seed_*.npz")
    parser.add_argument("--extra_train_glob", action="append", default=None)
    parser.add_argument("--extra_valid_glob", action="append", default=None)
    parser.add_argument("--train_list", action="append", default=None)
    parser.add_argument("--valid_list", action="append", default=None)
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--stats_name", type=str, default="rnn_transition_stats.npz")
    parser.add_argument("--best_ckpt_name", type=str, default="rnn_transition_best.pth")
    parser.add_argument("--last_ckpt_name", type=str, default="rnn_transition_last.pth")
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--seq_stride", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_valid_windows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hid_dim", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--rnn_hidden_dim", type=int, default=256)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--huber_beta", type=float, default=1.0)
    parser.add_argument("--action_loss_weight", type=float, default=1.0)
    parser.add_argument("--advance_loss_weight", type=float, default=0.05)
    parser.add_argument("--advance_positive_window", type=int, default=3)
    parser.add_argument("--max_pos_weight", type=float, default=50.0)
    parser.add_argument("--init_action_ckpt", type=str, default=None)
    parser.add_argument("--freeze_action_head", action="store_true")
    parser.add_argument("--min_delta", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
