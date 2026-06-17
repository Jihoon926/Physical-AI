"""Binary waypoint-transition RNN plus MLP action policy.

The action head follows the current geometric waypoint. The GRU predicts only
whether to advance from the current waypoint to the next one.
"""

import numpy as np
import mujoco
import mediapy as media
import torch
import torch.nn as nn

from waypoint import build_waypoints


DEFAULT_INPUT_DIM = 53
DEFAULT_OUTPUT_DIM = 8
DERIVED_FEATURE_DIM = 9
ADVANCE_INPUT_DIM = 53
MAX_JOINT_DELTA = 0.06
GRIP_MIN = 0.0
GRIP_MAX = 255.0

ADVANCE_THRESHOLD = 0.5
MIN_ADVANCE_STEPS = 30
WAYPOINT_DISTANCE_THRESHOLDS = [0.018, 0.025, 0.025, 0.025, 0.025, 0.05, 0.05, 0.05]
GRIP_SWITCH_THRESHOLD = 128.0
WAYPOINT_STAY_THRESHOLD = 100
GRASP_RELEASE_STAY_THRESHOLD = 250

RAW_FEATURE_KEYS = ("curr_bpos", "hand_pos", "target_pos", "waypoint_pos")
ADVANCE_FEATURE_KEYS = (
    "block_size",
    "curr_bpos",
    "curr_bquat",
    "curr_qpos",
    "curr_qvel",
    "hand_pos",
    "hand_quat",
    "target_pos",
    "waypoint_grip",
    "waypoint_pos",
)


class POLICY(nn.Module):
    """Residual MLP action head with a GRU binary transition head."""

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        hid_dim: int = 1024,
        dropout: float = 0.1,
        embed_dim: int = 256,
        rnn_hidden_dim: int = 256,
        hidden_dim: int | None = None,
        derived_mean=None,
        derived_std=None,
    ):
        super().__init__()
        if hidden_dim is not None:
            hid_dim = hidden_dim
        if derived_mean is None:
            derived_mean = np.zeros((DERIVED_FEATURE_DIM,), dtype=np.float32)
        if derived_std is None:
            derived_std = np.ones((DERIVED_FEATURE_DIM,), dtype=np.float32)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.embed_dim = embed_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.register_buffer(
            "derived_mean",
            torch.as_tensor(derived_mean, dtype=torch.float32).view(1, DERIVED_FEATURE_DIM),
        )
        self.register_buffer(
            "derived_std",
            torch.clamp(
                torch.as_tensor(derived_std, dtype=torch.float32).view(1, DERIVED_FEATURE_DIM),
                min=1e-6,
            ),
        )

        self.waypoint_embed = nn.Embedding(8, embed_dim)

        self.advance_rnn = nn.GRU(
            input_size=ADVANCE_INPUT_DIM + embed_dim,
            hidden_size=rnn_hidden_dim,
            batch_first=True,
        )
        self.advance_head = nn.Linear(rnn_hidden_dim, 1)

        self.res_block1 = nn.Sequential(
            nn.Linear(input_dim + DERIVED_FEATURE_DIM + embed_dim, hid_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(hid_dim + embed_dim, hid_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.res_block3 = nn.Sequential(
            nn.Linear(hid_dim + embed_dim, hid_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.res_block4 = nn.Sequential(
            nn.Linear(hid_dim + embed_dim, hid_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.last_layer = nn.Linear(hid_dim, output_dim)

    @staticmethod
    def _raw(x, key):
        raw_key = f"_raw_{key}"
        return x[raw_key] if raw_key in x else x[key]

    @staticmethod
    def _ensure_sequence(tensor):
        return tensor.unsqueeze(1) if tensor.ndim == 2 else tensor

    def _waypoint_embedding_sequence(self, waypoint_idx):
        if waypoint_idx.ndim == 1:
            waypoint_idx = waypoint_idx.unsqueeze(1)
        return self.waypoint_embed(waypoint_idx)

    def _advance_features(self, x):
        features = [self._ensure_sequence(x[k]) for k in ADVANCE_FEATURE_KEYS]
        float_features = torch.cat(features, dim=-1)
        waypoint_embed = self._waypoint_embedding_sequence(x["waypoint_idx"])
        return torch.cat([float_features, waypoint_embed], dim=-1)

    def advance_logits(self, x, hidden=None):
        seq = self._advance_features(x)
        out, hidden = self.advance_rnn(seq, hidden)
        return self.advance_head(out).squeeze(-1), hidden

    def _derived_features(self, x):
        curr_bpos = self._raw(x, "curr_bpos")
        hand_pos = self._raw(x, "hand_pos")
        target_pos = self._raw(x, "target_pos")
        waypoint_pos = self._raw(x, "waypoint_pos")

        raw = torch.cat(
            (
                target_pos - curr_bpos,
                curr_bpos - hand_pos,
                waypoint_pos - hand_pos,
            ),
            dim=-1,
        )
        return (raw - self.derived_mean) / self.derived_std

    def _legacy_order_features(self, x):
        derived = self._derived_features(x)
        block_to_target = derived[..., 0:3]
        hand_to_block = derived[..., 3:6]
        hand_to_waypoint = derived[..., 6:9]

        return torch.cat(
            (
                x["block_size"],
                block_to_target,
                x["curr_bpos"],
                x["curr_bquat"],
                x["curr_qpos"],
                x["curr_qvel"],
                x["hand_pos"],
                x["hand_quat"],
                hand_to_block,
                hand_to_waypoint,
                x["target_pos"],
                x["waypoint_grip"],
                x["waypoint_pos"],
            ),
            dim=-1,
        )

    def forward(self, x):
        waypoint_embed = self.waypoint_embed(x["waypoint_idx"])
        float_feat = self._legacy_order_features(x)

        out = self.res_block1(torch.cat([float_feat, waypoint_embed], dim=-1))
        out = self.res_block2(torch.cat([out, waypoint_embed], dim=-1)) + out
        out = self.res_block3(torch.cat([out, waypoint_embed], dim=-1)) + out
        out = self.res_block4(torch.cat([out, waypoint_embed], dim=-1)) + out
        return self.last_layer(out)


def build_policy(
    checkpoint_path: str | None,
    device: torch.device,
    input_dim: int = DEFAULT_INPUT_DIM,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    hid_dim: int = 1024,
    dropout: float = 0.1,
    embed_dim: int = 256,
    rnn_hidden_dim: int = 256,
    hidden_dim: int | None = None,
):
    if hidden_dim is not None:
        hid_dim = hidden_dim

    model = POLICY(
        input_dim=input_dim,
        output_dim=output_dim,
        hid_dim=hid_dim,
        dropout=dropout,
        embed_dim=embed_dim,
        rnn_hidden_dim=rnn_hidden_dim,
    ).to(device)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    model.eval()
    return model


def _normalize_input(model_input, stats, device):
    if stats is None:
        return model_input

    normalized = {}
    for key in RAW_FEATURE_KEYS:
        if key in model_input:
            normalized[f"_raw_{key}"] = model_input[key]

    for k in sorted(model_input.keys()):
        if k == "waypoint_idx":
            normalized[k] = model_input[k]
            continue

        mean_key = f"in_{k}_mean"
        std_key = f"in_{k}_std"
        if mean_key not in stats or std_key not in stats:
            raise KeyError(f"Missing normalization stats for key: {k}")

        mean = torch.from_numpy(np.asarray(stats[mean_key])).float().unsqueeze(0).to(device)
        std = torch.from_numpy(np.asarray(stats[std_key])).float().unsqueeze(0).to(device)
        normalized[k] = (model_input[k] - mean) / std
    return normalized


def _denormalize_output(pred, stats):
    if stats is None:
        return pred
    if "output_std" not in stats or "output_mean" not in stats:
        return pred
    return pred * np.asarray(stats["output_std"]) + np.asarray(stats["output_mean"])


def _build_model_input(
    curr_qpos,
    curr_qvel,
    curr_bpos,
    curr_bquat,
    curr_hand_pos,
    curr_hand_quat,
    block_size,
    target_pos,
    curr_waypoint,
    waypoint_index,
    device,
):
    return {
        "block_size": torch.tensor([[block_size]], dtype=torch.float32).to(device),
        "curr_bpos": torch.from_numpy(curr_bpos[None, :]).float().to(device),
        "curr_bquat": torch.from_numpy(curr_bquat[None, :]).float().to(device),
        "curr_qpos": torch.from_numpy(curr_qpos[None, :]).float().to(device),
        "curr_qvel": torch.from_numpy(curr_qvel[None, :]).float().to(device),
        "hand_pos": torch.from_numpy(curr_hand_pos[None, :]).float().to(device),
        "hand_quat": torch.from_numpy(curr_hand_quat[None, :]).float().to(device),
        "target_pos": torch.from_numpy(target_pos[None, :]).float().to(device),
        "waypoint_grip": torch.tensor([[curr_waypoint.grip]], dtype=torch.float32).to(device),
        "waypoint_idx": torch.tensor([waypoint_index], dtype=torch.long).to(device),
        "waypoint_pos": torch.from_numpy(curr_waypoint.pos[None, :]).float().to(device),
    }


def _advance_waypoint_by_geometry(
    curr_waypoint,
    waypoint_index,
    waypoints,
    dist_curr_waypoint,
    pred_grip,
    stay_counter,
    grasp_counter,
):
    if curr_waypoint.name == "grasp":
        if dist_curr_waypoint < WAYPOINT_DISTANCE_THRESHOLDS[waypoint_index] and waypoint_index < len(waypoints) - 1:
            if pred_grip < GRIP_SWITCH_THRESHOLD:
                grasp_counter += 1
            if grasp_counter > GRASP_RELEASE_STAY_THRESHOLD:
                waypoint_index += 1
                curr_waypoint = waypoints[waypoint_index]
                grasp_counter = 0

    elif curr_waypoint.name == "release":
        if dist_curr_waypoint < WAYPOINT_DISTANCE_THRESHOLDS[waypoint_index] and waypoint_index < len(waypoints) - 1:
            if pred_grip > GRIP_SWITCH_THRESHOLD:
                grasp_counter += 1
            if grasp_counter > GRASP_RELEASE_STAY_THRESHOLD:
                waypoint_index += 1
                curr_waypoint = waypoints[waypoint_index]
                grasp_counter = 0

    else:
        stay_counter += 1
        if stay_counter > WAYPOINT_STAY_THRESHOLD:
            if dist_curr_waypoint < WAYPOINT_DISTANCE_THRESHOLDS[waypoint_index] and waypoint_index < len(waypoints) - 1:
                waypoint_index += 1
                curr_waypoint = waypoints[waypoint_index]
                stay_counter = 0

    return waypoint_index, curr_waypoint, stay_counter, grasp_counter


def run_single_episode(
    policy,
    model,
    data,
    ids,
    env_data,
    stats,
    device,
    max_steps: int = 5000,
    render: bool = False,
    render_fps: int = 60,
    success_distance: float = 0.05,
):
    block_size = float(env_data["block_size"])
    block_pos = np.asarray(env_data["block_init_xpos"], dtype=float)
    target_pos = np.asarray(env_data["target_xpos"], dtype=float)

    waypoints = build_waypoints(block_pos, target_pos)
    waypoint_index = 0
    curr_waypoint = waypoints[waypoint_index]
    hidden = None
    stay_counter = 0
    grasp_counter = 0
    stage_steps = 0

    frames = []
    renderer = mujoco.Renderer(model, width=640, height=480) if render else None
    steps_taken = 0

    try:
        for step in range(max_steps):
            steps_taken = step + 1

            curr_qpos = data.qpos.copy()
            curr_qvel = data.qvel.copy()
            curr_bpos = data.xpos[ids["block_body_id"]].copy()
            curr_bquat = data.xquat[ids["block_body_id"]].copy()
            curr_hand_pos = data.xpos[ids["hand_body_id"]].copy()
            curr_hand_quat = data.xquat[ids["hand_body_id"]].copy()

            transition_input = _build_model_input(
                curr_qpos,
                curr_qvel,
                curr_bpos,
                curr_bquat,
                curr_hand_pos,
                curr_hand_quat,
                block_size,
                target_pos,
                curr_waypoint,
                waypoint_index,
                device,
            )
            transition_input = _normalize_input(transition_input, stats, device)
            with torch.no_grad():
                logits, hidden = policy.advance_logits(transition_input, hidden)
                advance_prob = float(torch.sigmoid(logits[:, -1]).item())

            advanced_by_rnn = False
            if (
                advance_prob > ADVANCE_THRESHOLD
                and stage_steps > MIN_ADVANCE_STEPS
                and waypoint_index < len(waypoints) - 1
            ):
                waypoint_index += 1
                curr_waypoint = waypoints[waypoint_index]
                stay_counter = 0
                grasp_counter = 0
                stage_steps = 0
                advanced_by_rnn = True

            action_input = _build_model_input(
                curr_qpos,
                curr_qvel,
                curr_bpos,
                curr_bquat,
                curr_hand_pos,
                curr_hand_quat,
                block_size,
                target_pos,
                curr_waypoint,
                waypoint_index,
                device,
            )
            action_input = _normalize_input(action_input, stats, device)
            with torch.no_grad():
                pred = policy(action_input).detach().cpu().numpy()[0]

            pred = _denormalize_output(pred, stats)
            joint_delta = np.clip(pred[:7], -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
            grip_cmd = float(np.clip(pred[7], GRIP_MIN, GRIP_MAX))

            if not advanced_by_rnn:
                old_index = waypoint_index
                dist_curr_waypoint = np.linalg.norm(curr_hand_pos - curr_waypoint.pos)
                waypoint_index, curr_waypoint, stay_counter, grasp_counter = _advance_waypoint_by_geometry(
                    curr_waypoint,
                    waypoint_index,
                    waypoints,
                    dist_curr_waypoint,
                    grip_cmd,
                    stay_counter,
                    grasp_counter,
                )
                if waypoint_index != old_index:
                    stage_steps = 0

            data.ctrl[:7] = curr_qpos[:7] + joint_delta
            data.ctrl[7] = grip_cmd
            mujoco.mj_step(model, data)
            stage_steps += 1

            if render and len(frames) < data.time * render_fps:
                renderer.update_scene(data, camera=ids["task_cam_id"])
                frames.append(renderer.render())

            if waypoint_index == len(waypoints) - 1:
                curr_bpos_after = data.xpos[ids["block_body_id"]].copy()
                if np.linalg.norm(curr_bpos_after - target_pos) < success_distance:
                    break

        if render and len(frames) > 0:
            media.show_video(frames, fps=render_fps)

        return {
            "num_steps": int(steps_taken),
            "final_waypoint_index": int(waypoint_index),
            "final_waypoint_name": curr_waypoint.name,
        }
    finally:
        if renderer is not None:
            renderer.close()
