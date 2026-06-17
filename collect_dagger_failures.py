"""Collect failed rollout states and relabel them with an IK expert action."""

import argparse
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from evaluate_policy import TARGET_RADIUS, build_model_data_for_env, summarize
from ik import solve_ik
from waypoint import build_waypoints


MAX_JOINT_DELTA = 0.06
GRIP_MIN = 0.0
GRIP_MAX = 255.0


def load_policy_module(policy_file: str):
    policy_path = Path(policy_file).resolve()
    spec = importlib.util.spec_from_file_location("dagger_source_policy", str(policy_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_policy_from_module(module, checkpoint_file: str, device):
    return module.build_policy(checkpoint_file, device)


def expert_ctrl_for_waypoint(model, data, ids, waypoint, fixed_quat, q_warmstart):
    q_target, _, _, _ = solve_ik(
        model,
        data,
        target_pos=waypoint.pos,
        target_quat=fixed_quat,
        body_id=ids["hand_body_id"],
        dof_ids=np.arange(7),
        q_init=q_warmstart,
        max_iters=80,
        step_scale=0.5,
        damping=0.01,
        rot_w=0.001,
    )
    ctrl = np.zeros(8, dtype=np.float32)
    delta = np.clip(q_target - data.qpos[:7], -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
    ctrl[:7] = data.qpos[:7] + delta
    ctrl[7] = waypoint.grip

    next_warmstart = data.qpos.copy()
    next_warmstart[:7] = q_target
    return ctrl, next_warmstart


def append_history(history, data, ids, expert_ctrl, waypoint):
    history["t"].append(float(data.time))
    history["qpos"].append(data.qpos.copy())
    history["qvel"].append(data.qvel.copy())
    history["ctrl"].append(expert_ctrl.copy())
    history["bpos"].append(data.xpos[ids["block_body_id"]].copy())
    history["bquat"].append(data.xquat[ids["block_body_id"]].copy())
    history["hpos"].append(data.xpos[ids["hand_body_id"]].copy())
    history["hquat"].append(data.xquat[ids["hand_body_id"]].copy())
    history["waypoint_pos"].append(waypoint.pos.copy())
    history["waypoint_grip"].append(float(waypoint.grip))


def finalize_history(history, env_data):
    out = {k: np.asarray(v) for k, v in history.items()}
    out["block_init_xpos"] = np.asarray(env_data["block_init_xpos"], dtype=np.float32)
    out["block_init_quat"] = np.asarray(env_data["block_init_quat"], dtype=np.float32)
    out["block_size"] = np.asarray(env_data["block_size"], dtype=np.float32)
    out["target_xpos"] = np.asarray(env_data["target_xpos"], dtype=np.float32)
    return out


def read_env_list(list_path):
    env_paths = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            env_paths.append(line)
    return env_paths


def rollout_and_collect(module, policy, stats, env_path, device, max_steps):
    model, data, ids, env_data = build_model_data_for_env(env_path)
    block_size = float(env_data["block_size"])
    block_pos = np.asarray(env_data["block_init_xpos"], dtype=float)
    target_pos = np.asarray(env_data["target_xpos"], dtype=float)
    success_distance = TARGET_RADIUS + block_size

    waypoints = build_waypoints(block_pos, target_pos)
    waypoint_index = 0
    curr_waypoint = waypoints[waypoint_index]
    hidden = None
    stay_counter = 0
    grasp_counter = 0
    stage_steps = 0
    fixed_quat = data.xquat[ids["hand_body_id"]].copy()
    q_warmstart = data.qpos.copy()

    history = {
        "t": [],
        "qpos": [],
        "qvel": [],
        "ctrl": [],
        "bpos": [],
        "bquat": [],
        "hpos": [],
        "hquat": [],
        "waypoint_pos": [],
        "waypoint_grip": [],
    }

    steps_taken = 0
    for step in range(max_steps):
        steps_taken = step + 1
        curr_qpos = data.qpos.copy()
        curr_qvel = data.qvel.copy()
        curr_bpos = data.xpos[ids["block_body_id"]].copy()
        curr_bquat = data.xquat[ids["block_body_id"]].copy()
        curr_hand_pos = data.xpos[ids["hand_body_id"]].copy()
        curr_hand_quat = data.xquat[ids["hand_body_id"]].copy()

        stage_input = module._build_model_input(
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
        stage_input = module._normalize_input(stage_input, stats, device)
        advanced_by_policy = False
        with torch.no_grad():
            if hasattr(policy, "stage_logits"):
                logits, hidden = policy.stage_logits(stage_input, hidden)
                predicted_stage = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                if predicted_stage > waypoint_index:
                    waypoint_index = min(waypoint_index + 1, predicted_stage, len(waypoints) - 1)
                    curr_waypoint = waypoints[waypoint_index]
                    stay_counter = 0
                    grasp_counter = 0
                    stage_steps = 0
                    advanced_by_policy = True
            elif hasattr(policy, "advance_logits"):
                logits, hidden = policy.advance_logits(stage_input, hidden)
                advance_prob = float(torch.sigmoid(logits[:, -1]).item())
                advance_threshold = getattr(module, "ADVANCE_THRESHOLD", 0.5)
                min_advance_steps = getattr(module, "MIN_ADVANCE_STEPS", 0)
                if (
                    advance_prob > advance_threshold
                    and stage_steps > min_advance_steps
                    and waypoint_index < len(waypoints) - 1
                ):
                    waypoint_index += 1
                    curr_waypoint = waypoints[waypoint_index]
                    stay_counter = 0
                    grasp_counter = 0
                    stage_steps = 0
                    advanced_by_policy = True

        expert_ctrl, q_warmstart = expert_ctrl_for_waypoint(
            model,
            data,
            ids,
            curr_waypoint,
            fixed_quat,
            q_warmstart,
        )
        append_history(history, data, ids, expert_ctrl, curr_waypoint)

        action_input = module._build_model_input(
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
        action_input = module._normalize_input(action_input, stats, device)
        with torch.no_grad():
            pred = policy(action_input).detach().cpu().numpy()[0]

        pred = module._denormalize_output(pred, stats)
        joint_delta = np.clip(pred[:7], -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
        grip_cmd = float(np.clip(pred[7], GRIP_MIN, GRIP_MAX))

        if not advanced_by_policy and hasattr(module, "_advance_waypoint_by_geometry"):
            old_index = waypoint_index
            dist_curr_waypoint = np.linalg.norm(curr_hand_pos - curr_waypoint.pos)
            waypoint_index, curr_waypoint, stay_counter, grasp_counter = module._advance_waypoint_by_geometry(
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

        if waypoint_index == len(waypoints) - 1:
            curr_bpos_after = data.xpos[ids["block_body_id"]].copy()
            if np.linalg.norm(curr_bpos_after - target_pos) < success_distance:
                break

    final_block_pos = data.xpos[ids["block_body_id"]].copy()
    final_dist = float(np.linalg.norm(final_block_pos - target_pos))
    success = final_dist < success_distance
    return {
        "env_path": str(env_path),
        "env_name": Path(env_path).name,
        "success": bool(success),
        "final_distance": final_dist,
        "success_distance": float(success_distance),
        "num_steps": int(steps_taken),
        "final_waypoint_index": int(waypoint_index),
        "final_waypoint_name": curr_waypoint.name,
        "history": finalize_history(history, env_data),
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    module = load_policy_module(args.policy_file)
    policy = build_policy_from_module(module, args.checkpoint_file, device)
    stats = np.load(args.stats_file) if args.stats_file else None

    if args.env_list:
        env_paths = read_env_list(args.env_list)
    else:
        env_paths = sorted(str(p) for p in Path(args.env_dir).glob("*.npz"))
    if args.max_envs is not None:
        env_paths = env_paths[: args.max_envs]
    if not env_paths:
        raise RuntimeError(f"No environment files found from {args.env_list or args.env_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    results = []
    failure_count = 0

    for i, env_path in enumerate(tqdm(env_paths, desc="Collecting DAgger failures")):
        outcome = rollout_and_collect(module, policy, stats, env_path, device, args.max_steps)
        result = {k: v for k, v in outcome.items() if k != "history"}
        results.append(result)
        print(
            f"Env {i + 1}/{len(env_paths)}: {Path(env_path).name} -> "
            f"success={outcome['success']}, final_dist={outcome['final_distance']:.4f} m",
            flush=True,
        )

        if not outcome["success"]:
            failure_count += 1
            seed_name = Path(env_path).stem
            save_path = output_dir / f"dagger_failure_{failure_count:03d}_{seed_name}.npz"
            np.savez(save_path, **outcome["history"])
            saved.append(str(save_path))
            print(f"  saved {save_path}", flush=True)
            if args.max_failures is not None and failure_count >= args.max_failures:
                break

    summary = summarize(results)
    payload = {
        "policy_file": args.policy_file,
        "checkpoint_file": args.checkpoint_file,
        "stats_file": args.stats_file,
        "env_dir": args.env_dir,
        "env_list": args.env_list,
        "max_steps": args.max_steps,
        "output_dir": str(output_dir),
        "saved_failure_files": saved,
        "summary": summary,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Saved failures: {len(saved)}")
    print(f"Success: {summary['success_count']}/{summary['count']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_file", default="rnn_policy.py")
    parser.add_argument("--checkpoint_file", default="rnn_stage_best.pth")
    parser.add_argument("--stats_file", default="rnn_stage_stats.npz")
    parser.add_argument("--env_dir", default="./open_env")
    parser.add_argument("--env_list", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_envs", type=int, default=50)
    parser.add_argument("--max_failures", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
