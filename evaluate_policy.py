"""Evaluate a policy module on pick-and-place environment files.

This mirrors the submission tester's scene construction and success criterion,
but writes machine-readable JSON so long experiments are easier to track.
"""

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

import mujoco
import numpy as np
import torch
from robot_descriptions import panda_mj_description
from tqdm.auto import tqdm

from helper import reset_to_keyframe


TARGET_RADIUS = 0.05


def load_policy_module(policy_file: str):
    policy_path = Path(policy_file).resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    spec = importlib.util.spec_from_file_location("evaluated_policy", str(policy_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_policy_from_module(module, checkpoint_file: str | None, device: torch.device):
    checkpoint_path = Path(checkpoint_file).resolve() if checkpoint_file else None
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if hasattr(module, "build_policy"):
        return module.build_policy(str(checkpoint_path) if checkpoint_path else None, device)
    if hasattr(module, "POLICY"):
        policy = module.POLICY(input_dim=53, output_dim=8).to(device)
        if checkpoint_path is not None:
            state_dict = torch.load(str(checkpoint_path), map_location=device)
            policy.load_state_dict(state_dict)
        policy.eval()
        return policy
    raise AttributeError("Policy module must define build_policy or POLICY.")


def build_scene_xml(
    block_size,
    block_pos,
    block_quat,
    target_pos,
    table_pos=None,
    table_size=None,
    target_radius=TARGET_RADIUS,
):
    with open(panda_mj_description.MJCF_PATH, "r") as f:
        base_xml = f.read()

    block_x, block_y, block_z = block_pos
    if table_size is None:
        table_top_z = block_z - block_size
        table_size_x, table_size_y = 0.25, 0.35
    else:
        table_size = np.asarray(table_size, dtype=float)
        table_size_x, table_size_y, table_top_z = table_size[:3]

    if table_pos is None:
        table_x, table_y = 0.58, 0.10
    else:
        table_pos = np.asarray(table_pos, dtype=float)
        table_x, table_y = table_pos[:2]

    target_x, target_y = target_pos[0], target_pos[1]
    target_z = target_pos[2]

    extras = f"""
    <geom name='floor' type='plane' size='2 2 0.05' rgba='0.85 0.85 0.9 1'
            friction='1.0 0.05 0.001'/>

    <geom name='table' type='box' pos='{table_x} {table_y} 0.0' size='{table_size_x} {table_size_y} {table_top_z}'
            rgba='0.75 0.65 0.5 1' friction='1.0 0.05 0.001'/>

    <body name='block' pos='{block_x} {block_y} {block_z}' quat='{block_quat[0]} {block_quat[1]} {block_quat[2]} {block_quat[3]}'>
        <freejoint name='block_free'/>
        <geom name='block_geom' type='box' size='{block_size} {block_size} {block_size}'
            rgba='0.9 0.2 0.2 1' mass='0.05'
            friction='1.0 0.05 0.001'/>
    </body>

    <body name='target_marker' mocap='true' pos='{target_x} {target_y} {target_z}'>
        <geom type='cylinder' size='{target_radius} 0.001' rgba='0.2 0.9 0.2 0.5'
            contype='0' conaffinity='0'/>
    </body>

    <camera name='task_view' pos='1.6 -0.6 1.0' xyaxes='0.5 0.87 0  -0.30 0.17 0.94'/>
    """

    patched_xml = base_xml.replace("</worldbody>", extras + "\n  </worldbody>")
    patched_xml = re.sub(
        r'(<key\s+name="home"\s+qpos=")([^"]+)(")',
        r"\1\2 "
        + f"{block_x} {block_y} {block_z} "
        + f"{block_quat[0]} {block_quat[1]} {block_quat[2]} {block_quat[3]}"
        + r"\3",
        patched_xml,
    )
    return patched_xml


def build_model_data_for_env(env_npz_path: str):
    raw = np.load(env_npz_path)
    env_data = {k: raw[k] for k in raw.files}
    block_size = float(env_data["block_size"])
    block_pos = env_data["block_init_xpos"]
    block_quat = env_data["block_init_quat"]
    target_pos = env_data["target_xpos"]
    table_pos = env_data.get("table_xpos")
    table_size = env_data.get("table_size")
    target_radius = float(env_data.get("target_radius", TARGET_RADIUS))

    patched_xml = build_scene_xml(
        block_size,
        block_pos,
        block_quat,
        target_pos,
        table_pos=table_pos,
        table_size=table_size,
        target_radius=target_radius,
    )
    tmp_path = Path(panda_mj_description.PACKAGE_PATH) / f"_submission_eval_scene_{os.getpid()}.xml"
    with open(tmp_path, "w") as f:
        f.write(patched_xml)

    model = mujoco.MjModel.from_xml_path(str(tmp_path))
    data = mujoco.MjData(model)
    reset_to_keyframe(model, data, "home")

    ids = {
        "block_body_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block"),
        "hand_body_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand"),
        "task_cam_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "task_view"),
    }
    return model, data, ids, env_data


def evaluate_single_env(module, policy, stats, env_npz_path, device, max_steps):
    model, data, ids, env_data = build_model_data_for_env(env_npz_path)
    success_distance = TARGET_RADIUS + float(env_data["block_size"])
    student_out = module.run_single_episode(
        policy=policy,
        model=model,
        data=data,
        ids=ids,
        env_data=env_data,
        stats=stats,
        device=device,
        max_steps=max_steps,
        render=False,
        render_fps=60,
        success_distance=success_distance,
    )

    final_block_pos = data.xpos[ids["block_body_id"]].copy()
    final_dist = float(np.linalg.norm(final_block_pos - env_data["target_xpos"]))
    success = final_dist < success_distance
    return {
        "env_path": str(env_npz_path),
        "env_name": Path(env_npz_path).name,
        "success": bool(success),
        "final_distance": final_dist,
        "success_distance": float(success_distance),
        "final_block_pos": final_block_pos.tolist(),
        "target_pos": env_data["target_xpos"].tolist(),
        "student_output": student_out,
    }


def summarize(results):
    dists = np.asarray([r["final_distance"] for r in results], dtype=np.float64)
    success_count = int(sum(r["success"] for r in results))
    return {
        "count": len(results),
        "success_count": success_count,
        "success_rate": float(success_count / len(results)) if results else 0.0,
        "mean_distance": float(dists.mean()) if len(dists) else None,
        "median_distance": float(np.median(dists)) if len(dists) else None,
        "max_distance": float(dists.max()) if len(dists) else None,
        "max_distance_env": results[int(dists.argmax())]["env_name"] if len(dists) else None,
        "min_distance": float(dists.min()) if len(dists) else None,
        "min_distance_env": results[int(dists.argmin())]["env_name"] if len(dists) else None,
    }


def read_env_list(list_path):
    env_paths = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            env_paths.append(line)
    return env_paths


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {device}")

    module = load_policy_module(args.policy_file)
    if not hasattr(module, "run_single_episode"):
        raise AttributeError("Policy module must define run_single_episode.")
    policy = build_policy_from_module(module, args.checkpoint_file, device)
    stats = np.load(args.stats_file) if args.stats_file else None

    if args.env_list:
        env_paths = read_env_list(args.env_list)
    else:
        env_paths = sorted(str(p) for p in Path(args.env_dir).glob("*.npz"))
    if args.start_idx is not None or args.end_idx is not None:
        start_idx = 0 if args.start_idx is None else args.start_idx
        end_idx = len(env_paths) if args.end_idx is None else args.end_idx
        env_paths = env_paths[start_idx:end_idx]
    if args.max_envs is not None:
        env_paths = env_paths[: args.max_envs]
    if not env_paths:
        raise RuntimeError(f"No .npz environment files found under {args.env_dir}")

    eval_source = args.env_list if args.env_list else args.env_dir
    print(f"Evaluating {len(env_paths)} environments from {eval_source}")
    results = []
    for i, env_path in enumerate(tqdm(env_paths, desc="Evaluating")):
        outcome = evaluate_single_env(module, policy, stats, env_path, device, args.max_steps)
        print(
            f"Env {i + 1}/{len(env_paths)}: {Path(env_path).name} -> "
            f"success={outcome['success']}, final_dist={outcome['final_distance']:.4f} m",
            flush=True,
        )
        results.append(outcome)

    summary = summarize(results)
    payload = {
        "policy_file": args.policy_file,
        "checkpoint_file": args.checkpoint_file,
        "stats_file": args.stats_file,
        "env_dir": args.env_dir,
        "env_list": args.env_list,
        "max_steps": args.max_steps,
        "summary": summary,
        "results": results,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")

    print("----------------------------------------")
    print(f"Success: {summary['success_count']}/{summary['count']}")
    print(f"Success rate: {summary['success_rate'] * 100:.2f}%")
    print(f"Mean distance: {summary['mean_distance']:.4f} m")
    print(f"Median distance: {summary['median_distance']:.4f} m")
    print(f"Max distance: {summary['max_distance']:.4f} m ({summary['max_distance_env']})")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_file", required=True)
    parser.add_argument("--checkpoint_file", default=None)
    parser.add_argument("--stats_file", default=None)
    parser.add_argument("--env_dir", default="./open_env")
    parser.add_argument("--env_list", default=None)
    parser.add_argument("--max_envs", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
