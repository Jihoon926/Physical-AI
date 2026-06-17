"""Generate randomized expert pick-and-place trajectories.

The saved npz files keep the same core keys as 00_data_generator.ipynb output,
with additional table metadata used by evaluate_policy.py when present.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from robot_descriptions import panda_mj_description
from tqdm.auto import tqdm

from controller import PickAndPlaceController
from evaluate_policy import TARGET_RADIUS, build_scene_xml
from helper import reset_to_keyframe
from waypoint import build_waypoints


@dataclass
class SceneParams:
    seed: int
    table_x: float
    table_y: float
    table_size_x: float
    table_size_y: float
    table_top_z: float
    block_size: float
    block_x: float
    block_y: float
    block_z: float
    block_yaw: float
    target_x: float
    target_y: float
    target_marker_z: float

    @property
    def block_pos(self):
        return np.array([self.block_x, self.block_y, self.block_z], dtype=np.float64)

    @property
    def target_marker_pos(self):
        return np.array([self.target_x, self.target_y, self.target_marker_z], dtype=np.float64)

    @property
    def waypoint_target_pos(self):
        return np.array([self.target_x, self.target_y, self.block_z], dtype=np.float64)

    @property
    def table_pos(self):
        return np.array([self.table_x, self.table_y, 0.0], dtype=np.float64)

    @property
    def table_size(self):
        return np.array([self.table_size_x, self.table_size_y, self.table_top_z], dtype=np.float64)


def sample_uniform(rng, lo, hi):
    return float(rng.uniform(float(lo), float(hi)))


def sample_scene(args, seed):
    rng = np.random.default_rng(seed)
    table_x = sample_uniform(rng, args.table_x_min, args.table_x_max)
    table_y = sample_uniform(rng, args.table_y_min, args.table_y_max)
    table_size_x = sample_uniform(rng, args.table_size_x_min, args.table_size_x_max)
    table_size_y = sample_uniform(rng, args.table_size_y_min, args.table_size_y_max)
    table_top_z = sample_uniform(rng, args.table_top_z_min, args.table_top_z_max)
    block_size = sample_uniform(rng, args.block_size_min, args.block_size_max)
    block_z = table_top_z + block_size
    target_marker_z = table_top_z + 0.001

    block_x_min = table_x - table_size_x / 2.0 + block_size + args.edge_margin
    block_x_max = table_x + table_size_x / 2.0 - block_size - args.edge_margin
    block_y_min = table_y - table_size_y / 2.0 + block_size + args.edge_margin
    block_y_max = table_y + table_size_y / 2.0 - block_size - args.edge_margin

    target_margin = TARGET_RADIUS + args.edge_margin
    target_x_min = table_x - table_size_x / 2.0 + target_margin
    target_x_max = table_x + table_size_x / 2.0 - target_margin
    target_y_min = table_y - table_size_y / 2.0 + target_margin
    target_y_max = table_y + table_size_y / 2.0 - target_margin

    if block_x_min >= block_x_max or block_y_min >= block_y_max:
        raise ValueError("Sampled table is too small for the block range.")
    if target_x_min >= target_x_max or target_y_min >= target_y_max:
        raise ValueError("Sampled table is too small for the target range.")

    for _ in range(args.max_position_resamples):
        block_x = sample_uniform(rng, block_x_min, block_x_max)
        block_y = sample_uniform(rng, block_y_min, block_y_max)
        target_x = sample_uniform(rng, target_x_min, target_x_max)
        target_y = sample_uniform(rng, target_y_min, target_y_max)
        xy_dist = np.linalg.norm([target_x - block_x, target_y - block_y])
        if args.min_target_block_xy <= xy_dist <= args.max_target_block_xy:
            break
    else:
        block_x = sample_uniform(rng, block_x_min, block_x_max)
        block_y = sample_uniform(rng, block_y_min, block_y_max)
        target_x = sample_uniform(rng, target_x_min, target_x_max)
        target_y = sample_uniform(rng, target_y_min, target_y_max)

    block_yaw = sample_uniform(rng, args.block_yaw_min, args.block_yaw_max)
    return SceneParams(
        seed=seed,
        table_x=table_x,
        table_y=table_y,
        table_size_x=table_size_x,
        table_size_y=table_size_y,
        table_top_z=table_top_z,
        block_size=block_size,
        block_x=block_x,
        block_y=block_y,
        block_z=block_z,
        block_yaw=block_yaw,
        target_x=target_x,
        target_y=target_y,
        target_marker_z=target_marker_z,
    )


def yaw_to_quat(yaw):
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, np.array([0.0, 0.0, yaw], dtype=np.float64), "xyz")
    return quat


def build_model_and_data(scene, tmp_xml_name):
    block_quat = yaw_to_quat(scene.block_yaw)
    patched_xml = build_scene_xml(
        scene.block_size,
        scene.block_pos,
        block_quat,
        scene.target_marker_pos,
        table_pos=scene.table_pos,
        table_size=scene.table_size,
        target_radius=TARGET_RADIUS,
    )
    tmp_path = Path(panda_mj_description.PACKAGE_PATH) / tmp_xml_name
    tmp_path.write_text(patched_xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(tmp_path))
    data = mujoco.MjData(model)
    reset_to_keyframe(model, data, "home")
    ids = {
        "block_body_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block"),
        "hand_body_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand"),
    }
    return model, data, ids, block_quat


def run_expert_trajectory(model, data, controller, duration):
    block_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
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

    while data.time < duration:
        controller(model, data)
        mujoco.mj_step(model, data)
        history["t"].append(float(data.time))
        history["qpos"].append(data.qpos.copy())
        history["qvel"].append(data.qvel.copy())
        history["ctrl"].append(data.ctrl.copy())
        history["bpos"].append(data.xpos[block_body_id].copy())
        history["bquat"].append(data.xquat[block_body_id].copy())
        history["hpos"].append(data.xpos[hand_body_id].copy())
        history["hquat"].append(data.xquat[hand_body_id].copy())
        history["waypoint_pos"].append(controller._current_waypoint().pos.copy())
        history["waypoint_grip"].append(float(controller._current_waypoint().grip))

    return {k: np.asarray(v) for k, v in history.items()}


def generate_one(scene, args):
    tmp_xml_name = f"_random_pick_place_scene_{os.getpid()}_{scene.seed}.xml"
    model, data, ids, block_quat = build_model_and_data(scene, tmp_xml_name)
    fixed_quat = data.xquat[ids["hand_body_id"]].copy()
    block_pos_before = data.xpos[ids["block_body_id"]].copy()
    waypoints = build_waypoints(block_pos_before, scene.waypoint_target_pos)
    controller = PickAndPlaceController(
        model,
        data,
        waypoints,
        hand_body_id=ids["hand_body_id"],
        fixed_quat=fixed_quat,
        verbose=args.verbose_controller,
    )
    history = run_expert_trajectory(model, data, controller, duration=args.duration)
    final_block_pos = data.xpos[ids["block_body_id"]].copy()
    final_distance = float(np.linalg.norm(final_block_pos - scene.waypoint_target_pos))
    success = final_distance < args.success_tolerance

    history["block_init_xpos"] = scene.block_pos
    history["block_init_quat"] = block_quat
    history["block_size"] = np.asarray(scene.block_size)
    history["target_xpos"] = scene.target_marker_pos
    history["table_xpos"] = scene.table_pos
    history["table_size"] = scene.table_size
    history["target_radius"] = np.asarray(TARGET_RADIUS)
    history["generator_seed"] = np.asarray(scene.seed, dtype=np.int64)
    history["success"] = np.asarray(success)
    history["final_distance_to_waypoint_target"] = np.asarray(final_distance)

    summary = asdict(scene)
    summary.update(
        {
            "success": bool(success),
            "final_distance_to_waypoint_target": final_distance,
            "final_block_pos": final_block_pos.tolist(),
            "num_steps": int(len(history["t"])),
        }
    )
    return history, summary


def main(args):
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.glob("*.npz")) and not args.overwrite:
        raise RuntimeError(f"{output_dir} already contains npz files. Use --overwrite to replace.")
    output_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    attempts = 0
    summaries = []
    max_attempts = args.max_attempts or args.num_trajectories * args.max_attempts_per_success
    progress = tqdm(total=args.num_trajectories, desc="Generating random expert trajectories")

    while successes < args.num_trajectories and attempts < max_attempts:
        seed = args.seed + attempts
        attempts += 1
        scene = sample_scene(args, seed)
        try:
            history, summary = generate_one(scene, args)
        except Exception as exc:
            summaries.append({"seed": seed, "success": False, "error": repr(exc)})
            continue

        summaries.append(summary)
        if not summary["success"]:
            if args.save_failures:
                fail_path = output_dir / f"failed_random_pick_place_seed_{seed}.npz"
                np.savez(fail_path, **history)
            continue

        save_path = output_dir / f"random_pick_place_seed_{seed}.npz"
        np.savez(save_path, **history)
        successes += 1
        progress.update(1)
        if successes % args.summary_every == 0:
            progress.set_postfix({"attempts": attempts, "last_seed": seed})

    progress.close()
    payload = {
        "output_dir": str(output_dir),
        "num_requested": args.num_trajectories,
        "num_success": successes,
        "num_attempts": attempts,
        "args": vars(args),
        "summaries": summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Generated {successes}/{args.num_trajectories} successful trajectories in {attempts} attempts.")
    if successes < args.num_trajectories:
        raise RuntimeError("Stopped before reaching requested trajectory count.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate randomized full expert pick-and-place trajectories.")
    parser.add_argument("--output_dir", default="random_trajectories")
    parser.add_argument("--num_trajectories", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--success_tolerance", type=float, default=0.05)
    parser.add_argument("--max_attempts", type=int, default=None)
    parser.add_argument("--max_attempts_per_success", type=int, default=5)
    parser.add_argument("--summary_every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_failures", action="store_true")
    parser.add_argument("--verbose_controller", action="store_true")

    parser.add_argument("--table_x_min", type=float, default=0.58)
    parser.add_argument("--table_x_max", type=float, default=0.58)
    parser.add_argument("--table_y_min", type=float, default=0.10)
    parser.add_argument("--table_y_max", type=float, default=0.10)
    parser.add_argument("--table_size_x_min", type=float, default=0.25)
    parser.add_argument("--table_size_x_max", type=float, default=0.25)
    parser.add_argument("--table_size_y_min", type=float, default=0.35)
    parser.add_argument("--table_size_y_max", type=float, default=0.35)
    parser.add_argument("--table_top_z_min", type=float, default=0.32)
    parser.add_argument("--table_top_z_max", type=float, default=0.42)
    parser.add_argument("--block_size_min", type=float, default=0.015)
    parser.add_argument("--block_size_max", type=float, default=0.035)
    parser.add_argument("--block_yaw_min", type=float, default=-np.pi)
    parser.add_argument("--block_yaw_max", type=float, default=np.pi)
    parser.add_argument("--edge_margin", type=float, default=0.01)
    parser.add_argument("--min_target_block_xy", type=float, default=0.10)
    parser.add_argument("--max_target_block_xy", type=float, default=0.42)
    parser.add_argument("--max_position_resamples", type=int, default=100)
    main(parser.parse_args())
