"""Generate randomized pick-and-place environment files without expert filtering."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from evaluate_policy import TARGET_RADIUS
from random_trajectory_generator import sample_scene, yaw_to_quat


def save_env(scene, output_dir):
    block_quat = yaw_to_quat(scene.block_yaw)
    path = output_dir / f"random_env_seed_{scene.seed}.npz"
    np.savez(
        path,
        block_init_xpos=scene.block_pos,
        block_init_quat=block_quat,
        block_size=np.asarray(scene.block_size),
        target_xpos=scene.target_marker_pos,
        table_xpos=scene.table_pos,
        table_size=scene.table_size,
        target_radius=np.asarray(TARGET_RADIUS),
        generator_seed=np.asarray(scene.seed, dtype=np.int64),
    )
    return path


def main(args):
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.glob("*.npz")) and not args.overwrite:
        raise RuntimeError(f"{output_dir} already contains npz files. Use --overwrite to replace.")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    saved = []
    for i in range(args.num_envs):
        seed = args.seed + i
        scene = sample_scene(args, seed)
        path = save_env(scene, output_dir)
        summary = asdict(scene)
        summary["env_path"] = str(path)
        summaries.append(summary)
        saved.append(str(path))

    payload = {
        "output_dir": str(output_dir),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "args": vars(args),
        "env_files": saved,
        "summaries": summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Generated {len(saved)} environment files without expert success filtering.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate randomized env-only pick-and-place .npz files.")
    parser.add_argument("--output_dir", default="random_envs")
    parser.add_argument("--num_envs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=500000)
    parser.add_argument("--overwrite", action="store_true")

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
