"""Render one policy rollout to a video file."""

import argparse
import json
import subprocess
from pathlib import Path

import mediapy as media
import mujoco
import numpy as np
import torch

from evaluate_policy import TARGET_RADIUS, build_model_data_for_env, build_policy_from_module, load_policy_module
from waypoint import build_waypoints


def write_video(path, frames, fps):
    try:
        media.write_video(path, frames, fps=fps)
        return
    except RuntimeError as media_error:
        try:
            import imageio_ffmpeg
        except ImportError as import_error:
            raise media_error from import_error

        first_frame = np.asarray(frames[0], dtype=np.uint8)
        height, width = first_frame.shape[:2]
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(path),
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            assert process.stdin is not None
            for frame in frames:
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            return_code = process.wait()
        except Exception:
            process.kill()
            process.wait()
            raise

        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed while writing {path}: {stderr}")


def maybe_get(module, name, default):
    return getattr(module, name, default)


def render_episode(module, policy, stats, env_path, device, max_steps, fps, width, height):
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

    frames = []
    transitions = []
    steps_taken = 0
    advance_threshold = maybe_get(module, "ADVANCE_THRESHOLD", 0.5)
    min_advance_steps = maybe_get(module, "MIN_ADVANCE_STEPS", 30)
    max_joint_delta = maybe_get(module, "MAX_JOINT_DELTA", 0.06)
    grip_min = maybe_get(module, "GRIP_MIN", 0.0)
    grip_max = maybe_get(module, "GRIP_MAX", 255.0)

    with mujoco.Renderer(model, width=width, height=height) as renderer:
        for step in range(max_steps):
            steps_taken = step + 1

            curr_qpos = data.qpos.copy()
            curr_qvel = data.qvel.copy()
            curr_bpos = data.xpos[ids["block_body_id"]].copy()
            curr_bquat = data.xquat[ids["block_body_id"]].copy()
            curr_hand_pos = data.xpos[ids["hand_body_id"]].copy()
            curr_hand_quat = data.xquat[ids["hand_body_id"]].copy()

            transition_input = module._build_model_input(
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
            transition_input = module._normalize_input(transition_input, stats, device)

            advanced_by_policy = False
            with torch.no_grad():
                if hasattr(policy, "advance_logits"):
                    logits, hidden = policy.advance_logits(transition_input, hidden)
                    advance_prob = float(torch.sigmoid(logits[:, -1]).item())
                    if (
                        advance_prob > advance_threshold
                        and stage_steps > min_advance_steps
                        and waypoint_index < len(waypoints) - 1
                    ):
                        old_index = waypoint_index
                        waypoint_index += 1
                        curr_waypoint = waypoints[waypoint_index]
                        stay_counter = 0
                        grasp_counter = 0
                        stage_steps = 0
                        advanced_by_policy = True
                        transitions.append(
                            {
                                "step": step,
                                "from": int(old_index),
                                "to": int(waypoint_index),
                                "name": curr_waypoint.name,
                                "source": "advance_rnn",
                                "advance_prob": advance_prob,
                            }
                        )
                elif hasattr(policy, "stage_logits"):
                    logits, hidden = policy.stage_logits(transition_input, hidden)
                    predicted_stage = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                    if predicted_stage > waypoint_index:
                        old_index = waypoint_index
                        waypoint_index = min(waypoint_index + 1, predicted_stage, len(waypoints) - 1)
                        curr_waypoint = waypoints[waypoint_index]
                        stay_counter = 0
                        grasp_counter = 0
                        stage_steps = 0
                        advanced_by_policy = True
                        transitions.append(
                            {
                                "step": step,
                                "from": int(old_index),
                                "to": int(waypoint_index),
                                "name": curr_waypoint.name,
                                "source": "stage_rnn",
                                "predicted_stage": predicted_stage,
                            }
                        )

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
            joint_delta = np.clip(pred[:7], -max_joint_delta, max_joint_delta)
            grip_cmd = float(np.clip(pred[7], grip_min, grip_max))

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
                    transitions.append(
                        {
                            "step": step,
                            "from": int(old_index),
                            "to": int(waypoint_index),
                            "name": curr_waypoint.name,
                            "source": "geometry",
                            "distance": float(dist_curr_waypoint),
                        }
                    )

            data.ctrl[:7] = curr_qpos[:7] + joint_delta
            data.ctrl[7] = grip_cmd
            mujoco.mj_step(model, data)
            stage_steps += 1

            if len(frames) < data.time * fps:
                renderer.update_scene(data, camera=ids["task_cam_id"])
                frames.append(renderer.render())

            if waypoint_index == len(waypoints) - 1:
                curr_bpos_after = data.xpos[ids["block_body_id"]].copy()
                if np.linalg.norm(curr_bpos_after - target_pos) < success_distance:
                    break

    final_block_pos = data.xpos[ids["block_body_id"]].copy()
    final_distance = float(np.linalg.norm(final_block_pos - target_pos))
    return frames, {
        "env_path": str(env_path),
        "env_name": Path(env_path).name,
        "num_steps": int(steps_taken),
        "success": bool(final_distance < success_distance),
        "final_distance": final_distance,
        "success_distance": float(success_distance),
        "final_block_pos": final_block_pos.tolist(),
        "target_pos": target_pos.tolist(),
        "final_waypoint_index": int(waypoint_index),
        "final_waypoint_name": curr_waypoint.name,
        "transitions": transitions,
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    module = load_policy_module(args.policy_file)
    policy = build_policy_from_module(module, args.checkpoint_file, device)
    stats = np.load(args.stats_file) if args.stats_file else None

    frames, result = render_episode(
        module=module,
        policy=policy,
        stats=stats,
        env_path=args.env_path,
        device=device,
        max_steps=args.max_steps,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )

    output_video = Path(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    write_video(output_video, frames, fps=args.fps)
    print(f"Wrote {output_video}")

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {output_json}")

    print(
        f"{result['env_name']} success={result['success']} "
        f"final_dist={result['final_distance']:.4f} "
        f"final_wp={result['final_waypoint_name']} steps={result['num_steps']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_file", required=True)
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--stats_file", required=True)
    parser.add_argument("--env_path", required=True)
    parser.add_argument("--output_video", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
