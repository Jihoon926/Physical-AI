import mujoco
import mediapy as media
import numpy as np

def init_render(model, data, showjoint=False, camera_id=-1):
    """Forward-evaluate the model and render a single still image."""
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = showjoint

    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, width=640, height=480) as renderer:
        renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
        media.show_image(renderer.render())


def run_controller(model, data, controller, duration=5.0, framerate=60,
                   reset=False, render=True, showjoint=False, camera_id=-1, log=False):
    """
    Simulate while calling `controller(model, data)` at every physics step.
    The controller writes desired commands into `data.ctrl`.

    Same shape as Tutorial 6's run_controller; the only addition is
    `camera_id` because Menagerie scenes have named cameras worth using.
    """
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = showjoint

    block_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'block')
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    if render:
        frames = []
    history = {'t': [], 'qpos': [], 'qvel': [], 'ctrl': [], 'bpos': [], 'bquat': [], 'hpos': [], 'hquat': [],
               'waypoint_pos': [], 'waypoint_grip': []}

    if reset:
        mujoco.mj_resetData(model, data)
        prev_qpos = data.qpos.copy()
    with mujoco.Renderer(model, width=640, height=480) as renderer:
        while data.time < duration:
            controller(model, data)
            mujoco.mj_step(model, data)
            if log:
                history['t'].append(data.time)
                history['qpos'].append(data.qpos.copy())
                history['qvel'].append(data.qvel.copy())
                history['ctrl'].append(data.ctrl.copy())
                history['bpos'].append(data.xpos[block_body_id].copy())
                history['bquat'].append(data.xquat[block_body_id].copy())
                history['hpos'].append(data.xpos[hand_body_id].copy())
                history['hquat'].append(data.xquat[hand_body_id].copy())
                history['waypoint_pos'].append(controller._current_waypoint().pos.copy())
                history['waypoint_grip'].append(controller._current_waypoint().grip)

            if render and len(frames) < data.time * framerate:
                renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
                frames.append(renderer.render())

    if render:
        media.show_video(frames, fps=framerate)
    if log:
        for k in history: history[k] = np.array(history[k])
        return history


def reset_to_keyframe(model, data, key_name='home'):
    """Snap the model into one of its predefined keyframes."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id < 0:
        raise ValueError(f'No keyframe named {key_name!r}')
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
