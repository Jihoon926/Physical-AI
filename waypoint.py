import numpy as np
from dataclasses import dataclass

@dataclass
class Waypoint:
    name: str
    pos: np.ndarray            # target EE position (world)
    grip: float                # gripper command (0 = closed, 255 = open)
    settle_time: float = 0.5   # extra seconds to hold this pose before moving on
    pos_tol: float = 5e-3      # convergence threshold for advancing


def build_waypoints(block_pos, target_pos):
    """Pick-and-place waypoint list for one block + one target."""
    APPROACH_HEIGHT = 0.20      # height above the block/target during approach
    GRASP_OFFSET    = 0.105     # hand-body height above the block top when grasping

    above_block  = block_pos  + np.array([0, 0, APPROACH_HEIGHT])
    on_block     = block_pos  + np.array([0, 0, GRASP_OFFSET])
    above_target = target_pos + np.array([0, 0, APPROACH_HEIGHT])
    on_target    = target_pos + np.array([0, 0, GRASP_OFFSET])

    # 1cm tolerance avoids stalling on boundary values (e.g., 0.0080m plateau).
    return [
        Waypoint('approach', above_block,  grip=255, settle_time=2.0, pos_tol=1e-2),
        Waypoint('descend',  on_block,     grip=255, settle_time=2.0, pos_tol=1e-2),
        Waypoint('grasp',    on_block,     grip=0,   settle_time=0.5, pos_tol=np.inf),
        Waypoint('lift',     above_block,  grip=0,   settle_time=1.0, pos_tol=1e-2),
        Waypoint('move',     above_target, grip=0,   settle_time=2.0, pos_tol=1e-2),
        Waypoint('place',    on_target,    grip=0,   settle_time=1.0, pos_tol=1e-2),
        Waypoint('release',  on_target,    grip=255, settle_time=0.5, pos_tol=np.inf),
        Waypoint('retreat',  above_target, grip=255, settle_time=1.0, pos_tol=1e-2),
    ]
