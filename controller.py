import numpy as np
from ik import solve_ik

class PickAndPlaceController:
    MAX_DCTRL = 0.001

    def __init__(self, model, data, waypoints, hand_body_id, fixed_quat,
                 verbose=True):
        self.model = model
        self.data  = data
        self.waypoints = waypoints
        self.hand_body_id = hand_body_id
        self.fixed_quat = fixed_quat.copy()
        self.verbose = verbose

        self.state_idx = 0
        self.state_entered_at = data.time
        self.q_warmstart = data.qpos.copy()

        self.max_iters = 100

        data.ctrl[:7] = data.qpos[:7]

    def _current_waypoint(self):
        return self.waypoints[self.state_idx]

    def _advance_if_ready(self):
        wp = self._current_waypoint()
        ee_pos = self.data.xpos[self.hand_body_id]
        err = np.linalg.norm(ee_pos - wp.pos)
        held = self.data.time - self.state_entered_at

        if err < wp.pos_tol and held > wp.settle_time:
            if self.state_idx < len(self.waypoints) - 1:
                self.state_idx += 1
                self.state_entered_at = self.data.time
                if self.verbose:
                    print(f'  [{self.data.time:5.2f}s]  -> {self._current_waypoint().name}')

    def __call__(self, model, data):
        self._advance_if_ready()
        wp = self._current_waypoint()

        i = self.max_iters
        cnt = 0

        while i >= self.max_iters:
            q_target, i, e_pos, e_rot = solve_ik(
                model, data,
                target_pos=wp.pos,
                target_quat=self.fixed_quat,
                body_id=self.hand_body_id,
                dof_ids=np.arange(7),
                q_init=self.q_warmstart + np.random.uniform(-0.001, 0.001, size=16),  # add noise to warmstart for robustness
                max_iters=self.max_iters, step_scale=0.5, damping=0.01,
                rot_w=0.001,   # SVD min-norm solution preserves downward orientation without explicit constraint
            )
            cnt += 1

            if i < self.max_iters:
                break

            if cnt > 10:
                print(f'  Warning: IK solver struggling at waypoint "{wp.name}" (err {e_pos:.4f} m, rot err {e_rot:.4f} rad). Applying best effort solution.')
                break

        self.q_warmstart = data.qpos.copy()
        self.q_warmstart[:7] = q_target

        dq = q_target - data.ctrl[:7]
        dq = np.clip(dq, -self.MAX_DCTRL, self.MAX_DCTRL)
        data.ctrl[:7] += dq
        data.ctrl[7]   = wp.grip
