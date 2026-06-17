import mujoco
import numpy as np

def ik_step(model, data, target_pos, target_quat,
            body_id, dof_ids, damping=0.1, rot_w=0.5):
    """
    Task-priority IK step (Nakamura & Hanafusa, 1987).

    Primary task   : POSITION  — solved with damped pseudo-inverse (stable).
    Secondary task : ORIENTATION — applied only in the TRUE null space of Jp
                     (SVD-based projector), so it cannot disturb position.

    Key implementation detail
    -------------------------
    The null-space projector  N = I - Jp† Jp  must be computed with the TRUE
    (undamped) pseudo-inverse, not the damped one.  If the damped Jp†_dam is
    used to form N, the "null space" leaks into the row space and the
    orientation correction contaminates position convergence — which is exactly
    the bug we observed.  We therefore use SVD to get both the exact projector
    and the damped step direction separately.
    """
    n = len(dof_ids)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body_id], body_id)
    Jp = jacp[:, dof_ids]   # (3, n)  position Jacobian
    Jr = jacr[:, dof_ids]   # (3, n)  rotation Jacobian

    # --- SVD of Jp ---
    U, S, Vt = np.linalg.svd(Jp, full_matrices=True)   # Vt: (n, n)
    r = np.sum(S > 1e-6)   # numerical rank

    # True (undamped) right pseudo-inverse — only used for null-space projector.
    # Jp†_true = Vt[:r].T @ diag(1/S[:r]) @ U[:,:r].T
    Jp_pinv_true = Vt[:r].T @ np.diag(1.0 / S[:r]) @ U[:, :r].T  # (n, 3)

    # Exact null-space projector  N = I - Jp†_true Jp = I - Vt[:r].T Vt[:r]
    N = np.eye(n) - Vt[:r].T @ Vt[:r]                             # (n, n)

    # Damped step for the primary task (avoids singularity blow-up).
    Jp_pinv_dam = Jp.T @ np.linalg.solve(Jp @ Jp.T + damping**2 * np.eye(3),
                                          np.eye(3))               # (n, 3)
    err_pos = target_pos - data.xpos[body_id]
    dq      = Jp_pinv_dam @ err_pos

    # --- Orientation correction in null space ---
    err_quat = np.zeros(3)
    mujoco.mju_subQuat(err_quat, target_quat, data.xquat[body_id])
    e_rot = np.linalg.norm(err_quat)

    if rot_w > 0.0 and e_rot > 1e-6:
        Jr_pinv = Jr.T @ np.linalg.solve(Jr @ Jr.T + damping**2 * np.eye(3),
                                          np.eye(3))               # (n, 3)
        dq += N @ (Jr_pinv @ (rot_w * err_quat))

    return dq, np.linalg.norm(err_pos), e_rot


def solve_ik(model, data, target_pos, target_quat, body_id, dof_ids,
             q_init=None, max_iters=200, pos_tol=1e-3, rot_tol=1e9,
             step_scale=0.5, damping=0.1, rot_w=0.5):
    """
    Iterative task-priority IK.

    Position is the only hard convergence criterion (pos_tol).
    Orientation is a soft secondary task in the null space of position.
    rot_w : weight for the orientation secondary task (default 0.5).
    """
    q_saved = data.qpos.copy()
    q = q_init.copy() if q_init is not None else q_saved.copy()

    lo = model.jnt_range[dof_ids, 0]
    hi = model.jnt_range[dof_ids, 1]

    e_pos = e_rot = np.inf
    for i in range(max_iters):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        dq, e_pos, e_rot = ik_step(model, data, target_pos, target_quat,
                                   body_id, dof_ids, damping=damping, rot_w=rot_w)
        if e_pos < pos_tol:
            break

        q[dof_ids] = np.clip(q[dof_ids] + step_scale * dq, lo, hi)

    # Restore simulation state.
    data.qpos[:] = q_saved
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    return q[dof_ids], i, e_pos, e_rot
