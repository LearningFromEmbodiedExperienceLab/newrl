"""Steel augmented-drive actuator-force reconstruction for scalar robot joints."""

import warp as wp


@wp.kernel
def reconstruct_augmented_drive_torque(
    q_index: wp.array[wp.int32],
    dof_index: wp.array[wp.int32],
    target_index: wp.array[wp.int32],
    q: wp.array[wp.float32],
    qd: wp.array[wp.float32],
    qd_next: wp.array[wp.float32],
    target_q: wp.array[wp.float32],
    target_qd: wp.array[wp.float32],
    stiffness: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    effort_limit: wp.array[wp.float32],
    dt: float,
    torque: wp.array[wp.float32],
):
    row = wp.tid()
    dof = dof_index[row]
    ke = stiffness[dof]
    kd = damping[dof]
    result = float(0.0)  # noqa: UP018 - mutable Warp variable
    if (ke > 0.0 or kd > 0.0) and (ke * dt * dt + kd * dt > 0.0):
        velocity = qd[dof]
        explicit = -(
            ke * (q[q_index[row]] - target_q[target_index[row]] + dt * velocity)
            + kd * (velocity - target_qd[dof])
        )
        limit = effort_limit[dof]
        if limit > 0.0:
            explicit = wp.clamp(explicit, -limit, limit)
        result = explicit - (dt * ke + kd) * (qd_next[dof] - velocity)
    torque[dof] = result
