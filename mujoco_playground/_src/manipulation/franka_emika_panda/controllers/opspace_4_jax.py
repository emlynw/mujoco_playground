from typing import Optional, Tuple, Union
import mujoco
import jax
import jax.numpy as jnp
from dm_robotics.transformations import transformations as tr


def pseudo_inverse(M, damped=False, lambda_=0.2):
    if not damped:
        return jnp.linalg.pinv(M)
    U, sing_vals, V = jnp.linalg.svd(M, full_matrices=False)
    S_inv = sing_vals / (sing_vals**2 + lambda_**2)
    return (V.T * S_inv) @ U.T


def pd_control(
    x: jnp.ndarray,
    x_des: jnp.ndarray,
    dx: jnp.ndarray,
    kp_kv: jnp.ndarray,
    max_pos_error: float = 0.05,
) -> jnp.ndarray:
    x_err = jnp.clip(x - x_des, -max_pos_error, max_pos_error)
    dx_err = dx
    return -kp_kv[:, 0] * x_err - kp_kv[:, 1] * dx_err


def pd_control_orientation(
    quat: jnp.ndarray,
    quat_des: jnp.ndarray,
    w: jnp.ndarray,
    kp_kv: jnp.ndarray,
    max_ori_error: float = 0.05,
) -> jnp.ndarray:
    quat_err = tr.quat_diff_active(source_quat=quat_des, target_quat=quat)
    ori_err = jnp.clip(tr.quat_to_axisangle(quat_err), -max_ori_error, max_ori_error)
    w_err = w
    return -kp_kv[:, 0] * ori_err - kp_kv[:, 1] * w_err


def saturate_torque_rate(tau_d_calculated, tau_J_d, delta_tau_max):
    return tau_J_d + jnp.clip(tau_d_calculated - tau_J_d, -delta_tau_max, delta_tau_max)


def opspace_4(
    model,
    data,
    site_id,
    dof_ids: jnp.ndarray,
    pos: Optional[jnp.ndarray] = None,
    ori: Optional[jnp.ndarray] = None,
    joint: Optional[jnp.ndarray] = None,
    pos_gains: Union[Tuple[float, float, float], jnp.ndarray] = (1500.0, 1500.0, 1500.0),
    ori_gains: Union[Tuple[float, float, float], jnp.ndarray] = (200.0, 200.0, 200.0),
    joint_upper_limits: Union[Tuple[float, float, float, float, float, float, float], jnp.ndarray] = (
        2.8,
        1.7,
        2.8,
        -0.08,
        2.8,
        3.74,
        2.8,
    ),
    joint_lower_limits: Union[Tuple[float, float, float, float, float, float, float], jnp.ndarray] = (
        -2.8,
        -1.7,
        -2.8,
        -3.0,
        -2.8,
        -0.010,
        -2.8,
    ),
    translational_damping: float = 89.0,
    rotational_damping: float = 7.0,
    nullspace_stiffness: float = 0.2,
    joint1_nullspace_stiffness: float = 100.0,
    max_pos_error: float = 0.01,
    max_ori_error: float = 0.01,
    delta_tau_max: float = 0.5,
    gravity_comp: bool = True,
    damped: bool = False,
    lambda_: float = 0.2,
) -> jnp.ndarray:
    x_des = jnp.asarray(data.site_xpos[site_id] if pos is None else pos)
    quat_des = (
        tr.mat_to_quat(jnp.reshape(data.site_xmat[site_id], (3, 3)))
        if ori is None
        else tr.mat_to_quat(jnp.asarray(ori))
        if ori.shape == (3, 3)
        else ori
    )
    q_d_nullspace = jnp.asarray(data.qpos[dof_ids] if joint is None else joint)

    kp_pos = jnp.asarray(pos_gains)
    kd_pos = translational_damping * jnp.ones_like(kp_pos)
    kp_kv_pos = jnp.stack([kp_pos, kd_pos], axis=-1)

    kp_ori = jnp.asarray(ori_gains)
    kd_ori = rotational_damping * jnp.ones_like(kp_ori)
    kp_kv_ori = jnp.stack([kp_ori, kd_ori], axis=-1)

    q = data.qpos[dof_ids]
    dq = data.qvel[dof_ids]

    J_v = jnp.zeros((3, model.nv))
    J_w = jnp.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, J_v, J_w, site_id)
    J_v, J_w = J_v[:, dof_ids], J_w[:, dof_ids]
    J = jnp.vstack((J_v, J_w))

    x = data.site_xpos[site_id]
    dx = J_v @ dq
    ddx = pd_control(x, x_des, dx, kp_kv_pos, max_pos_error)

    quat = tr.mat_to_quat(jnp.reshape(data.site_xmat[site_id], (3, 3)))
    if quat @ quat_des < 0.0:
        quat *= -1.0
    w = J_w @ dq
    dw = pd_control_orientation(quat, quat_des, w, kp_kv_ori, max_ori_error)

    C = data.qfrc_bias[dof_ids]
    F = jnp.concatenate([ddx, dw])
    tau_task = J.T @ F

    q_error = q_d_nullspace - q
    q_error = q_error.at[0].set(q_error[0] * joint1_nullspace_stiffness)
    dq_error = dq
    dq_error = dq_error.at[0].set(dq_error[0] * 2 * jnp.sqrt(joint1_nullspace_stiffness))
    tau_nullspace = nullspace_stiffness * q_error - 2 * jnp.sqrt(nullspace_stiffness) * dq_error

    if damped:
        jacobian_transpose_pinv = pseudo_inverse(J.T, damped=True, lambda_=lambda_)
        tau_nullspace = (jnp.eye(len(dof_ids)) - J.T @ jacobian_transpose_pinv) @ tau_nullspace

    tau = tau_task + tau_nullspace

    # Saturate torques near joint limits
    tau = jnp.where(
        (q >= joint_upper_limits) & (tau > 0.0),
        0.0,
        jnp.where((q <= joint_lower_limits) & (tau < 0.0), 0.0, tau),
    )

    if gravity_comp:
        tau += C

    return saturate_torque_rate(tau, data.qfrc_actuator[dof_ids], delta_tau_max)
