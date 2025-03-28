from typing import Optional, Tuple, Union
import jax
import jax.numpy as jnp

# -----------------------------------------------------------------------------
# Simple quaternion utilities in JAX
# -----------------------------------------------------------------------------

def quat_mul(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
  """Multiply two quaternions, returning q1 * q2."""
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  return jnp.array([
      w1*w2 - x1*x2 - y1*y2 - z1*z2,
      w1*x2 + x1*w2 + y1*z2 - z1*y2,
      w1*y2 + y1*w2 + z1*x2 - x1*z2,
      w1*z2 + z1*w2 + x1*y2 - y1*x2,
  ])

def quat_conjugate(q: jnp.ndarray) -> jnp.ndarray:
  """Return the conjugate of a quaternion [w, x, y, z]."""
  return jnp.array([q[0], -q[1], -q[2], -q[3]])

def quat_diff_active(source_quat: jnp.ndarray, target_quat: jnp.ndarray) -> jnp.ndarray:
  """
  Equivalent of dm_robotics.transformations.quat_diff_active(source, target).

  Often defined as: source_quat^{-1} * target_quat
  so that rotating by `quat_diff_active(source, target)` takes `source` onto `target`.
  """
  return quat_mul(quat_conjugate(source_quat), target_quat)

def quat_to_axisangle(q: jnp.ndarray) -> jnp.ndarray:
  """Convert quaternion to an axis-angle (axis * angle)."""
  q = q / jnp.linalg.norm(q)
  w, x, y, z = q
  eps = 1e-8
  angle = 2.0 * jnp.arccos(jnp.clip(w, -1.0, 1.0))
  sin_half = jnp.sqrt(1.0 - w*w)
  sin_half = jnp.maximum(sin_half, eps)
  axis = jnp.array([x, y, z]) / sin_half
  return axis * angle

def mat_to_quat(R: jnp.ndarray) -> jnp.ndarray:
  """
  Convert a 3x3 rotation matrix to quaternion [w, x, y, z].
  """
  R = R.reshape((3, 3))
  trace = R[0, 0] + R[1, 1] + R[2, 2]
  eps = 1e-8

  def case0():
    return jnp.array([
        (R[2, 1] - R[1, 2]) / (2.0 * jnp.sqrt(jnp.maximum(1.0 + trace, eps))),
        (R[0, 2] - R[2, 0]) / (2.0 * jnp.sqrt(jnp.maximum(1.0 + trace, eps))),
        (R[1, 0] - R[0, 1]) / (2.0 * jnp.sqrt(jnp.maximum(1.0 + trace, eps))),
        jnp.sqrt(jnp.maximum(1.0 + trace, eps)) / 2.0
    ])

  def case1():
    i = jnp.argmax(jnp.diag(R))
    return case0()

  return jax.lax.cond(trace > 0.0, case0, case1)

# -----------------------------------------------------------------------------
# Controller logic
# -----------------------------------------------------------------------------

def pseudo_inverse(M: jnp.ndarray, damped: bool = False, lambda_: float = 0.2) -> jnp.ndarray:
  """Compute (damped) pseudo-inverse in JAX."""
  if not damped:
    return jnp.linalg.pinv(M)
  U, sing_vals, Vt = jnp.linalg.svd(M, full_matrices=False)
  denom = sing_vals**2 + lambda_**2
  S_inv = sing_vals / denom
  return (Vt.T * S_inv) @ U.T

def pd_control(
    x: jnp.ndarray,
    x_des: jnp.ndarray,
    dx: jnp.ndarray,
    kp_kv: jnp.ndarray,
    max_pos_error: float = 0.05,
) -> jnp.ndarray:
  """PD control in position space."""
  x_err = jnp.clip(x - x_des, -max_pos_error, max_pos_error)
  return -kp_kv[:, 0] * x_err - kp_kv[:, 1] * dx

def pd_control_orientation(
    quat: jnp.ndarray,
    quat_des: jnp.ndarray,
    w: jnp.ndarray,
    kp_kv: jnp.ndarray,
    max_ori_error: float = 0.05,
) -> jnp.ndarray:
  """PD control in orientation space (axis-angle)."""
  quat_err = quat_diff_active(quat_des, quat)
  ori_err = jnp.clip(quat_to_axisangle(quat_err), -max_ori_error, max_ori_error)
  return -kp_kv[:, 0] * ori_err - kp_kv[:, 1] * w

def saturate_torque_rate(
    tau_des: jnp.ndarray,
    tau_prev: jnp.ndarray,
    delta_tau_max: float
) -> jnp.ndarray:
  """Rate limit torque change."""
  diff = jnp.clip(tau_des - tau_prev, -delta_tau_max, delta_tau_max)
  return tau_prev + diff

# -----------------------------------------------------------------------------
# Jacobian Computation (JAX version)
# -----------------------------------------------------------------------------

def jac(m, d, point: jnp.ndarray, body_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
  Compute pair of (NV, 3) Jacobians (translational and rotational) of a global point
  attached to a given body.
  
  Note: This uses a helper `scan.body_tree` function (assumed available) to traverse
  the body tree.
  """
  # Build mask: start with a 1 at the target body and then propagate upward.
  fn = lambda carry, b: b if carry is None else b + carry
  mask = (jnp.arange(m.nbody) == body_id) * 1
  mask = scan.body_tree(m, fn, 'b', 'b', mask, reverse=True)
  mask = mask[jnp.array(m.dof_bodyid)] > 0

  offset = point - d.subtree_com[jnp.array(m.body_rootid)[body_id]]
  jacp = jax.vmap(lambda a: a[3:] + jnp.cross(a[:3], offset))(d.cdof)
  jacp = jax.vmap(lambda x, m_: x * m_)(jacp, mask)
  jacr = jax.vmap(lambda a: a[:3])(d.cdof)
  jacr = jax.vmap(lambda x, m_: x * m_)(jacr, mask)
  return jacp, jacr

def jac_site(m, d, site: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
  Compute the Jacobian for a site (both translational and rotational),
  analogous to mj_jacSite.
  """
  # Extract the 3D position for the site (sites are stored as flat arrays: 3 numbers per site)
  point = d.site_xpos[3 * site : 3 * site + 3]
  # Get the body id for the site
  body_id = m.site_bodyid[site]
  return jac(m, d, point, body_id)

# -----------------------------------------------------------------------------
# Main opspace function (JAX-based, with internal jacobian computation)
# -----------------------------------------------------------------------------

def opspace(
    m,                        # Model object containing fields like nbody, dof_bodyid, site_bodyid, body_rootid, etc.
    d,                        # Data object containing fields like site_xpos, site_xmat, qpos, qvel, qfrc_bias, qfrc_actuator, subtree_com, cdof, etc.
    site_id: int,             # Index of the site to control
    dof_ids: jnp.ndarray,     # Array of controlled joint indices (e.g., [0,1,...,6])
    q: jnp.ndarray,           # Current joint positions (for the controlled joints)
    dq: jnp.ndarray,          # Current joint velocities (for the controlled joints)
    tau_J_d: jnp.ndarray,     # Last commanded torques (for rate limiting)
    pos: Optional[jnp.ndarray] = None,
    ori: Optional[jnp.ndarray] = None,
    q_null_des: Optional[jnp.ndarray] = None,
    pos_gains: Union[Tuple[float, ...], jnp.ndarray] = (1500.0, 1500.0, 1500.0),
    ori_gains: Union[Tuple[float, ...], jnp.ndarray] = (200.0, 200.0, 200.0),
    joint_upper_limits: jnp.ndarray = jnp.array([ 2.8 ,  1.7 ,  2.8 , -0.08,  2.8 ,  3.74,  2.8 ]),
    joint_lower_limits: jnp.ndarray = jnp.array([-2.8 , -1.7 , -2.8 , -3.0 , -2.8 , -0.01, -2.8 ]),
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
  """
  JAX-based operational-space controller.
  This version computes the site Jacobians internally (using jac_site) and returns a torque command.
  """
  # Determine desired position/orientation from data if not provided
  x_des = jax.lax.cond(
      pos is None,
      lambda _: d.site_xpos[3 * site_id : 3 * site_id + 3],
      lambda p: p,
      operand=pos
  )
  R_des = jax.lax.cond(
      ori is None,
      lambda _: d.site_xmat[site_id],  # assume d.site_xmat[site_id] is a 3x3 rotation matrix
      lambda o: o,
      operand=ori
  )
  quat_des = mat_to_quat(R_des)
  quat = mat_to_quat(d.site_xmat[site_id])
  q_d_nullspace = jax.lax.cond(
      q_null_des is None,
      lambda _: q,
      lambda qq: qq,
      operand=q_null_des
  )

  # Gains
  kp_pos = jnp.array(pos_gains)
  kd_pos = translational_damping * jnp.ones_like(kp_pos)
  kp_kv_pos = jnp.stack([kp_pos, kd_pos], axis=-1)

  kp_ori = jnp.array(ori_gains)
  kd_ori = rotational_damping * jnp.ones_like(kp_ori)
  kp_kv_ori = jnp.stack([kp_ori, kd_ori], axis=-1)

  # Compute Jacobians for the site using our jac_site helper.
  # Our jac function returns arrays of shape (NV, 3); we transpose them to get (3, NV)
  J_v_full, J_w_full = jac_site(m, d, site_id)
  J_v = J_v_full.T  # shape (3, m.nv)
  J_w = J_w_full.T  # shape (3, m.nv)
  # Select columns corresponding to the controlled joints.
  J_v = J_v[:, dof_ids]
  J_w = J_w[:, dof_ids]

  # PD control in task-space position
  x = d.site_xpos[3 * site_id : 3 * site_id + 3]
  dx = J_v @ dq
  ddx = pd_control(x, x_des, dx, kp_kv_pos, max_pos_error)

  # PD control in task-space orientation (handle sign ambiguity)
  dot_q = jnp.dot(quat, quat_des)
  quat_fix = jax.lax.cond(dot_q < 0.0, lambda q: -q, lambda q: q, quat)
  w = J_w @ dq
  dw = pd_control_orientation(quat_fix, quat_des, w, kp_kv_ori, max_ori_error)

  # Combine translational and rotational feedback
  F_task = jnp.concatenate([ddx, dw])  # shape (6,)
  J_full = jnp.vstack([J_v, J_w])        # shape (6, number of controlled dofs)
  tau_task = J_full.T @ F_task           # shape (number of controlled dofs,)

  # Nullspace PD control
  q_error = q_d_nullspace - q
  dq_error = dq
  q_error = q_error.at[0].set(q_error[0] * joint1_nullspace_stiffness)
  dq_error = dq_error.at[0].set(dq_error[0] * (2.0 * jnp.sqrt(joint1_nullspace_stiffness)))
  tau_nullspace = nullspace_stiffness * q_error - 2.0 * jnp.sqrt(nullspace_stiffness) * dq_error

  # Optionally apply damping in the nullspace
  def damped_null_fn(tau_ns):
    jT_pinv = pseudo_inverse(J_full.T, damped=True, lambda_=lambda_)
    proj = jnp.eye(J_full.shape[1]) - (J_full.T @ jT_pinv)
    return proj @ tau_ns

  tau_nullspace = jax.lax.cond(damped, damped_null_fn, lambda x: x, tau_nullspace)
  tau = tau_task + tau_nullspace

  # Enforce joint limits: if a joint is at (or beyond) its limit, zero out torque pushing it further.
  def clip_torque_at_limits(args):
    i, tau_i, q_i, q_upper, q_lower = args
    tau_i = jax.lax.cond(
        (q_i >= q_upper) & (tau_i > 0.0),
        lambda _: 0.0,
        lambda _: tau_i,
        operand=None
    )
    tau_i = jax.lax.cond(
        (q_i <= q_lower) & (tau_i < 0.0),
        lambda _: 0.0,
        lambda _: tau_i,
        operand=None
    )
    return tau_i

  tau_clipped = jax.vmap(
      clip_torque_at_limits, in_axes=(0, 0, 0, 0, 0), out_axes=0
  )(jnp.arange(q.shape[0]), tau, q, joint_upper_limits, joint_lower_limits)

  # Apply gravity compensation if desired.
  tau_gc = jax.lax.cond(gravity_comp, lambda: tau_clipped + d.qfrc_bias[dof_ids], lambda: tau_clipped)
  # Rate limit the commanded torques.
  tau_final = saturate_torque_rate(tau_gc, tau_J_d, delta_tau_max)

  return tau_final
