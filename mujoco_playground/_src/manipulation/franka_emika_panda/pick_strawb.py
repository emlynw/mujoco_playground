# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple task with demonstrating sim2real transfer for pixels observations.
Pick up a cube to a fixed location using a cartesian controller."""

from typing import Any, Dict, Optional, Union
import warnings

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda_serl
from mujoco_playground._src.manipulation.franka_emika_panda.panda_kinematics_serl import opspace

_ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]
_FINGER_JOINTS = ["finger_joint1", "finger_joint2"]

def get_assets() -> Dict[str, bytes]:
  assets = {}
  path = mjx_env.ROOT_PATH / "manipulation" / "franka_emika_panda" / "xmls" / "panda_serl"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "textures")
  mjx_env.update_assets(assets, path / "assets")
  mjx_env.update_assets(assets, path / "textures" / "skyboxes")
  return assets


def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      gpu_id=0,
      render_batch_size=4,
      render_width=128,
      render_height=128,
      use_rasterizer=False,
      enabled_geom_groups=[0, 1, 2],
  )


def default_config():
  config = config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.005,
      episode_length=200,
      action_repeat=1,
      # Size of cartesian increment.
      action_scale=0.005,
      reward_config=config_dict.create(
          reward_scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=4.0,
              # Box goes to the target mocap.
              box_target=8.0,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25,
              # Do not collide cube with gripper
              no_box_collision=0.05,
              # Destabilizes training in cartesian action space.
              robot_target_qpos=0.0,
          ),
          action_rate=-0.0005,
          no_soln_reward=-0.01,
          lifted_reward=0.5,
          success_reward=2.0,
      ),
      vision=False,
      vision_config=default_vision_config(),
      obs_noise=config_dict.create(brightness=[1.0, 1.0]),
      box_init_range=0.05,
      success_threshold=0.05,
      action_history_length=1,
  )
  return config


def adjust_brightness(img, scale):
  """Adjusts the brightness of an image by scaling the pixel values."""
  return jp.clip(img * scale, 0, 1)


class PandaPickStrawb(panda_serl.PandaBaseSERL):
  """Environment for training the Franka Panda robot to pick a strawberry."""

  def __init__(  # pylint: disable=non-parent-init-called,super-init-not-called
      self,
      config=default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):

    mjx_env.MjxEnv.__init__(self, config, config_overrides)
    self._vision = config.vision

    xml_path = (
        mjx_env.ROOT_PATH
        / 'manipulation'
        / 'franka_emika_panda'
        / 'xmls'
        / 'panda_serl'
        / 'mjmodel.xml'
    )
    self._xml_path = xml_path.as_posix()

    mj_model = self.modify_model(
        mujoco.MjModel.from_xml_string(
            xml_path.read_text(), assets=panda_serl.get_assets()
        )
    )
    mj_model.opt.timestep = config.sim_dt

    self._mj_model = mj_model
    self._mjx_model = mjx.put_model(mj_model)

    self._PANDA_HOME = np.array([0.0, -1.6, 0.0, -2.54, -0.05, 2.49, 0.822], dtype=np.float32)
    self._GRIPPER_HOME = np.array([0.0141, 0.0141], dtype=np.float32)
    self._GRIPPER_MIN = 0.0
    self._GRIPPER_MAX = 0.007
    self._PANDA_XYZ = np.array([0.1, 0, 0.8], dtype=np.float32)
    self._CARTESIAN_BOUNDS = np.array([[0.05, -0.2, 0.6], [0.55, 0.2, 0.95]], dtype=np.float32)
    self._ROTATION_BOUNDS = np.array([[-np.pi/3, -np.pi/6, -np.pi/10],[np.pi/3, np.pi/6, np.pi/10]], dtype=np.float32)
    self.default_obj_pos = np.array([0.42, 0, 0.95])
    self.initial_position = np.array([0.1, 0.0, 0.75], dtype=np.float32)
    self.initial_orientation = [0.725, 0.0, 0.688, 0.0]

    # Set gripper in sight of camera
    self._post_init()

    if self._vision:
      try:
        # pylint: disable=import-outside-toplevel
        from madrona_mjx.renderer import BatchRenderer  # pytype: disable=import-error
      except ImportError:
        warnings.warn(
            'Madrona MJX not installed. Cannot use vision with'
            ' PandaPickCubeCartesian.'
        )
        return
      self.renderer = BatchRenderer(
          m=self._mjx_model,
          gpu_id=self._config.vision_config.gpu_id,
          num_worlds=self._config.vision_config.render_batch_size,
          batch_render_view_width=self._config.vision_config.render_width,
          batch_render_view_height=self._config.vision_config.render_height,
          enabled_geom_groups=np.asarray(
              self._config.vision_config.enabled_geom_groups
          ),
          enabled_cameras=None,  # Use all cameras.
          add_cam_debug_geo=False,
          use_rasterizer=self._config.vision_config.use_rasterizer,
          viz_gpu_hdls=None,
      )

  def _post_init(self) -> None:
    all_joints = _ARM_JOINTS + _FINGER_JOINTS
    self._robot_arm_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in _ARM_JOINTS
    ])
    self._robot_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in all_joints
    ])
    self._gripper_site = self._mj_model.site("pinch").id
    self._mocap_target = self._mj_model.body("target").mocapid
    self._floor_geom = self._mj_model.geom("floor").id
    self.init_q = np.concatenate([self._PANDA_HOME, self._GRIPPER_HOME])
    self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

  def modify_model(self, mj_model: mujoco.MjModel):
    # Expand floor size to non-zero so Madrona can render it
    mj_model.geom_size[mj_model.geom('floor').id, :2] = [5.0, 5.0]

    return mj_model

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Resets the environment to an initial state."""

    # Fixed target position to simplify pixels-only training.
    target_pos = self.initial_position

    # initialize pipeline state
    data = mjx_env.init(
        self._mjx_model,
        jp.array(self.init_q),
        jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
    )

    target_quat = jp.array(self.initial_orientation, dtype=float)
    data = data.replace(
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat)
    )
    # if not self._vision:
    #   # mocap target should not appear in the pixels observation.
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos)
    )

    # initialize env state and info
    metrics = {
        'out_of_bounds': jp.array(0.0),
        **{
            f'reward/{k}': 0.0
            for k in self._config.reward_config.reward_scales.keys()
        },
    }

    info = {
        'rng': rng,
        'target_pos': target_pos,
        'reached_box': jp.array(0.0, dtype=float),
        'prev_reward': jp.array(0.0, dtype=float),
        'current_pos': jp.array(self.data.sensor("pinch_pos").data),
        'newly_reset': jp.array(False, dtype=bool),
        'prev_action': jp.zeros(3),
        '_steps': jp.array(0, dtype=int),
    }

    reward, done = jp.zeros(2)

    obs = self._get_obs(data, info)
    obs = jp.concat([obs, jp.zeros(1), jp.zeros(3)], axis=0)
    if self._vision:
      rng_brightness, rng = jax.random.split(rng)
      brightness = jax.random.uniform(
          rng_brightness,
          (1,),
          minval=self._config.obs_noise.brightness[0],
          maxval=self._config.obs_noise.brightness[1],
      )
      info.update({'brightness': brightness})

      render_token, rgb, _ = self.renderer.init(data, self._mjx_model)
      info.update({'render_token': render_token})

      obs = jp.asarray(rgb[0][..., :3], dtype=jp.float32) / 255.0
      obs = adjust_brightness(obs, brightness)
      obs = {'pixels/view_0': obs}

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Runs one timestep of the environment's dynamics."""
    # Add action delay
    state.info['rng'], key = jax.random.split(state.info['rng'])

    state.info['newly_reset'] = state.info['_steps'] == 0

    newly_reset = state.info['newly_reset']
    state.info['prev_reward'] = jp.where(
        newly_reset, 0.0, state.info['prev_reward']
    )
    state.info['current_pos'] = jp.array(self.data.sensor("pinch_pos").data)
    state.info['reached_box'] = jp.where(
        newly_reset, 0.0, state.info['reached_box']
    )
    state.info['prev_action'] = jp.where(
        newly_reset, jp.zeros(3), state.info['prev_action']
    )

    # Cartesian control
    increment = jp.zeros(4)
    increment = increment.at[1:].set(action)  # set y, z and gripper commands.
    ctrl, new_tip_position, no_soln = self._move_tip(
        state.info['current_pos'],
        self._start_tip_transform[:3, :3],
        data.ctrl,
        increment,
    )
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)
    state.info.update({'current_pos': new_tip_position})

    # Simulator step
    data = mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps)

    # Dense rewards
    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.reward_scales[k]
        for k, v in raw_rewards.items()
    }

    # Penalize collision with box.
    hand_box = collision.geoms_colliding(data, self._box_geom, self._hand_geom)
    raw_rewards['no_box_collision'] = jp.where(hand_box, 0.0, 1.0)

    total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

    if not self._vision:
      # Vision policy cannot access the required state-based observations.
      da = jp.linalg.norm(action - state.info['prev_action'])
      state.info['prev_action'] = action
      total_reward += self._config.reward_config.action_rate * da
      total_reward += no_soln * self._config.reward_config.no_soln_reward

    # Sparse rewards
    box_pos = data.xpos[self._obj_body]
    total_reward += (
        box_pos[2] > 0.05
    ) * self._config.reward_config.lifted_reward
    success = self._get_success(data, state.info)
    total_reward += success * self._config.reward_config.success_reward

    # Reward progress
    reward = jp.maximum(
        total_reward - state.info['prev_reward'], jp.zeros_like(total_reward)
    )
    state.info['prev_reward'] = jp.maximum(
        total_reward, state.info['prev_reward']
    )
    reward = jp.where(newly_reset, 0.0, reward)  # Prevent first-step artifact

    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    state.metrics.update(out_of_bounds=out_of_bounds.astype(float))
    state.metrics.update({f'reward/{k}': v for k, v in raw_rewards.items()})

    done = (
        out_of_bounds
        | jp.isnan(data.qpos).any()
        | jp.isnan(data.qvel).any()
        | success
    )

    # Ensure exact sync between newly_reset and the autoresetwrapper.
    state.info['_steps'] += self._config.action_repeat
    state.info['_steps'] = jp.where(
        done | (state.info['_steps'] >= self._config.episode_length),
        0,
        state.info['_steps'],
    )

    obs = self._get_obs(data, state.info)
    obs = jp.concat([obs, no_soln.reshape(1), action], axis=0)
    if self._vision:
      _, rgb, _ = self.renderer.render(state.info['render_token'], data)
      obs = jp.asarray(rgb[0][..., :3], dtype=jp.float32) / 255.0
      obs = adjust_brightness(obs, state.info['brightness'])
      obs = {'pixels/view_0': obs}

    return state.replace(
        data=data,
        obs=obs,
        reward=reward,
        done=done.astype(float),
        info=state.info,
    )

  def _get_success(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    box_pos = data.xpos[self._obj_body]
    target_pos = info['target_pos']
    if (
        self._vision
    ):  # Randomized camera positions cannot see location along y line.
      box_pos, target_pos = box_pos[2], target_pos[2]
    return jp.linalg.norm(box_pos - target_pos) < self._config.success_threshold

  def _move_tip(
      self,
      current_tip_pos: jax.Array,
      current_tip_rot: jax.Array,
      current_ctrl: jax.Array,
      action: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Calculate new tip position from cartesian increment."""
    # Discrete gripper action where a < 0 := closed
    close_gripper = jp.where(action[3] < 0, 1.0, 0.0)

    scaled_pos = action[:3] * self._config.action_scale
    new_tip_pos = current_tip_pos.at[:3].add(scaled_pos)

    new_ctrl = current_ctrl

    new_tip_pos = new_tip_pos.at[0].set(jp.clip(new_tip_pos[0], 0.25, 0.77))
    new_tip_pos = new_tip_pos.at[1].set(jp.clip(new_tip_pos[1], -0.32, 0.32))
    new_tip_pos = new_tip_pos.at[2].set(jp.clip(new_tip_pos[2], 0.02, 0.5))

    new_tip_mat = jp.identity(4)
    new_tip_mat = new_tip_mat.at[:3, :3].set(current_tip_rot)
    new_tip_mat = new_tip_mat.at[:3, 3].set(new_tip_pos)

    out_jp = panda_kinematics.compute_franka_ik(
        new_tip_mat, current_ctrl[6], current_ctrl[:7]
    )
    # mujoco.mj_jacSite(model, data, J_v, J_w, site_id)
    J_v = jp.zeros((3, self._mjx_model.nv), dtype=float)
    J_w = jp.zeros((3, self._mjx_model.nv), dtype=float)
    # site_id = self._mjx_model.site('gripper').id
    jax.debug.print(f"jac: {mjx.mjx_jacsite(self._mjx_model, J_v, J_w, 0)}")
    no_soln = jp.any(jp.isnan(out_jp))
    out_jp = jp.where(no_soln, current_ctrl[:7], out_jp)
    no_soln = jp.logical_or(no_soln, jp.any(jp.isnan(out_jp)))
    new_tip_pos = jp.where(
        jp.any(jp.isnan(out_jp)), current_tip_pos, new_tip_pos
    )

    new_ctrl = new_ctrl.at[:7].set(out_jp)
    jaw_action = jp.where(close_gripper, -1.0, 1.0)
    claw_delta = jaw_action * 0.02  # up to 2 cm movement per ctrl.
    new_ctrl = new_ctrl.at[7].set(new_ctrl[7] + claw_delta)

    return new_ctrl, new_tip_pos, no_soln
  
  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    target_pos = info["target_pos"]
    box_pos = data.xpos[self._obj_body]
    gripper_pos = data.site_xpos[self._gripper_site]
    pos_err = jp.linalg.norm(target_pos - box_pos)
    box_mat = data.xmat[self._obj_body]
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])

    box_target = 1 - jp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
    gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))
    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )

    # Check for collisions with the floor
    hand_floor_collision = [
        collision.geoms_colliding(data, self._floor_geom, g)
        for g in [
            self._left_finger_geom,
            self._right_finger_geom,
            self._hand_geom,
        ]
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = (1 - floor_collision).astype(float)

    info["reached_box"] = 1.0 * jp.maximum(
        info["reached_box"],
        (jp.linalg.norm(box_pos - gripper_pos) < 0.012),
    )

    rewards = {
        "gripper_box": gripper_box,
        "box_target": box_target * info["reached_box"],
        "no_floor_collision": no_floor_collision,
        "robot_target_qpos": robot_target_qpos,
    }
    return rewards

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
        info["target_pos"] - data.xpos[self._obj_body],
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])

    return obs

  @property
  def observation_size(self) -> mjx_env.ObservationSize:
    ret = {'pixels/view_0': (64, 64, 3)} if self._vision else 70
    return ret

  @property
  def action_size(self) -> int:
    return 3

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
