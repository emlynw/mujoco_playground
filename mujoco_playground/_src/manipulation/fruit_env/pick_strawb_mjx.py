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
"""Fruit Env mujoco playground"""

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
from mujoco_playground._src.manipulation.fruit_env import panda
from mujoco.mjx._src import math


def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      gpu_id=0,
      render_batch_size=1024,
      render_width=64,
      render_height=64,
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
      vision=True,
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


class PandaPickStrawb(panda.PandaBase):
  """Environment for training the Franka Panda robot to pick up a cube in
  Cartesian space."""

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
        / 'fruit_env'
        / 'xmls'
        / 'mjmodel.xml'
    )
    self._xml_path = xml_path.as_posix()
    print('Loading XML from:', xml_path)
    mj_model = self.modify_model(
        mujoco.MjModel.from_xml_string(
            xml_path.read_text(), assets=panda.get_assets()
        )
    )
    mj_model.opt.timestep = config.sim_dt

    self._mj_model = mj_model
    self._mjx_model = mjx.put_model(mj_model)

    # Set gripper in sight of camera
    self._action_scale = config.action_scale    
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

  def _post_init(self):
    super()._post_init()
    
  def modify_model(self, mj_model: mujoco.MjModel):
    # Expand floor size to non-zero so Madrona can render it
    # mj_model.geom_size[mj_model.geom('floor').id, :2] = [5.0, 5.0]
    return mj_model

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Resets the environment to an initial state."""

    # intialize box position
    rng, rng_box = jax.random.split(rng)
    r_range = self._config.box_init_range
    box_noise = jax.random.uniform(rng_box, (3,), minval=-r_range, maxval=r_range)
    ee_noise = jax.random.uniform(rng_box, (3,), minval=-r_range, maxval=r_range)

    box_pos = self._init_obj_pos + box_noise
    ee_pos = self._initial_position + ee_noise

    # initialize pipeline state
    init_q = (jp.array(self._init_q).at[self._obj_qposadr : self._obj_qposadr + 3].set(box_pos))
    init_q = init_q.at[self._robot_qposadr].set(self._ROBOT_HOME)

    data = mjx_env.init(
        self._mjx_model,
        init_q,
        jp.zeros(self._mjx_model.nv, dtype=float),
    )

    data = data.replace(mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(self._initial_orientation))
    data = data.replace(mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(ee_pos))

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
        'reached_box': jp.array(0.0, dtype=float),
        'prev_reward': jp.array(0.0, dtype=float),
        'current_pos': ee_pos,
        'newly_reset': jp.array(False, dtype=bool),
        'prev_action': jp.zeros(3),
        '_steps': jp.array(0, dtype=int),
        'action_history': jp.zeros((
            self._config.action_history_length,
        )),  # Gripper only
    }

    reward, done = jp.zeros(2)

    obs = self._get_obs(data, info)
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

    current_pos = data.mocap_pos[self._mocap_target]
    delta_pos = action[:3]
    new_pos = current_pos + delta_pos
    # replace the gripper position with the new position
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(new_pos)
    )

    ctrl = jp.zeros_like(action)
    # Simulator step
    data = mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps)

    done = False

    # Ensure exact sync between newly_reset and the autoresetwrapper.
    state.info['_steps'] += self._config.action_repeat
    state.info['_steps'] = jp.where(
        done | (state.info['_steps'] >= self._config.episode_length),
        0,
        state.info['_steps'],
    )

    reward = jp.zeros(1)

    obs = self._get_obs(data, state.info)
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
  
  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    gripper_width = data.qpos[self._robot_qposadr[-1]]
    target_pos = data.mocap_pos[self._mocap_target]
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = {}
    obs['gripper_pos'] = gripper_pos
    obs['gripper_mat'] = gripper_mat
    obs['gripper_width'] = gripper_width
    obs['target_pos'] = target_pos
    obs['target_mat'] = target_mat.ravel()
    

    return obs

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
