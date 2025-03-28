# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.``
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
"""Franka Emika Panda base class."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env

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
  mjx_env.update_assets(assets, path, "mjmodel.xml")
  mjx_env.update_assets(assets, path / "textures")
  mjx_env.update_assets(assets, path / "assets")
  mjx_env.update_assets(assets, path / "textures" / "skyboxes")
  return assets


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.04,
  )


class PandaBaseSERL(mjx_env.MjxEnv):
  """Base environment for Franka Emika Panda."""

  def __init__(
      self,
      xml_path: epath.Path,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)

    self._xml_path = xml_path.as_posix()
    xml = xml_path.read_text()
    mj_model = mujoco.MjModel.from_xml_string(xml, assets=get_assets())
    mj_model.opt.timestep = self.sim_dt

    self._mj_model = mj_model
    self._mjx_model = mjx.put_model(mj_model)
    self._action_scale = config.action_scale

    self._PANDA_HOME = np.array([0.0, -1.6, 0.0, -2.54, -0.05, 2.49, 0.822], dtype=np.float32)
    self._GRIPPER_HOME = np.array([0.0141, 0.0141], dtype=np.float32)
    self._GRIPPER_MIN = 0.0
    self._GRIPPER_MAX = 0.007
    self._PANDA_XYZ = np.array([0.1, 0, 0.8], dtype=np.float32)
    self._CARTESIAN_BOUNDS = np.array([[0.05, -0.2, 0.6], [0.55, 0.2, 0.95]], dtype=np.float32)
    self._ROTATION_BOUNDS = np.array([[-np.pi/3, -np.pi/6, -np.pi/10],[np.pi/3, np.pi/6, np.pi/10]], dtype=np.float32)
    self.default_obj_pos = np.array([0.42, 0, 0.95])

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

  @property
  def xml_path(self) -> str:
    raise self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
