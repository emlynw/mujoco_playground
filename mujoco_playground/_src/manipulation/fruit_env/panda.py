from typing import Any, Dict, Optional, Union

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
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
  path = mjx_env.ROOT_PATH / "manipulation" / "fruit_env" / "xmls"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  mjx_env.update_assets(assets, path / "textures")
  mjx_env.update_assets(assets, path / "textures"/ "skyboxes")
  print("Assets loaded from:", path)
  return assets

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.04,
  )

class PandaBase(mjx_env.MjxEnv):
  """Environment for picking strawberries."""

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



  def _post_init(self):
    self._PANDA_HOME = np.array([0.0, -1.6, 0.0, -2.54, -0.05, 2.49, 0.822], dtype=np.float32)
    self._GRIPPER_HOME = np.array([0.0141, 0.0141], dtype=np.float32)
    self._ROBOT_HOME = np.concatenate((self._PANDA_HOME, self._GRIPPER_HOME), axis=0)
    self._GRIPPER_MIN = 0.0
    self._GRIPPER_MAX = 0.007
    self._PANDA_XYZ = np.array([0.1, 0, 0.8], dtype=np.float32)
    self._CARTESIAN_BOUNDS = np.array([[0.05, -0.2, 0.6], [0.55, 0.2, 0.95]], dtype=np.float32)
    self._ROTATION_BOUNDS = np.array([[-np.pi/3, -np.pi/6, -np.pi/10],[np.pi/3, np.pi/6, np.pi/10]], dtype=np.float32)
    self.default_obj_pos = np.array([0.42, 0, 0.95])
    self.gripper_sleep = 0.6
    self.initial_position = np.array([0.1, 0.0, 0.75], dtype=np.float32)
    self.initial_orientation = np.array([0.725, 0.0, 0.688, 0.0])

    
    all_joints = _ARM_JOINTS + _FINGER_JOINTS
    self._panda_ctrl_ids = np.array([self._mj_model.actuator(f"actuator{j}").id for j in range(1, 8)])
    self._gripper_ctrl_id = self._mj_model.actuator("fingers_actuator").id
    self._gripper_site = self._mj_model.site("long_pinch").id

    self._robot_arm_qposadr = np.array([self._mj_model.jnt_qposadr[self._mj_model.joint(j).id] for j in _ARM_JOINTS])
    self._robot_qposadr = np.array([self._mj_model.jnt_qposadr[self._mj_model.joint(j).id] for j in all_joints])

    self._obj_body = self._mj_model.body("vine1").id
    self._obj_qposadr = self._mj_model.jnt_qposadr[self._mj_model.body("vine1").jntadr[0]]
    self._mocap_target = self._mj_model.body("long_pinch").mocapid
    self._floor_geom = self._mj_model.geom("floor").id
    self._init_q = self._mj_model.qpos0
    self._init_obj_pos = jp.array(self.default_obj_pos, dtype=jp.float32)
    self._initial_position = jp.array(self.initial_position, dtype=jp.float32)
    self._initial_orientation = jp.roll(self.initial_orientation, 1)
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