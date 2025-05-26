import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Union
import yaml
import mujoco
import random
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box, Dict
from scipy.spatial.transform import Rotation
from fruit_gym.controllers.opspace import opspace
from fruit_gym.envs.randomization import (
    lighting_noise,
    action_scale_noise,
    initial_state_noise,
    camera_noise,
    floor_noise,
    skybox_noise,
)

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

class PickStrawbEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description

    **PickStrawbEnv** is a robotic manipulation environment in which a Franka Panda robot must reach for and grasp a red strawberries among 
    multiple distractor (green) strawberries. The environment supports domain randomization, image observations, and multi-target grasping.
    This environment is designed to challenge reinforcement learning algorithms in continuous control and multi-object manipulation tasks.

    ## Action Space

    The action space is continuous and is defined as a `Box` with shape `(ee_dof + 1,)` of type `float32`. For example, if `ee_dof=6`, then the action space is: Box(-1, 1, (7,), float32)


    An action is represented as:  
    `[z, y, x, roll, pitch, yaw, grasp], in the end effector frame. z and x swapped for intuitive first person control`

    | Num | Action Component   | Description                                                                                                   | Range   |
    |-----|--------------------|---------------------------------------------------------------------------------------------------------------|---------|
    | 0   | dz                  | Displacement along the z-axis (vertical movement)                                                             | [-1, 1] |
    | 1   | dy                  | Displacement along the y-axis                                                                                 | [-1, 1] |
    | 2   | dx                  | Displacement along the x-axis                                                                                 | [-1, 1] |
    | 3   | droll               | Rotation about the x-axis (roll)                                                                              | [-1, 1] |
    | 4   | dpitch              | Rotation about the y-axis (pitch)                                                                             | [-1, 1] |
    | 5   | dyaw                | Rotation about the z-axis (yaw)                                                                               | [-1, 1] |
    | 6   | dgrasp              | Grasp command. Values above a threshold indicate an attempt to grasp; below indicate release.                 | [-1, 1] |

    ## Observation Space

    The observation space is a dictionary containing both a state vector and (optionally) image observations.

    ### State

    The state is a dictionary with the following keys:

    - **tcp_pose (7 elements):**  
    The position (3 elements) and orientation (quaternion, 4 elements) of the robot’s end-effector.
    
    - **tcp_vel (6 elements):**  
    The linear and angular velocities of the end-effector.
    
    - **gripper_pos (1 element):**  
    The current opening of the gripper.
    
    - **gripper_vec (4 elements):**  
    A vector representing the gripper state (e.g., one-hot encoding of open, closed, etc.).

    Thus, the overall state vector has 7 + 6 + 1 + 4 = 18 elements.

    ### Images

    If `image_obs=True`, the observation also includes an `images` dictionary mapping camera names (e.g., `"wrist1"`, `"wrist2"`, `"front"`) to RGB images of 
    shape `(height, width, 3)` with values in `[0, 255]`.

    ## Rewards

    The total reward is a weighted sum of several components:
    - **r_red:** A positive reward proportional to how close the end-effector is to the red strawberry target.
    - **r_green:** A penalty based on the movement of green (distractor) objects from their initial positions.
    - **r_grasp:** A binary reward given when both gripper fingers make contact with the desired strawberry stem.
    - **r_energy:** A penalty proportional to the magnitude of the action, encouraging energy-efficient control.
    - **r_time:** A constant penalty at each timestep to encourage the agent to filinish the episodes quickly.
    - **r_smooth:** A penalty for large changes in actions between consecutive steps to promote smooth control.
    - **r_bad_grasp:** A penalty for grasping the wrong things.

    ## Starting State

    At the beginning of each episode, the robot arm and gripper are reset to their home positions. The positions and orientations of the target vine (holding the strawberries) 
    and the distractor vines are randomized within predefined bounds. Domain randomization may also be applied to lighting, camera parameters, and object properties as specified in a configuration file.

    ## Episode End

    ### Termination

    An episode terminates when either all red strawberry targets have been grasped (i.e., removed from the environment) or an external time limit is reached.

    ### Truncation

    Truncation is managed by the Gymnasium `TimeLimit` wrapper, and is not handled intrinsically by the environment.

    ## Arguments

    The environment accepts a variety of parameters upon instantiation:

    | Parameter            | Type            | Default                        | Description                                                                                       |
    |----------------------|-----------------|--------------------------------|---------------------------------------------------------------------------------------------------|
    | `image_obs`          | bool            | True                           | Whether to include image observations.                                                            |
    | `randomize_domain`   | bool            | True                           | Whether to apply domain randomization to lighting, camera, and object properties.                 |
    | `ee_dof`             | int             | 6                              | Degrees of freedom for the end-effector (3 for position only; 6 for position and orientation).    |
    | `control_dt`         | float           | 0.05                           | Time interval between control updates.                                                            |
    | `physics_dt`         | float           | 0.002                          | Simulation time step.                                                                             |
    | `width`              | int             | 480                            | Image width (if `image_obs` is True).                                                             |
    | `height`             | int             | 480                            | Image height (if `image_obs` is True).                                                            |
    | `pos_scale`          | float           | 0.008                          | Scaling factor for positional changes.                                                            |
    | `rot_scale`          | float           | 0.5                            | Scaling factor for rotational changes.                                                            |
    | `cameras`            | List[str]       | ["wrist1", "wrist2", "front"]  | List of camera names for rendering images.                                                        |
    | `reward_type`        | str             | "dense"                        | Reward type; can be "dense" or "sparse".                                                          |
    | `gripper_pause`      | bool            | False                          | If True, the simulation pauses briefly after a gripper action.                                    |
    | `render_mode`        | str             | "rgb_array"                    | Rendering mode, e.g., "human" or "rgb_array".                                                     |

    """
    metadata = { 
        "render_modes": ["human", "rgb_array", "depth_array"], 
    }
    
    def __init__(
        self,
        image_obs: bool = True,
        randomize_domain: bool = True,
        ee_dof: int = 6, # 3 for position, 3 for orientation
        control_dt: float = 0.05,
        physics_dt: float = 0.002,
        width: int = 480,
        height: int = 480,
        pos_scale: float = 0.008,
        rot_scale: float = 0.5,
        cameras: List[str] = None,
        reward_type: str = "dense",
        gripper_pause: bool = False,
        render_mode: str = "rgb_array",
        config_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, image_obs=image_obs, **kwargs)

        if cameras is None:
            cameras = ["wrist1", "wrist2", "front"]

        self.image_obs = image_obs
        self.randomize_domain = randomize_domain
        self.ee_dof = ee_dof
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.cameras = cameras
        self.reward_type = reward_type
        self.gripper_pause = gripper_pause

        self._PANDA_HOME = np.array([0.0, -1.6, 0.0, -2.54, -0.05, 2.49, 0.822], dtype=np.float32)
        self._GRIPPER_HOME = np.array([0.0141, 0.0141], dtype=np.float32)
        self._GRIPPER_MIN = 0.0
        self._GRIPPER_MAX = 0.007
        self._PANDA_XYZ = np.array([0.1, 0, 0.8], dtype=np.float32)
        self._CARTESIAN_BOUNDS = np.array([[0.05, -0.2, 0.6], [0.55, 0.2, 0.95]], dtype=np.float32)
        self._ROTATION_BOUNDS = np.array([[-np.pi/3, -np.pi/6, -np.pi/10],[np.pi/3, np.pi/6, np.pi/10]], dtype=np.float32)
        self.default_obj_pos = np.array([0.42, 0, 0.95])
        self.gripper_sleep = 0.6

        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "strawb_hanging.yaml"
        self.cfg = load_config(config_path)

        state_space = Dict(
            {
                "tcp_pose": Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "tcp_vel": Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "gripper_pos": Box(-1, 1, shape=(1,), dtype=np.float32),
                "gripper_vec": Box(0.0, 1.0, shape=(4,), dtype=np.float32),
            }
        )
        if not image_obs:
            state_space["block_pos"] = Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.observation_space = Dict({"state": state_space})
        if image_obs:
            self.observation_space["images"] = Dict()
            for camera in self.cameras:
                self.observation_space["images"][camera] = Box(
                    0, 255, shape=(self.height, self.width, 3), dtype=np.uint8
                )

        p = Path(__file__).parent
        env_dir = os.path.join(p, "xmls/mjmodel.xml")
        self._n_substeps = int(float(control_dt) / float(physics_dt))
        self.frame_skip = 1
        MujocoEnv.__init__(
            self, 
            env_dir, 
            self.frame_skip, 
            observation_space=self.observation_space, 
            render_mode=self.render_mode,
            width=self.width,
            height=self.height, 
            camera_id=0, 
            **kwargs,
        )
        self.model.opt.timestep = physics_dt
        self.camera_id = ()
        for cam in self.cameras:
            self.camera_id += (self.model.camera(cam).id,)
        self.action_space = Box(
            np.array([-1.0]*(self.ee_dof+1)), 
            np.array([1.0]*(self.ee_dof+1)),
            dtype=np.float32,
        )
        self._viewer = MujocoRenderer(self.model, self.data,)
        self.setup()

    def setup(self):

        self._panda_dof_ids = np.array([self.model.joint(f"joint{i}").id for i in range(1, 8)])
        self._panda_ctrl_ids = np.array([self.model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._gripper_ctrl_id = self.model.actuator("fingers_actuator").id
        self._pinch_site_id = self.model.site("pinch").id

        self.prev_action = np.zeros(self.action_space.shape)
        self.prev_grasp_time = 0.0
        self.prev_grasp = 0.0
        # Gripper state encoding
        self.gripper_dict = {
            "open": np.array([1, 0, 0, 0], dtype=np.float32),
            "closed": np.array([0, 1, 0, 0], dtype=np.float32),
            "opening": np.array([0, 0, 1, 0], dtype=np.float32),
            "closing": np.array([0, 0, 0, 1], dtype=np.float32),
        }

        self.reset_arm_and_gripper()

        # Store initial values for randomization
        for camera_name in self.cameras:
            setattr(self, f"{camera_name}_pos", self.model.body_pos[self.model.body(camera_name).id].copy())
            setattr(self, f"{camera_name}_quat", self.model.body_quat[self.model.body(camera_name).id].copy())

        self.init_light_pos = self.model.body_pos[self.model.body('light0').id].copy()
        self.skybox_tex_ids = []
        self.floor_tex_ids = []

        for i in range(self.model.ntex):
            if i < self.model.ntex - 1:
                # For all but the last texture, use the next index
                name_start = self.model.name_texadr[i]
                name_end = self.model.name_texadr[i + 1] - 1
            else:
                # For the last texture, go until the first null byte or the end of the names array
                name_start = self.model.name_texadr[i]
                name_end = len(self.model.names)
            # Decode the name slice
            texture_name = self.model.names[name_start:name_end].split(b'\x00', 1)[0].decode('utf-8')
            if self.model.texture(texture_name).type[0] == 2:
                self.skybox_tex_ids.append(self.model.texture(texture_name).id)
            else:
                self.floor_tex_ids.append(self.model.texture(texture_name).id)

        self.initial_vine_rotation = Rotation.from_quat(np.roll(self.model.body_quat[self.model.body("vine1").id], -1))

        self.initial_position = np.array([0.1, 0.0, 0.75], dtype=np.float32)
        self.initial_orientation = [0.725, 0.0, 0.688, 0.0]
        self.initial_rotation = Rotation.from_quat(self.initial_orientation)

        self.init_headlight_diffuse = self.model.vis.headlight.diffuse.copy()
        self.init_headlight_ambient = self.model.vis.headlight.ambient.copy()
        self.init_headlight_specular = self.model.vis.headlight.specular.copy()

        self.num_green = 7
        self.model.body_pos[self.model.body("vine1").id] = self.default_obj_pos
        for i in range(2, self.num_green+2):
            self.model.body_pos[self.model.body(f"vine{i}").id] = self.default_obj_pos + np.array([-0.05, 0.0, 0.0])
        self.active_indices = np.array(list(range(2, self.num_green + 2)))

    def object_noise(self):
        dr = self.cfg.get("domain_randomization", {})
        object_cfg = dr.get("objects", {})
        if not object_cfg.get("enabled", False):
            return
        num_green = object_cfg.get("num_green", 7)
        # Target pos
        target_pos_noise_low = object_cfg.get("target_pos_noise_low", [0.0, 0.0, 0.0])
        target_pos_noise_high = object_cfg.get("target_pos_noise_high", [0.0, 0.0, 0.0])
        target_pos_noise = np.random.uniform(low=target_pos_noise_low, high=target_pos_noise_high, size=3)
        # target_pos = self.default_obj_pos + target_pos_noise
        # self.model.body_pos[self.model.body("vine1").id] = target_pos
        target_pos = self.data.mocap_pos[0].copy()
        target_pos[0] = target_pos[0] + 0.2
        target_pos[2] = target_pos[2] + 0.1
        target_pos = target_pos + target_pos_noise
        self.model.body_pos[self.model.body("vine1").id] = target_pos
        # Target orientation
        random_z_angle = np.random.uniform(low=-np.pi, high=np.pi)  # Random angle in radians
        z_rotation = Rotation.from_euler('z', random_z_angle)
        new_rotation = z_rotation * self.initial_vine_rotation
        new_quat = new_rotation.as_quat()
        self.model.body_quat[self.model.body("vine1").id] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

        target_names = ["block1", "block1_big", "block1_small"]
        sub_geom_ids = {}
        for name in target_names:
            sub_body = self.model.body(name)
            geom_start = self.model.body_geomadr[sub_body.id]
            geom_count = self.model.body_geomnum[sub_body.id]
            sub_geom_ids[name] = list(range(geom_start, geom_start + geom_count))

        if object_cfg.get("random_size"):
            active_sub = np.random.choice(target_names)
        else:
            active_sub = "block1"
        for name in target_names:
            for geom_id in sub_geom_ids[name]:
                geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                if name == active_sub:
                    if geom_name == active_sub:
                        active_geom_name = geom_name
                        self.model.geom_group[geom_id] = 3
                        self.model.geom_contype[geom_id] = 1
                        self.model.geom_conaffinity[geom_id] = 1
                    else:
                        self.model.geom_group[geom_id] = 0
                        self.model.geom_contype[geom_id] = 0
                        self.model.geom_conaffinity[geom_id] = 0
                else:
                    self.model.geom_group[geom_id] = 3
                    self.model.geom_contype[geom_id] = 0
                    self.model.geom_conaffinity[geom_id] = 0

        distract_pos_noise_low = object_cfg.get("distract_pos_noise_low", [0.0, 0.0, 0.0])
        distract_pos_noise_high = object_cfg.get("distract_pos_noise_high", [0.0, 0.0, 0.0])

        distractor_indices = list(range(2, self.num_green + 2))
        if object_cfg.get("random_count"):
            active_count = np.random.randint(1, num_green + 1)
        else:
            active_count = num_green
        active_indices = np.random.choice(distractor_indices, size=active_count, replace=False)
        self.active_indices = active_indices

        for i in distractor_indices:
            # Randomize the distractor vine's position.
            distract_pos_noise = np.random.uniform(low=distract_pos_noise_low,
                                                    high=distract_pos_noise_high,
                                                    size=3)
            vine_body = self.model.body(f"vine{i}")
            self.model.body_pos[vine_body.id] = target_pos + distract_pos_noise

            # Randomize its orientation.
            random_z_angle = np.random.uniform(low=-np.pi, high=np.pi)
            z_rotation = Rotation.from_euler('z', random_z_angle)
            new_rotation = z_rotation * self.initial_vine_rotation
            new_quat = new_rotation.as_quat()
            self.model.body_quat[vine_body.id] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

            # change strawb size
            sub_names = [f"block{i}", f"block{i}_big", f"block{i}_small"]
            sub_geom_ids = {}
            # Gather geom id lists for each sub-body.
            for name in sub_names:
                sub_body = self.model.body(name)
                geom_start = self.model.body_geomadr[sub_body.id]
                geom_count = self.model.body_geomnum[sub_body.id]
                sub_geom_ids[name] = list(range(geom_start, geom_start + geom_count))

            # If this vine is NOT active, disable its collisions.
            if i not in active_indices:
                self.model.body_pos[vine_body.id] = np.array([-1.0, 0.0, 1.0])
                for name in sub_names:
                    for geom_id in sub_geom_ids[name]:
                        self.model.geom_group[geom_id] = 3
                        self.model.geom_contype[geom_id] = 0
                        self.model.geom_conaffinity[geom_id] = 0
            else:
                # Otherwise, ensure default collision settings are in place.
                if object_cfg.get("random_size"):
                    active_sub = np.random.choice(sub_names)
                else:
                    active_sub = f"block{i}"
                for name in sub_names:
                    for geom_id in sub_geom_ids[name]:
                        geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                        if name == active_sub:
                            if geom_name == name:
                                self.model.geom_group[geom_id] = 3
                                self.model.geom_contype[geom_id] = 1
                                self.model.geom_conaffinity[geom_id] = 1
                            else:
                                self.model.geom_group[geom_id] = 0
                                self.model.geom_contype[geom_id] = 0
                                self.model.geom_conaffinity[geom_id] = 0
                        else:
                            self.model.geom_group[geom_id] = 3
                            self.model.geom_contype[geom_id] = 0
                            self.model.geom_conaffinity[geom_id] = 0


        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def domain_randomization(self) -> None:
        dr = self.cfg.get("domain_randomization", {})
        if dr.get("lighting", {}).get("enabled", False):
            lighting_noise(self)
        if dr.get("action_scale", {}).get("enabled", False):
            action_scale_noise(self)
        if dr.get("initial_state", {}).get("enabled", False):
            initial_state_noise(self)
        if dr.get("cameras", {}).get("enabled", False):
            camera_noise(self)
        if dr.get("skybox", {}).get("enabled", False):
            skybox_noise(self)
        if dr.get("floor", {}).get("enabled", False):
            floor_noise(self)
        if dr.get("objects", {}).get("enabled", False):
            self.object_noise()
        self._viewer = MujocoRenderer(self.model, self.data)


    def reset_arm_and_gripper(self):
        self.data.qpos[self._panda_dof_ids] = self._PANDA_HOME
        self.data.qpos[7:9] = self._GRIPPER_HOME
        self.data.ctrl[self._gripper_ctrl_id] = self._GRIPPER_MAX
        self.gripper_vec = self.gripper_dict["open"]
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[0] = self.data.sensor("pinch_pos").data.copy()
        self.data.mocap_quat[0] = self.data.sensor("pinch_quat").data.copy()
        mujoco.mj_step(self.model, self.data)


    def reset_model(self):
        # Some random resets were getting mujoco Nan warnings that's why the loop
        attempt = 0
        while True:
            attempt += 1
            self.reset_arm_and_gripper()
            if self.randomize_domain:
                self.domain_randomization()

            self.data.qvel[:] = 0
            self.data.qacc[:] = 0
            self.data.qfrc_applied[:] = 0
            self.data.xfrc_applied[:] = 0
            mujoco.mj_forward(self.model, self.data)

            if not self.randomize_domain:
                self.data.mocap_pos[0] = self.initial_position
                self.data.mocap_quat[0] = np.roll(self.initial_orientation, 1)

            desired_pos = self.data.mocap_pos[0].copy()
            desired_quat = self.data.mocap_quat[0].copy()

            for _ in range(10*self._n_substeps):
                tau = opspace(
                    model=self.model,
                    data=self.data,
                    site_id=self._pinch_site_id,
                    dof_ids=self._panda_dof_ids,
                    pos=self.data.mocap_pos[0],
                    ori=self.data.mocap_quat[0],
                    joint=self._PANDA_HOME,
                    gravity_comp=True,
                )
                self.data.ctrl[self._panda_ctrl_ids] = tau
                mujoco.mj_step(self.model, self.data)
            
            self._block_init = self.data.sensor("block1_pos").data.copy()

            self.distractor_displacements = {}
            for i in self.active_indices:
                self.distractor_displacements[i] = self.data.sensor(f"block{i}_pos").data.copy()

            self.grasp = -1.0
            self.prev_grasp_time = 0.0
            self.prev_gripper_state = 0 # 0 for open, 1 for closed
            self.gripper_state = 0
            self.gripper_blocked = False
            self.last_gripper_pos = 2*self.data.qpos[8]/self._GRIPPER_HOME[0]

             # Get the current end-effector pose from sensors.
            current_pos = self.data.sensor("pinch_pos").data.copy()
            current_quat = self.data.sensor("pinch_quat").data.copy()

            # Check that sensor readings are finite.
            if (np.any(np.isnan(current_pos)) or np.any(np.isnan(current_quat)) or
                np.any(np.isinf(current_pos)) or np.any(np.isinf(current_quat))):
                continue

            # Compute the difference in position.
            pos_diff = np.linalg.norm(current_pos - desired_pos)
            # Compute orientation difference using the dot-product of unit quaternions.
            current_quat_norm = current_quat / np.linalg.norm(current_quat)
            desired_quat_norm = desired_quat / np.linalg.norm(desired_quat)
            dot = np.abs(np.dot(current_quat_norm, desired_quat_norm))
            dot = np.clip(dot, -1.0, 1.0)
            orient_diff = 2 * np.arccos(dot)

            pos_threshold = 0.1    
            orient_threshold = 0.2    

            if pos_diff < pos_threshold and orient_diff < orient_threshold:
                return self._get_obs()
            else:
                print(
                    f"Reset attempt {attempt+1}: pose error too high "
                    f"(pos_diff: {pos_diff:.4f}, orient_diff: {orient_diff:.4f}), retrying reset."
                )
                if attempt > 100:
                    raise RuntimeError("Failed to achieve valid reset after multiple attempts")


    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Scale actions (zyx because end effector frame z is along the gripper axis)
        if self.ee_dof == 3:
            z, y, x, grasp = action
        elif self.ee_dof == 4:
            z, y, x, yaw, grasp = action
            roll, pitch = 0, 0
            drot = np.array([roll, pitch, yaw]) * self.rot_scale
        elif self.ee_dof == 6:
            z, y, x, roll, pitch, yaw, grasp = action
            drot = np.array([roll, pitch, yaw]) * self.rot_scale
        dpos = np.array([x, y, z]) * self.pos_scale
        # Apply position change
        pos = self.data.sensor("pinch_pos").data
        current_quat = np.roll(self.data.sensor("pinch_quat").data, -1)
        current_rotation = Rotation.from_quat(current_quat)

        dpos_world = current_rotation.apply(dpos)
        npos = np.clip(pos + dpos_world, *self._CARTESIAN_BOUNDS)
        self.data.mocap_pos[0] = npos

        if self.ee_dof > 3:
            # Convert mujoco wxyz to scipy xyzw
            current_quat = np.roll(self.data.sensor("pinch_quat").data, -1)
            current_rotation = Rotation.from_quat(current_quat)
            # Convert the action rotation to a Rotation object
            action_rotation = Rotation.from_euler('xyz', drot)
            # Apply the action rotation
            new_rotation = action_rotation * current_rotation
            # Calculate the new relative rotation
            new_relative_rotation = self.initial_rotation.inv() * new_rotation
            # Convert to euler angles and clip
            relative_euler = new_relative_rotation.as_euler('xyz')
            clipped_euler = np.clip(relative_euler, self._ROTATION_BOUNDS[0], self._ROTATION_BOUNDS[1])
            # Convert back to rotation and apply to initial orientation
            clipped_rotation = Rotation.from_euler('xyz', clipped_euler)
            final_rotation = self.initial_rotation * clipped_rotation
            # Set the final orientation
            self.data.mocap_quat[0] = np.roll(final_rotation.as_quat(), 1)

        # Handle grasping
        moving_gripper = False
        if self.data.time - self.prev_grasp_time < self.gripper_sleep:
            self.gripper_blocked = True
            grasp = self.prev_grasp
        else:
            if grasp <= 0.5 and self.gripper_state == 0:
                self.gripper_vec = self.gripper_dict["open"]
                self.gripper_blocked = False
                moving_gripper=False
            elif grasp >= -0.5 and self.gripper_state == 1:
                self.gripper_vec = self.gripper_dict["closed"]
                self.gripper_blocked = False
                moving_gripper=False
            elif grasp < -0.5 and self.gripper_state == 1:
                self.data.ctrl[self._gripper_ctrl_id] = self._GRIPPER_MAX
                self.gripper_state = 0
                self.gripper_vec = self.gripper_dict["opening"]
                self.prev_grasp_time = self.data.time
                self.prev_grasp = grasp
                self.gripper_blocked=True
                moving_gripper=True
                target_sim_time = self.data.time + self.gripper_sleep
            elif grasp > 0.5 and self.gripper_state == 0:
                self.data.ctrl[self._gripper_ctrl_id] = self._GRIPPER_MIN
                self.gripper_state = 1
                self.gripper_vec = self.gripper_dict["closing"]
                self.prev_grasp_time = self.data.time
                self.prev_grasp = grasp
                self.gripper_blocked=True
                moving_gripper=True
                target_sim_time = self.data.time + self.gripper_sleep

        if self.gripper_pause and moving_gripper:
            while self.data.time < target_sim_time:
                tau = opspace(
                model=self.model,
                data=self.data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self.data.mocap_pos[0],
                ori=self.data.mocap_quat[0],
                joint=self._PANDA_HOME,
                gravity_comp=True,
            )
                self.data.ctrl[self._panda_ctrl_ids] = tau
                mujoco.mj_step(self.model, self.data)
        else:
            for i in range(self._n_substeps):
                if i < self._n_substeps/5:
                    continue
                else:
                    tau = opspace(
                        model=self.model,
                        data=self.data,
                        site_id=self._pinch_site_id,
                        dof_ids=self._panda_dof_ids,
                        pos=self.data.mocap_pos[0],
                        ori=self.data.mocap_quat[0],
                        joint=self._PANDA_HOME,
                        gravity_comp=True,
                    )
                    self.data.ctrl[self._panda_ctrl_ids] = tau
                    mujoco.mj_step(self.model, self.data)

        # Observation
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        # Reward
        reward, info = self._compute_reward(action)
        if info['success'] == True and self.reward_type == "sparse":
            reward = 1.0
            terminated = True
        else:
            terminated = False
        self.prev_gripper_state = self.gripper_state

        return obs, reward, terminated, False, info 
    
    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames
    
    def _get_vel(self):
        """
        Compute the Cartesian speed (linear and angular velocity) of the end-effector.
        
        Returns:
            cartesian_speed: A (6,) numpy array where the first 3 elements are the
                            linear velocities and the last 3 elements are the angular velocities.
        """
        dq = self.data.qvel[self._panda_dof_ids]
        J_v = np.zeros((3, self.model.nv), dtype=np.float64)
        J_w = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, J_v, J_w, self._pinch_site_id)
        J_v, J_w = J_v[:, self._panda_dof_ids], J_w[:, self._panda_dof_ids]
        J = np.vstack((J_v, J_w))
        dx = J @ dq
        return dx.astype(np.float32)

    def _get_obs(self):
        obs = {"state": {}}
        
        # Original position and orientation observations
        tcp_pose = np.concatenate([self.data.sensor("pinch_pos").data, 
                                  np.roll(self.data.sensor("pinch_quat").data, -1)])
        # Define noise parameters
        position_noise_std = self.cfg.get("ee_pos_noise", 0.01)  # e.g., 1 cm standard deviation
        orientation_noise_std = self.cfg.get("ee_ori_noise", 0.005)  # e.g., small rotations in quaternion
        # Add Gaussian noise to position and orientation
        # if self.randomize_domain:
        #     tcp_pose[:3] = tcp_pose[:3] + np.random.normal(0, position_noise_std, size=3)
        #     tcp_pose[3:] = tcp_pose[3:] + np.random.normal(0, orientation_noise_std, size=4)
        #     tcp_pose[3:] /= np.linalg.norm(tcp_pose[3:])
        
        # Populate noisy observations
        obs["state"]["tcp_pose"] = tcp_pose.astype(np.float32)
        obs["state"]["tcp_vel"] = self._get_vel()
        obs["state"]["gripper_pos"] = np.array([2*self.data.qpos[8]/self._GRIPPER_HOME[0]], dtype=np.float32)
        obs["state"]["gripper_vec"] = self.gripper_vec

        if not self.image_obs:
            obs["state"]["block_pos"] = self.data.sensor("block1_pos").data.astype(np.float32)
        if self.image_obs:
            obs["images"] = {}
            for cam_name in self.cameras:
                cam_id = self.model.camera(cam_name).id
                obs["images"][cam_name] = self._viewer.render(render_mode="rgb_array", camera_id=cam_id)

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs
        
    def _compute_reward(self, action):
        block1_pos = self.data.sensor("block1_pos").data
        tcp_pos = self.data.sensor("long_pinch_pos").data
        r_red =  -np.tanh(5 * np.linalg.norm(block1_pos - tcp_pos))

        red_distance = np.linalg.norm(block1_pos - self._block_init)
        green_distance = 0
        for i in self.distractor_displacements:
            green_distance += np.linalg.norm(self.distractor_displacements[i] - self.data.sensor(f"block{i}_pos").data)

        total_distance = red_distance + green_distance
        r_dist = -np.tanh(5*np.sum(total_distance))

        # Movement rewards
        r_energy = -np.tanh(5*np.linalg.norm(action[:-1]))
        r_smooth = -np.tanh(np.linalg.norm(action[:-1] - self.prev_action[:-1]) )
        self.prev_action = action
        # Gripper Penalty
        if (self.last_gripper_pos > 0.9 and action[-1] > 0.5) or (self.last_gripper_pos < 0.1 and action[-1] < -0.5):
            grasp_penalty = -1.0
        else:
            grasp_penalty = 0.0
        self.last_gripper_pos = 2*self.data.qpos[8]/self._GRIPPER_HOME[0]
        

        # Check if gripper pads are in contact with the object
        right_finger_contact_good = False
        left_finger_contact_good = False
        right_finger_contact_bad = False
        left_finger_contact_bad = False
        good_grasp = False
        bad_grasp = False
        r_green_col = 0

        # Check collisions
        for i in range(self.data.ncon):
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[i].geom1) or ""
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[i].geom2) or ""

            if ("finger" in geom1_name) or ("finger" in geom2_name):
                if ("block" in geom1_name) or (f"block" in geom2_name):
                    r_green_col = -1.0

            if geom1_name == "right_finger_inner" or geom2_name == "right_finger_inner":
                if geom1_name == "stem1" or geom2_name == "stem1":
                    right_finger_contact_good = True
                elif geom1_name == "aG3" or geom2_name == "aG3" or geom1_name == "left_finger_inner" or geom2_name == "left_finger_inner":
                    pass
                else:
                    right_finger_contact_good = False
                    right_finger_contact_bad = True
            if geom1_name =="left_finger_inner" or geom2_name =="left_finger_inner":
                if geom1_name == "stem1" or geom2_name == "stem1":
                    left_finger_contact_good = True
                elif geom1_name == "aG3" or geom2_name == "aG3" or geom1_name == "right_finger_inner" or geom2_name == "right_finger_inner":
                    pass
                else:
                    left_finger_contact_good = False
                    left_finger_contact_bad = True
            if right_finger_contact_good and left_finger_contact_good:
                good_grasp = True
            if right_finger_contact_bad and left_finger_contact_bad:
                bad_grasp = True

        if good_grasp and (np.linalg.norm(self.data.sensor("block1_pos").data - self._block_init) < 0.05):
            r_grasp = 1.0
            success = True
        else:
            r_grasp = 0.0
            success = False
        r_bad_grasp = -float(bad_grasp)
        
        info = {}
        if self.reward_type == "dense":
            rewards = {'r_grasp': r_grasp, 'r_red': r_red, 'r_green_col': r_green_col, 'r_dist': r_dist, 'r_bad_grasp': r_bad_grasp, 'r_energy': r_energy, 'r_smooth': r_smooth, 'grasp_penalty': grasp_penalty}
            reward_scales = {'r_grasp': 10.0, 'r_red': 4.0, 'r_green_col': 1.0, 'r_dist': 1.0, 'r_bad_grasp': 1.0, 'r_energy': 1.0, 'r_smooth': 1.0, 'grasp_penalty': 1.0}
            rewards = {k: v * reward_scales[k] for k, v in rewards.items()}
            reward = np.clip(sum(rewards.values()), -1e4, 1e4)
            info = rewards
        elif self.reward_type == "sparse":
            reward = float(success)

        info['success'] = success
        return reward, info
    
    # def _compute_reward(self, action):
    #     # 18th Feb reward function
    #     block1_pos = self.data.sensor("block1_pos").data
    #     tcp_pos = self.data.sensor("long_pinch_pos").data
    #     r_red = 1 - np.tanh(5 * np.linalg.norm(block1_pos - tcp_pos))

    #     green_distance = 0
    #     for i in self.distractor_displacements:
    #         green_distance += np.linalg.norm(self.distractor_displacements[i] - self.data.sensor(f"block{i}_pos").data)

    #     r_dist = 1 - np.tanh(5*np.sum(green_distance))

    #     # Movement rewards
    #     r_energy = -np.linalg.norm(action)
    #     r_smooth = -np.linalg.norm(action - self.prev_action) 
    #     self.prev_action = action
        
        

    #     # Check if gripper pads are in contact with the object
    #     right_finger_contact = False
    #     left_finger_contact = False
    #     success = False
    #     # Check collisions
    #     for i in range(self.data.ncon):
    #         geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[i].geom1) or ""
    #         geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[i].geom2) or ""

    #         if geom1_name == "right_finger_inner" or geom2_name == "right_finger_inner":
    #             if geom1_name == "stem1" or geom2_name == "stem1":
    #                 right_finger_contact = True
    #         if geom1_name =="left_finger_inner" or geom2_name =="left_finger_inner":
    #             if geom1_name == "stem1" or geom2_name == "stem1":
    #                 left_finger_contact = True
    #         if right_finger_contact and left_finger_contact:
    #             success=True
    #             break 
    #     r_grasp = float(success)
        
    #     info = {}
    #     if self.reward_type == "dense":
    #         rewards = {'r_grasp': r_grasp, 'r_red': r_red, 'r_dist': r_dist, 'r_energy': r_energy, 'r_smooth': r_smooth}
    #         reward_scales = {'r_grasp': 8.0, 'r_red': 4.0, 'r_dist': 1.0, 'r_energy': 2.0, 'r_smooth': 1.0}
    #         rewards = {k: v * reward_scales[k] for k, v in rewards.items()}
    #         reward = np.clip(sum(rewards.values()), -1e4, 1e4)
    #         info = rewards
    #     elif self.reward_type == "sparse":
    #         reward = float(success)

    #     info['success'] = success
    #     return reward, info