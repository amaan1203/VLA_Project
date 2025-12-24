import os
from panda_mujoco_gym.envs.panda_env import FrankaEnv

import mujoco
import mujoco.viewer
import numpy as np
import cv2 






MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "pick_and_place.xml")

m = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, height=480, width=640)


class FrankaPickAndPlaceEnv(FrankaEnv):
    def __init__(
        self,
        reward_type,
        **kwargs,
    ):
        super().__init__(
            model_path=MODEL_XML_PATH,
            n_substeps=25,
            reward_type=reward_type,
            block_gripper=False,
            distance_threshold=0.05,
            goal_xy_range=0.3,
            obj_xy_range=0.3,
            goal_x_offset=0.0,
            goal_z_range=0.2,
            **kwargs,
        )
