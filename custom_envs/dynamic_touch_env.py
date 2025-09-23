import numpy as np
import os 
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.shadow_dexterous_hand import MujocoManipulateTouchSensorsEnv

class DynamicXMLTouchEnv(MujocoManipulateTouchSensorsEnv, EzPickle):
    def __init__(self, 
                 xml_path,
                 target_position = "random",
                 target_rotation = "xyz",
                 ignore_z_target_rotation=False,
                 touch_get_obs = "sensordata",
                 reward_type = "sparse", 
                 **kwargs,
    ): 
        MujocoManipulateTouchSensorsEnv.__init__(
            self,
            model_path = xml_path,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            ignore_z_target_rotation=ignore_z_target_rotation,
            **kwargs,
        )
        EzPickle.__init__(
            self,
            xml_path,
            target_position,
            target_rotation,
            ignore_z_target_rotation,
            touch_get_obs,
            reward_type,
            **kwargs
        )


