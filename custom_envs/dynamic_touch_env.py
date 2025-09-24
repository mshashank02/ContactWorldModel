# custom_envs/dynamic_touch_env.py
import numpy as np
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate import MujocoManipulateEnv  # or your exact parent
# ^ adjust import to your actual parent env (e.g., MujocoManipulateTouchSensorsEnv)

class DynamicXMLTouchEnv(MujocoManipulateEnv, EzPickle):
    def __init__(
        self,
        xml_path: str,
        target_position: str = "random",
        target_rotation: str = "xyz",
        ignore_z_target_rotation: bool = False,   # <-- NEW: accept the flag
        touch_get_obs: str = "sensordata",
        reward_type: str = "sparse",
        **kwargs,
    ):
        """
        Build a ShadowHand manipulation env directly from a generated MuJoCo XML.
        This forwards Gym's standard knobs including 'ignore_z_target_rotation'.
        """

        # forward everything to the parent – it already knows how to handle
        # ignore_z_target_rotation inside _goal_distance()
        super().__init__(
            model_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            ignore_z_target_rotation=ignore_z_target_rotation,  # <-- pass through
            touch_get_obs=touch_get_obs,
            reward_type=reward_type,
            # you probably already set this range; keep whatever you use now:
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            **kwargs,
        )

        EzPickle.__init__(
            self,
            xml_path,
            target_position,
            target_rotation,
            ignore_z_target_rotation,    # <-- include in pickle state
            touch_get_obs,
            reward_type,
            **kwargs,
        )
