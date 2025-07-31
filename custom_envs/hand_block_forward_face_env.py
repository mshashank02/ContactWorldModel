import os 
import numpy as np 
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.shadow_dexterous_hand import MujocoManipulateTouchSensorsEnv
from gymnasium_robotics.utils import rotations
#Path to the Mujcoco XML file 
MANIPULATE_BLOCK_XML = os.path.join("hand", "manipulate_block_touch_sensors.xml")


class MujocoHandBlockForwardFaceTouchEnv(MujocoManipulateTouchSensorsEnv, EzPickle):
    """

    ## Description 
    # This environment overrides the compute_reward function to give a reward when any one of the face of
    cubes is facing the front of the environment which is defined as the positive X axis in world coordinated.

    # The reward is sparse and is -1 when none of the cubes faces are aligned with the positive X axis in world coordinates
    and is 0 when any face of the cube is aligned with the positive X axis in world coordinated.

    ##Observation Space 

    We change the observation space to only contain the proprioceptive inputs of the robot i.e. the joint angles
    and velocities and remove the states of the cube. Touch sensor data from the 92 sensors is also included.

    """

    def __init__(
            self,
            target_position="random",
            target_rotation="xyz",
            touch_get_obs="sensordata",
            reward_type="sparse",
            **kwargs
    ):
        super().__init__(
            model_path=MANIPULATE_BLOCK_XML,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            **kwargs
        )
        EzPickle.__init__(
            self, target_position, target_rotation, touch_get_obs, reward_type, **kwargs
        )

    def _is_forward_facing(self, quat):
        R = rotations.quat2mat(quat)
        axes = R.T
        forward = np.array([1.0, 0.0, 0.0])
        return any(abs(np.dot(axis, forward)) > 0.95 for axis in axes)
    
    def _is_success(self, achieved_goal, desired_goal):
        quat = achieved_goal[..., 3:]  # Always extracts quaternion correctly

        # Handle single or batch cases
        if quat.ndim == 1:
            is_forward = self._is_forward_facing(quat)
        elif quat.ndim == 2:
            is_forward = np.array([self._is_forward_facing(q) for q in quat])
        else:
            raise ValueError(f"Unexpected quaternion shape: {quat.shape}")

        return np.array(is_forward, dtype=np.float32)


    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Sparse reward: 0 if success, -1 otherwise.
        """
        success = self._is_success(achieved_goal, desired_goal)
        return success - 1.0  # 0 if success, -1 if not

