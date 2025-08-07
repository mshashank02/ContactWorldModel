import os
import numpy as np
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.shadow_dexterous_hand import MujocoManipulateTouchSensorsEnv
from gymnasium_robotics.utils import rotations

# Path to the MuJoCo XML file
MANIPULATE_BLOCK_XML = os.path.join("hand", "manipulate_block_touch_sensors.xml")


def _get_yaw_from_quat(quat):
    """
    Extract yaw (Z-axis rotation) from quaternion.
    Returns angle in [-π, π].
    """
    R = rotations.quat2mat(quat)
    return np.arctan2(R[1, 0], R[0, 0])


class MujocoHandBlockYawTouchEnv(MujocoManipulateTouchSensorsEnv, EzPickle):
    """
    This environment rewards the agent only when the cube's Z-axis rotation (yaw)
    matches the target yaw, ignoring other rotational or positional errors.

    The observation includes robot proprioception + 92 touch sensors.
    """

    def __init__(
        self,
        target_position="random",
        target_rotation="z",  # random yaw
        touch_get_obs="sensordata",
        reward_type="sparse",
        rotation_threshold=np.deg2rad(5),  # 5 degrees
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
            self, target_position, target_rotation, touch_get_obs,
            reward_type, rotation_threshold, **kwargs
        )
        self.rotation_threshold = rotation_threshold

    def _yaw_error(self, quat_a, quat_b):
        """
        Returns the absolute yaw difference between two quaternions.
        """
        yaw_a = _get_yaw_from_quat(quat_a)
        yaw_b = _get_yaw_from_quat(quat_b)
        return np.abs(((yaw_a - yaw_b + np.pi) % (2 * np.pi)) - np.pi)

    def _is_success(self, achieved_goal, desired_goal):
        quat_a = achieved_goal[..., 3:]
        quat_b = desired_goal[..., 3:]

        if quat_a.ndim == 1:
            error = self._yaw_error(quat_a, quat_b)
            return np.array(error < self.rotation_threshold, dtype=np.float32)

        elif quat_a.ndim == 2:
            errors = np.array([self._yaw_error(qa, qb) for qa, qb in zip(quat_a, quat_b)])
            return (errors < self.rotation_threshold).astype(np.float32)

        else:
            raise ValueError(f"Unexpected quaternion shape: {quat_a.shape}")

    def compute_reward(self, achieved_goal, desired_goal, info):
        success = self._is_success(achieved_goal, desired_goal)
        return success - 1.0  # 0 if success, -1 otherwise
