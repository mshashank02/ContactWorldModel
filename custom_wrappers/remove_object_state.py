import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RemoveObjectStateWrapper(gym.ObservationWrapper):
    """Remove object state while retaining named robot and policy-touch fields.

    Field sizes are discovered from the underlying Shadow Hand environment;
    no observation indices or sensor counts are hard-coded.
    """

    def __init__(self, env):
        super().__init__(env)
        base = self.unwrapped
        robot_qpos, robot_qvel = base._utils.robot_get_obs(
            base.model, base.data, base._model_names.joint_names
        )
        self.proprioception_size = int(robot_qpos.size + robot_qvel.size)
        self.policy_touch_size = int(len(getattr(base, "_touch_sensor_id", ())))
        original_obs_space = self.observation_space.spaces["observation"]
        base_observation_size = int(base.observation_space.spaces["observation"].shape[0])
        head = np.arange(self.proprioception_size)
        tail = (
            np.arange(
                base_observation_size - self.policy_touch_size,
                base_observation_size,
            )
            if self.policy_touch_size
            else np.empty(0, dtype=int)
        )
        self.kept_indices = np.concatenate((head, tail))
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=original_obs_space.low[self.kept_indices],
                    high=original_obs_space.high[self.kept_indices],
                    dtype=original_obs_space.dtype,
                ),
                "achieved_goal": self.observation_space.spaces["achieved_goal"],
                "desired_goal": self.observation_space.spaces["desired_goal"],
            }
        )

    def observation(self, obs):
        result = dict(obs)
        result["observation"] = np.asarray(obs["observation"])[self.kept_indices]
        return result
