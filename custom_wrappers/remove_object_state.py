import gymnasium as gym
import numpy as np
from gymnasium import spaces

class RemoveObjectStateWrapper(gym.ObservationWrapper):
    """
    Removes block-related info (indices 48–60) from the observation vector.
    Keeps robot joint info (0–47) and touch sensor data (61–152).
    Assumes observation['observation'] is of shape (153,).
    """

    def __init__(self, env):
        super().__init__(env)

        # Indices to keep: robot state [0–47], touch data [61–152]
        self.kept_indices = list(range(48)) + list(range(61, 153))

        # Modify observation space accordingly
        original_obs_space = self.observation_space.spaces["observation"]
        low = original_obs_space.low[self.kept_indices]
        high = original_obs_space.high[self.kept_indices]
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=low, high=high, dtype=original_obs_space.dtype),
            "achieved_goal": self.observation_space.spaces["achieved_goal"],
            "desired_goal": self.observation_space.spaces["desired_goal"],
        })

    def observation(self, obs):
        obs["observation"] = obs["observation"][self.kept_indices]
        return obs
