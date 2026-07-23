"""Named observation access without hard-coded flat-vector indices."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from world_model.sensor_manifest import SensorSite


def _find_wrapper(env: gym.Env, attribute: str) -> Any:
    current: Any = env
    while current is not None:
        if hasattr(current, attribute):
            return current
        current = getattr(current, "env", None)
    raise AttributeError(f"No wrapper exposes {attribute!r}")


class StructuredObservationWrapper(gym.Wrapper):
    """Expose named policy, oracle, goal, and proprioceptive observations.

    With ``preserve_original=True`` (the collection default), reset/step return
    the exact policy observation unchanged while ``latest_structured_observation``
    provides recorder-only privileged fields. Set it to false for consumers that
    explicitly want the named dict as the environment observation.
    """

    def __init__(
        self,
        env: gym.Env,
        sensors: list[SensorSite],
        preserve_original: bool = True,
    ):
        super().__init__(env)
        self.sensors = sensors
        self.preserve_original = preserve_original
        self.latest_structured_observation: dict[str, Any] = {}
        self._oracle_wrapper = _find_wrapper(env, "latest_oracle_touch")
        active_rows = sorted(
            (row for row in sensors if row.active_in_layout and row.touch_sensor_name),
            key=lambda row: (
                row.policy_touch_index is None,
                row.policy_touch_index if row.policy_touch_index is not None else row.sensor_id,
            ),
        )
        self._active_sensor_names = [row.touch_sensor_name for row in active_rows]
        self._sensor_mask = np.asarray([row.active_in_layout for row in sensors], dtype=bool)
        self._model = self.unwrapped.model
        self._data = self.unwrapped.data
        self._mujoco = getattr(self.unwrapped, "_mujoco", None)
        if self._mujoco is None:
            import mujoco

            self._mujoco = mujoco
        sensor_pairs = [(name, self._sensor_id(name)) for name in self._active_sensor_names]
        missing = sorted(name for name, sensor_id in sensor_pairs if sensor_id < 0)
        if missing:
            raise ValueError(f"Active policy touch sensors are absent from the model: {missing[:5]}")
        # This is the exact list used by MujocoManipulateTouchSensorsEnv._get_obs;
        # it deliberately excludes unrelated legacy sensors present in the model.
        self._policy_sensor_ids = [int(value) for value in self.unwrapped._touch_sensor_id]
        expected_ids = sorted(sensor_id for _, sensor_id in sensor_pairs)
        if sorted(self._policy_sensor_ids) != expected_ids:
            raise ValueError(
                "Manifest active sensors do not match the environment policy-touch sensors"
            )
        if not preserve_original:
            self.observation_space = self._make_observation_space()

    def _sensor_id(self, name: str) -> int:
        return int(
            self._mujoco.mj_name2id(
                self._model, self._mujoco.mjtObj.mjOBJ_SENSOR, name
            )
        )

    def _read_policy_touch(self) -> np.ndarray:
        values: list[np.ndarray] = []
        for sensor_id in self._policy_sensor_ids:
            address = int(self._model.sensor_adr[sensor_id])
            dimension = int(self._model.sensor_dim[sensor_id])
            values.append(np.asarray(self._data.sensordata[address : address + dimension]))
        touch = np.concatenate(values).astype(np.float32) if values else np.empty(0, np.float32)
        touch_mode = getattr(self.unwrapped, "touch_get_obs", "sensordata")
        if touch_mode == "boolean":
            touch = (touch > 0).astype(np.float32)
        elif touch_mode == "log":
            touch = np.log(touch + 1.0).astype(np.float32)
        elif touch_mode not in {"sensordata", "boolean", "log"}:
            touch = np.empty(0, dtype=np.float32)
        return touch

    def _read_proprioception(self) -> np.ndarray:
        qpos, qvel = self.unwrapped._utils.robot_get_obs(
            self._model,
            self._data,
            self.unwrapped._model_names.joint_names,
        )
        return np.concatenate((qpos, qvel)).astype(np.float32, copy=False)

    def _make_observation_space(self) -> spaces.Dict:
        proprio = self._read_proprioception()
        policy_touch = self._read_policy_touch()
        oracle_count = len(self.sensors)
        goal_space = getattr(self.env.observation_space, "spaces", {})
        achieved_space = goal_space.get(
            "achieved_goal", spaces.Box(-np.inf, np.inf, shape=(0,), dtype=np.float32)
        )
        desired_space = goal_space.get(
            "desired_goal", spaces.Box(-np.inf, np.inf, shape=(0,), dtype=np.float32)
        )
        oracle_space = spaces.Dict(
            {
                "contact_active": spaces.MultiBinary(oracle_count),
                "normal_force": spaces.Box(0, np.inf, (oracle_count,), np.float32),
                "tangential_force": spaces.Box(-np.inf, np.inf, (oracle_count, 2), np.float32),
                "contact_position": spaces.Box(-np.inf, np.inf, (oracle_count, 3), np.float32),
                "contact_normal": spaces.Box(-np.inf, np.inf, (oracle_count, 3), np.float32),
                "contacting_geom_id": spaces.Box(-1, np.iinfo(np.int32).max, (oracle_count,), np.int32),
                "contact_count": spaces.Box(0, np.iinfo(np.int16).max, (oracle_count,), np.int16),
                "site_world_position": spaces.Box(-np.inf, np.inf, (oracle_count, 3), np.float32),
            }
        )
        return spaces.Dict(
            {
                "proprioception": spaces.Box(-np.inf, np.inf, proprio.shape, np.float32),
                "policy_touch": spaces.Box(-np.inf, np.inf, policy_touch.shape, np.float32),
                "oracle_touch": oracle_space,
                "achieved_goal": achieved_space,
                "desired_goal": desired_space,
                "sensor_mask": spaces.MultiBinary(oracle_count),
            }
        )

    def structure(self, observation: Any) -> dict[str, Any]:
        if not isinstance(observation, dict):
            raise TypeError("Shadow Hand collection expects a goal-conditioned dict observation")
        structured = {
            "proprioception": self._read_proprioception(),
            "policy_touch": self._read_policy_touch(),
            "oracle_touch": {
                key: np.array(value, copy=True)
                for key, value in self._oracle_wrapper.latest_oracle_touch.items()
            },
            "achieved_goal": np.asarray(observation["achieved_goal"], dtype=np.float32).copy(),
            "desired_goal": np.asarray(observation["desired_goal"], dtype=np.float32).copy(),
            "sensor_mask": self._sensor_mask.copy(),
        }
        self.latest_structured_observation = structured
        return structured

    def get_structured_observation(self) -> dict[str, Any]:
        return deepcopy(self.latest_structured_observation)

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        structured = self.structure(observation)
        return (observation if self.preserve_original else structured), info

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        structured = self.structure(observation)
        return (
            observation if self.preserve_original else structured,
            reward,
            terminated,
            truncated,
            info,
        )
