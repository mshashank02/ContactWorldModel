"""Episode-sharded compressed NPZ trajectory recorder."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from world_model.events import derive_event_tags
from world_model.schemas import RunMetadata, episode_array_key


def _copy_structured(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            {inner_key: np.array(inner_value, copy=True) for inner_key, inner_value in item.items()}
            if isinstance(item, dict)
            else np.array(item, copy=True)
        )
        for key, item in value.items()
    }


class TrajectoryRecorder(gym.Wrapper):
    """Record complete Gymnasium episodes as independent compressed NPZ shards."""

    def __init__(
        self,
        env: gym.Env,
        run_dir: str | Path,
        run_metadata: RunMetadata,
        compressed: bool = True,
        event_oversample_factor: float = 2.0,
    ):
        super().__init__(env)
        self.run_dir = Path(run_dir)
        self.episodes_dir = self.run_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.run_metadata = run_metadata
        self.compressed = compressed
        self.event_oversample_factor = float(event_oversample_factor)
        existing = sorted(self.episodes_dir.glob("episode_*.npz"))
        self._next_episode_id = len(existing)
        self._current_observation: dict[str, Any] | None = None
        self._steps: list[dict[str, Any]] = []

    def _structured(self) -> dict[str, Any]:
        return _copy_structured(self.env.get_structured_observation())

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self._steps = []
        self._current_observation = self._structured()
        return observation, info

    def step(self, action: np.ndarray):
        if self._current_observation is None:
            raise RuntimeError("TrajectoryRecorder.step() called before reset()")
        observation, reward, terminated, truncated, info = self.env.step(action)
        next_observation = self._structured()
        success_value = info.get("is_success", info.get("success", np.nan))
        self._steps.append(
            {
                "observation": self._current_observation,
                "next_observation": next_observation,
                "action": np.asarray(action, dtype=np.float32).copy(),
                "reward": np.float32(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": np.float32(success_value),
            }
        )
        self._current_observation = next_observation
        if terminated or truncated:
            self._write_episode()
            self._current_observation = None
        return observation, reward, terminated, truncated, info

    def _stack_observation(
        self, steps: list[dict[str, Any]], source_key: str, output_prefix: str
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        simple_fields = (
            "proprioception",
            "policy_touch",
            "achieved_goal",
            "desired_goal",
            "sensor_mask",
        )
        for field in simple_fields:
            logical = f"{output_prefix}/{field}"
            result[episode_array_key(logical)] = np.stack(
                [step[source_key][field] for step in steps]
            )
        oracle_fields = steps[0][source_key]["oracle_touch"]
        for field in oracle_fields:
            logical = f"{output_prefix}/oracle_touch/{field}"
            result[episode_array_key(logical)] = np.stack(
                [step[source_key]["oracle_touch"][field] for step in steps]
            )
        return result

    def _write_episode(self) -> None:
        if not self._steps:
            return
        arrays = self._stack_observation(self._steps, "observation", "observations")
        arrays.update(
            self._stack_observation(self._steps, "next_observation", "next_observations")
        )
        arrays[episode_array_key("actions")] = np.stack([step["action"] for step in self._steps])
        arrays[episode_array_key("rewards")] = np.asarray(
            [step["reward"] for step in self._steps], dtype=np.float32
        )
        arrays[episode_array_key("terminated")] = np.asarray(
            [step["terminated"] for step in self._steps], dtype=bool
        )
        arrays[episode_array_key("truncated")] = np.asarray(
            [step["truncated"] for step in self._steps], dtype=bool
        )
        arrays[episode_array_key("success")] = np.asarray(
            [step["success"] for step in self._steps], dtype=np.float32
        )

        events = derive_event_tags(
            arrays[episode_array_key("next_observations/oracle_touch/contact_active")],
            arrays[episode_array_key("next_observations/oracle_touch/tangential_force")],
            arrays[episode_array_key("success")],
            arrays[episode_array_key("next_observations/oracle_touch/normal_force")],
        )
        for name, values in events.items():
            arrays[episode_array_key(f"events/{name}")] = values
        event_score = int(sum(int(values.sum()) for values in events.values()))
        sampling_weight = 1.0 + self.event_oversample_factor * min(event_score, 10)

        episode_id = self._next_episode_id
        metadata = {
            **self.run_metadata.to_dict(),
            "episode_id": episode_id,
            "length": len(self._steps),
            "return": float(arrays[episode_array_key("rewards")].sum()),
            "event_score": event_score,
            "sampling_weight": sampling_weight,
            "has_contact": bool(
                arrays[
                    episode_array_key("next_observations/oracle_touch/contact_active")
                ].any()
            ),
        }
        for key, value in metadata.items():
            if value is None or isinstance(value, dict):
                continue
            arrays[f"metadata__{key}"] = np.asarray(value)

        target = self.episodes_dir / f"episode_{episode_id:06d}.npz"
        temporary = target.with_suffix(".npz.tmp")
        writer = np.savez_compressed if self.compressed else np.savez
        with temporary.open("wb") as handle:
            writer(handle, **arrays)
        os.replace(temporary, target)
        self._next_episode_id += 1
        self._steps = []

    @property
    def episodes_written(self) -> int:
        return self._next_episode_id

