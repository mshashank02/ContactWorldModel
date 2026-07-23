"""Passive dense tactile supervision from MuJoCo's contact-query API."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np

from world_model.sensor_manifest import SensorSite


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class OracleTouchWrapper(gym.Wrapper):
    """Measure dense touch without adding collision geoms or policy inputs.

    Candidate sites come from a canonical XML manifest and need not exist in the
    active MuJoCo model. Their local volumes are transformed with the matching
    hand body pose, then real MuJoCo hand-object contacts are assigned only when
    the contact point lies inside a site volume. No forces or positions are
    synthesized when the simulator does not report a matching contact.
    """

    def __init__(
        self,
        env: gym.Env,
        sensors: list[SensorSite],
        contact_margin: float = 1.0e-3,
        object_body_name: str = "object",
    ):
        super().__init__(env)
        self.sensors = sensors
        self.contact_margin = float(contact_margin)
        self.object_body_name = object_body_name
        self.latest_oracle_touch: dict[str, np.ndarray] = {}
        self._model = self.unwrapped.model
        self._data = self.unwrapped.data
        self._mujoco = getattr(self.unwrapped, "_mujoco", None)
        if self._mujoco is None:
            try:
                import mujoco
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise ImportError("OracleTouchWrapper requires the mujoco Python package") from exc
            self._mujoco = mujoco
        self._sensor_body_ids = np.array(
            [self._name_to_id("body", row.body_name) for row in sensors], dtype=np.int32
        )
        missing_bodies = [
            sensors[idx].body_name for idx in np.flatnonzero(self._sensor_body_ids < 0)
        ]
        if missing_bodies:
            unique = ", ".join(sorted(set(missing_bodies))[:5])
            raise ValueError(f"Oracle sensor bodies are absent from active model: {unique}")
        self._by_body: dict[int, list[int]] = defaultdict(list)
        for idx, body_id in enumerate(self._sensor_body_ids):
            self._by_body[int(body_id)].append(idx)
        self._hand_body_ids = self._descendant_body_ids("robot0:")
        self._object_body_ids = self._descendant_body_ids(object_body_name)

    def _name_to_id(self, kind: str, name: str) -> int:
        enum_name = {
            "body": "mjOBJ_BODY",
            "geom": "mjOBJ_GEOM",
        }[kind]
        enum_value = getattr(self._mujoco.mjtObj, enum_name)
        return int(self._mujoco.mj_name2id(self._model, enum_value, name))

    def _id_to_name(self, kind: str, item_id: int) -> str:
        if item_id < 0:
            return ""
        enum_name = {"body": "mjOBJ_BODY", "geom": "mjOBJ_GEOM"}[kind]
        enum_value = getattr(self._mujoco.mjtObj, enum_name)
        return self._mujoco.mj_id2name(self._model, enum_value, int(item_id)) or ""

    def _descendant_body_ids(self, name_token: str) -> set[int]:
        direct = {
            body_id
            for body_id in range(int(self._model.nbody))
            if (
                self._id_to_name("body", body_id).startswith(name_token)
                if name_token.endswith(":")
                else self._id_to_name("body", body_id) == name_token
            )
        }
        if not direct:
            return set()
        result = set(direct)
        changed = True
        while changed:
            changed = False
            for body_id in range(1, int(self._model.nbody)):
                if int(self._model.body_parentid[body_id]) in result and body_id not in result:
                    result.add(body_id)
                    changed = True
        return result

    def _geom_body(self, geom_id: int) -> int:
        return int(self._model.geom_bodyid[geom_id]) if geom_id >= 0 else -1

    def _hand_contact(self, geom1: int, geom2: int) -> tuple[int, int, bool] | None:
        body1, body2 = self._geom_body(geom1), self._geom_body(geom2)
        # A negative geom id is how MuJoCo represents a flex contact. Generated
        # tasks have one object flex; the hand side is still an ordinary geom.
        if body1 in self._hand_body_ids and (body2 in self._object_body_ids or geom2 < 0):
            return body1, geom2, True
        if body2 in self._hand_body_ids and (body1 in self._object_body_ids or geom1 < 0):
            return body2, geom1, False
        return None

    def _world_site_pose(self, sensor_idx: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.sensors[sensor_idx]
        body_id = int(self._sensor_body_ids[sensor_idx])
        body_rotation = np.asarray(self._data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        body_position = np.asarray(self._data.xpos[body_id], dtype=np.float64)
        site_rotation = body_rotation @ _quat_matrix(np.asarray(row.site_quat_wxyz))
        site_position = body_position + body_rotation @ np.asarray(row.local_position_xyz)
        return site_position, site_rotation

    def _matching_sites(self, body_id: int, contact_position: np.ndarray) -> list[int]:
        matched: list[int] = []
        for sensor_idx in self._by_body.get(body_id, []):
            center, rotation = self._world_site_pose(sensor_idx)
            local_delta = rotation.T @ (contact_position - center)
            halfsize = np.asarray(self.sensors[sensor_idx].site_size_xyz, dtype=np.float64)
            if np.all(np.abs(local_delta) <= halfsize + self.contact_margin):
                matched.append(sensor_idx)
        return matched

    def query_oracle_touch(self) -> dict[str, np.ndarray]:
        count = len(self.sensors)
        active = np.zeros(count, dtype=bool)
        normal_force = np.zeros(count, dtype=np.float32)
        tangential_force = np.zeros((count, 2), dtype=np.float32)
        position_sum = np.zeros((count, 3), dtype=np.float64)
        normal_sum = np.zeros((count, 3), dtype=np.float64)
        contact_count = np.zeros(count, dtype=np.int16)
        geom_id = np.full(count, -1, dtype=np.int32)
        site_world_position = np.empty((count, 3), dtype=np.float32)
        for sensor_idx in range(count):
            site_world_position[sensor_idx] = self._world_site_pose(sensor_idx)[0]

        for contact_idx in range(int(self._data.ncon)):
            contact = self._data.contact[contact_idx]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            hand_contact = self._hand_contact(geom1, geom2)
            if hand_contact is None:
                continue
            hand_body, other_geom, hand_is_geom1 = hand_contact
            contact_position = np.asarray(contact.pos, dtype=np.float64)
            matched = self._matching_sites(hand_body, contact_position)
            if not matched:
                continue

            wrench = np.zeros(6, dtype=np.float64)
            self._mujoco.mj_contactForce(self._model, self._data, contact_idx, wrench)
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            world_normal = frame[0] if hand_is_geom1 else -frame[0]
            for sensor_idx in matched:
                active[sensor_idx] = True
                normal_force[sensor_idx] += np.float32(abs(wrench[0]))
                tangential_force[sensor_idx] += wrench[1:3].astype(np.float32)
                position_sum[sensor_idx] += contact_position
                normal_sum[sensor_idx] += world_normal
                contact_count[sensor_idx] += 1
                geom_id[sensor_idx] = other_geom

        contact_position = np.full((count, 3), np.nan, dtype=np.float32)
        contact_normal = np.full((count, 3), np.nan, dtype=np.float32)
        for sensor_idx in np.flatnonzero(active):
            divisor = max(int(contact_count[sensor_idx]), 1)
            contact_position[sensor_idx] = position_sum[sensor_idx] / divisor
            norm = np.linalg.norm(normal_sum[sensor_idx])
            if norm > 0:
                contact_normal[sensor_idx] = normal_sum[sensor_idx] / norm
        return {
            "contact_active": active,
            "normal_force": normal_force,
            "tangential_force": tangential_force,
            "contact_position": contact_position,
            "contact_normal": contact_normal,
            "contacting_geom_id": geom_id,
            "contact_count": contact_count,
            "site_world_position": site_world_position,
        }

    def _refresh(self) -> None:
        self.latest_oracle_touch = self.query_oracle_touch()
        for idx, row in enumerate(self.sensors):
            if row.global_rest_position_xyz is None:
                row.global_rest_position_xyz = self.latest_oracle_touch["site_world_position"][idx].tolist()

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self._refresh()
        return observation, info

    def step(self, action: np.ndarray):
        result = self.env.step(action)
        self._refresh()
        return result
