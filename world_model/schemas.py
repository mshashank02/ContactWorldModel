"""Canonical schemas shared by collection, indexing, and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DATASET_FORMAT_VERSION = "1.0"

ORACLE_TOUCH_FIELDS: dict[str, str] = {
    "contact_active": "bool",
    "normal_force": "float32",
    "tangential_force": "float32",
    "contact_position": "float32",
    "contact_normal": "float32",
    "contacting_geom_id": "int32",
    "contact_count": "int16",
    "site_world_position": "float32",
}

REQUIRED_TIMESTEP_ARRAYS = (
    "actions",
    "rewards",
    "terminated",
    "truncated",
    "success",
    "observations/proprioception",
    "observations/policy_touch",
    "observations/achieved_goal",
    "observations/desired_goal",
    "observations/sensor_mask",
    "next_observations/proprioception",
    "next_observations/policy_touch",
    "next_observations/achieved_goal",
    "next_observations/desired_goal",
    "next_observations/sensor_mask",
) + tuple(
    f"{prefix}/oracle_touch/{field}"
    for prefix in ("observations", "next_observations")
    for field in ORACLE_TOUCH_FIELDS
)


@dataclass(frozen=True)
class RunMetadata:
    """Metadata that is constant for every episode in a collection run."""

    run_id: str
    candidate_id: str
    object_id: str
    physics_mode: str
    seed: int
    alpha: float | None = None
    beta: float | None = None
    Rppx: float | None = None
    Rpt: float | None = None
    sensor_count_total: int = 0
    sensor_count_palm: int = 0
    sensor_count_tip: int = 0
    sensor_count_non_tip: int = 0
    oracle_sensor_count: int = 0
    policy_type: str = "random"
    policy_stage: str = "random"
    policy_checkpoint_step: int | None = None
    policy_checkpoint_path: str | None = None
    sensor_manifest_path: str = "sensor_manifest.json"
    object_manifest_path: str = "object_manifest.json"
    format_version: str = DATASET_FORMAT_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_RUN_METADATA = (
    "run_id",
    "candidate_id",
    "object_id",
    "physics_mode",
    "seed",
    "sensor_count_total",
    "oracle_sensor_count",
    "policy_type",
    "policy_stage",
    "sensor_manifest_path",
    "object_manifest_path",
    "format_version",
)


def episode_array_key(path: str) -> str:
    """Map a logical slash-separated field to its NPZ storage key."""

    return path.replace("/", "__")


def logical_array_key(key: str) -> str:
    """Map an NPZ storage key back to its logical slash-separated field."""

    return key.replace("__", "/")
