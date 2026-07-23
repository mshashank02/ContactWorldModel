"""Shape, dtype, metadata, and coverage validation for world-model datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_index import discover_episode_shards
from .schemas import (
    ORACLE_TOUCH_FIELDS,
    REQUIRED_RUN_METADATA,
    REQUIRED_TIMESTEP_ARRAYS,
    episode_array_key,
)


@dataclass
class ValidationIssue:
    severity: str
    shard: str
    message: str


@dataclass
class ValidationReport:
    dataset_root: str
    episode_count: int = 0
    timestep_count: int = 0
    total_bytes: int = 0
    empty_contact_episode_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    coverage: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def ready(self) -> bool:
        return self.episode_count > 0 and self.error_count == 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            error_count=self.error_count,
            warning_count=self.warning_count,
            ready=self.ready,
        )
        return result


_EXPECTED_DTYPES = {
    "actions": np.float32,
    "rewards": np.float32,
    "terminated": np.bool_,
    "truncated": np.bool_,
    "success": np.float32,
    "observations/oracle_touch/contact_active": np.bool_,
    "next_observations/oracle_touch/contact_active": np.bool_,
    "observations/oracle_touch/normal_force": np.float32,
    "next_observations/oracle_touch/normal_force": np.float32,
    "observations/oracle_touch/tangential_force": np.float32,
    "next_observations/oracle_touch/tangential_force": np.float32,
    "observations/oracle_touch/contact_position": np.float32,
    "next_observations/oracle_touch/contact_position": np.float32,
    "observations/oracle_touch/contact_normal": np.float32,
    "next_observations/oracle_touch/contact_normal": np.float32,
    "observations/oracle_touch/contacting_geom_id": np.int32,
    "next_observations/oracle_touch/contacting_geom_id": np.int32,
    "observations/oracle_touch/contact_count": np.int16,
    "next_observations/oracle_touch/contact_count": np.int16,
}
for _prefix in ("observations", "next_observations"):
    for _field in (
        "proprioception",
        "policy_touch",
        "achieved_goal",
        "desired_goal",
    ):
        _EXPECTED_DTYPES[f"{_prefix}/{_field}"] = np.float32
    _EXPECTED_DTYPES[f"{_prefix}/sensor_mask"] = np.bool_
    for _field, _dtype in ORACLE_TOUCH_FIELDS.items():
        _EXPECTED_DTYPES[f"{_prefix}/oracle_touch/{_field}"] = np.dtype(_dtype)

_ORACLE_SUFFIX_SHAPES = {
    "contact_active": (),
    "normal_force": (),
    "tangential_force": (2,),
    "contact_position": (3,),
    "contact_normal": (3,),
    "contacting_geom_id": (),
    "contact_count": (),
    "site_world_position": (3,),
}


def _metadata(archive: np.lib.npyio.NpzFile, name: str) -> Any:
    key = f"metadata__{name}"
    if key not in archive:
        return None
    array = archive[key]
    return array.item() if array.ndim == 0 else array.tolist()


def _resolve_manifest(shard: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    run_dir = shard.parent.parent
    return run_dir / path


def validate_dataset(
    dataset_root: str | Path,
    *,
    empty_contact_is_error: bool = False,
) -> ValidationReport:
    """Validate all episode shards and return a serializable report."""

    root = Path(dataset_root).expanduser().resolve()
    report = ValidationReport(dataset_root=str(root))
    coverage: dict[str, set[Any]] = {
        "run_ids": set(),
        "object_ids": set(),
        "candidate_ids": set(),
        "physics_modes": set(),
        "policy_stages": set(),
        "checkpoint_steps": set(),
    }
    validated_sensor_manifests: set[Path] = set()
    validated_object_manifests: set[Path] = set()
    shards = discover_episode_shards(root)
    if not shards:
        report.issues.append(ValidationIssue("error", "", "No episode shards found"))
    for shard in shards:
        relative = str(shard.relative_to(root))
        report.total_bytes += shard.stat().st_size
        try:
            archive_context = np.load(shard, allow_pickle=False)
        except Exception as exc:
            report.issues.append(
                ValidationIssue("error", relative, f"Cannot open NPZ shard: {exc}")
            )
            continue
        with archive_context as archive:
            missing = [
                logical
                for logical in REQUIRED_TIMESTEP_ARRAYS
                if episode_array_key(logical) not in archive
            ]
            if missing:
                report.issues.append(
                    ValidationIssue("error", relative, f"Missing arrays: {missing}")
                )
                continue
            length = int(archive[episode_array_key("rewards")].shape[0])
            report.episode_count += 1
            report.timestep_count += length
            for logical in REQUIRED_TIMESTEP_ARRAYS:
                array = archive[episode_array_key(logical)]
                if array.ndim == 0 or array.shape[0] != length:
                    report.issues.append(
                        ValidationIssue(
                            "error",
                            relative,
                            f"{logical} has shape {array.shape}; expected leading length {length}",
                        )
                    )
                expected_dtype = _EXPECTED_DTYPES.get(logical)
                if expected_dtype is not None and array.dtype != np.dtype(expected_dtype):
                    report.issues.append(
                        ValidationIssue(
                            "error",
                            relative,
                            f"{logical} has dtype {array.dtype}; expected {np.dtype(expected_dtype)}",
                        )
                    )

            oracle_sensor_count = _metadata(archive, "oracle_sensor_count")
            if oracle_sensor_count is None:
                oracle_sensor_count = archive[
                    episode_array_key("observations/sensor_mask")
                ].shape[-1]
            oracle_sensor_count = int(oracle_sensor_count)
            for prefix in ("observations", "next_observations"):
                for field_name in ORACLE_TOUCH_FIELDS:
                    array = archive[episode_array_key(f"{prefix}/oracle_touch/{field_name}")]
                    if array.ndim < 2 or array.shape[1] != oracle_sensor_count:
                        report.issues.append(
                            ValidationIssue(
                                "error",
                                relative,
                                f"{prefix}/oracle_touch/{field_name} sensor axis "
                                f"{array.shape} does not match {oracle_sensor_count}",
                            )
                        )
                    elif array.shape[2:] != _ORACLE_SUFFIX_SHAPES[field_name]:
                        report.issues.append(
                            ValidationIssue(
                                "error",
                                relative,
                                f"{prefix}/oracle_touch/{field_name} has trailing shape "
                                f"{array.shape[2:]}; expected {_ORACLE_SUFFIX_SHAPES[field_name]}",
                            )
                        )
            layout_sensor_count = _metadata(archive, "sensor_count_total")
            sensor_mask = archive[episode_array_key("observations/sensor_mask")]
            if layout_sensor_count is not None and not np.all(
                sensor_mask.sum(axis=1) == int(layout_sensor_count)
            ):
                report.issues.append(
                    ValidationIssue(
                        "error",
                        relative,
                        "sensor_mask active count does not match sensor_count_total",
                    )
                )
            if layout_sensor_count is not None:
                for prefix in ("observations", "next_observations"):
                    policy_touch = archive[
                        episode_array_key(f"{prefix}/policy_touch")
                    ]
                    if (
                        policy_touch.ndim != 2
                        or policy_touch.shape[1] != int(layout_sensor_count)
                    ):
                        report.issues.append(
                            ValidationIssue(
                                "error",
                                relative,
                                f"{prefix}/policy_touch has shape {policy_touch.shape}; "
                                f"expected ({length}, {int(layout_sensor_count)})",
                            )
                        )

            for metadata_name in REQUIRED_RUN_METADATA:
                if _metadata(archive, metadata_name) is None:
                    report.issues.append(
                        ValidationIssue(
                            "error", relative, f"Missing metadata field {metadata_name}"
                        )
                    )
            for manifest_name in ("sensor_manifest_path", "object_manifest_path"):
                value = _metadata(archive, manifest_name)
                resolved = _resolve_manifest(shard, value)
                if resolved is None or not resolved.is_file():
                    report.issues.append(
                        ValidationIssue(
                            "error",
                            relative,
                            f"{manifest_name} does not resolve to a file: {value!r}",
                        )
                    )
                    continue
                if (
                    manifest_name == "sensor_manifest_path"
                    and resolved not in validated_sensor_manifests
                ):
                    validated_sensor_manifests.add(resolved)
                    try:
                        rows = json.loads(resolved.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        report.issues.append(
                            ValidationIssue(
                                "error", relative, f"Invalid sensor manifest: {exc}"
                            )
                        )
                    else:
                        required_sensor_fields = {
                            "sensor_id",
                            "site_name",
                            "touch_sensor_name",
                            "body_name",
                            "finger_id",
                            "link_id",
                            "region_type",
                            "local_position_xyz",
                            "local_normal_xyz",
                            "global_rest_position_xyz",
                            "active_in_layout",
                            "candidate_id",
                        }
                        if len(rows) != oracle_sensor_count:
                            report.issues.append(
                                ValidationIssue(
                                    "error",
                                    relative,
                                    f"Sensor manifest has {len(rows)} rows; "
                                    f"expected {oracle_sensor_count}",
                                )
                            )
                        for index, row in enumerate(rows):
                            missing_fields = sorted(required_sensor_fields.difference(row))
                            if missing_fields:
                                report.issues.append(
                                    ValidationIssue(
                                        "error",
                                        relative,
                                        f"Sensor manifest row {index} is missing {missing_fields}",
                                    )
                                )
                                break
                        active_count = sum(
                            bool(row.get("active_in_layout")) for row in rows
                        )
                        if (
                            layout_sensor_count is not None
                            and active_count != int(layout_sensor_count)
                        ):
                            report.issues.append(
                                ValidationIssue(
                                    "error",
                                    relative,
                                    f"Sensor manifest has {active_count} active rows; "
                                    f"expected {int(layout_sensor_count)}",
                                )
                            )
                if (
                    manifest_name == "object_manifest_path"
                    and resolved not in validated_object_manifests
                ):
                    validated_object_manifests.add(resolved)
                    try:
                        object_row = json.loads(resolved.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        report.issues.append(
                            ValidationIssue(
                                "error", relative, f"Invalid object manifest: {exc}"
                            )
                        )
                    else:
                        required_object_fields = (
                            "object_id",
                            "base_object",
                            "size",
                            "aspect_ratio",
                            "roughness",
                            "rigid_deformable",
                        )
                        missing_fields = [
                            name
                            for name in required_object_fields
                            if name not in object_row
                        ]
                        if missing_fields:
                            report.issues.append(
                                ValidationIssue(
                                    "error",
                                    relative,
                                    f"Object manifest is missing fields: {missing_fields}",
                                )
                            )
                        unknown_fields = [
                            name
                            for name in required_object_fields
                            if object_row.get(name) is None
                        ]
                        if unknown_fields:
                            report.issues.append(
                                ValidationIssue(
                                    "warning",
                                    relative,
                                    f"Object metadata values are unknown: {unknown_fields}",
                                )
                            )

            contact = archive[
                episode_array_key("next_observations/oracle_touch/contact_active")
            ]
            if not bool(contact.any()):
                report.empty_contact_episode_count += 1
                report.issues.append(
                    ValidationIssue(
                        "error" if empty_contact_is_error else "warning",
                        relative,
                        "Episode contains no oracle contact",
                    )
                )
            for field_name, metadata_name in (
                ("run_ids", "run_id"),
                ("object_ids", "object_id"),
                ("candidate_ids", "candidate_id"),
                ("physics_modes", "physics_mode"),
                ("policy_stages", "policy_stage"),
                ("checkpoint_steps", "policy_checkpoint_step"),
            ):
                value = _metadata(archive, metadata_name)
                if value is not None:
                    coverage[field_name].add(value)

    report.coverage = {
        key: sorted(values, key=lambda item: str(item)) for key, values in coverage.items()
    }
    return report


def write_validation_report(
    report: ValidationReport, output_path: str | Path
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target
