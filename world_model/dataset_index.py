"""Scan episode shards and build Parquet dataset index tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schemas import episode_array_key


def discover_episode_shards(dataset_root: str | Path) -> list[Path]:
    """Return all NPZ episode shards below ``dataset_root`` in stable order."""

    return sorted(Path(dataset_root).expanduser().resolve().glob("**/episodes/episode_*.npz"))


def _scalar(archive: np.lib.npyio.NpzFile, name: str, default: Any = None) -> Any:
    key = f"metadata__{name}"
    if key not in archive:
        return default
    value = archive[key]
    if value.ndim == 0:
        return value.item()
    if value.size == 1:
        return value.reshape(()).item()
    return value.tolist()


def read_episode_index_row(shard: str | Path, dataset_root: str | Path) -> dict[str, Any]:
    """Read only metadata and compact episode summaries from a shard."""

    source = Path(shard)
    root = Path(dataset_root).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        rewards = archive[episode_array_key("rewards")]
        contact = archive[episode_array_key("next_observations/oracle_touch/contact_active")]
        row = {
            "shard_path": str(source.resolve().relative_to(root)),
            "shard_bytes": source.stat().st_size,
            "length": int(rewards.shape[0]),
            "return": float(np.asarray(rewards, dtype=np.float64).sum()),
            "has_contact": bool(np.asarray(contact).any()),
        }
        for name in (
            "run_id",
            "episode_id",
            "object_id",
            "candidate_id",
            "physics_mode",
            "seed",
            "policy_type",
            "policy_stage",
            "policy_checkpoint_step",
            "policy_checkpoint_path",
            "sensor_manifest_path",
            "object_manifest_path",
            "sensor_count_total",
            "sensor_count_palm",
            "sensor_count_tip",
            "sensor_count_non_tip",
            "oracle_sensor_count",
            "alpha",
            "beta",
            "Rppx",
            "Rpt",
            "format_version",
            "event_score",
            "sampling_weight",
        ):
            row[name] = _scalar(archive, name)
        run_dir = source.parent.parent
        for manifest_name in ("sensor_manifest_path", "object_manifest_path"):
            value = row.get(manifest_name)
            if value:
                index_name = manifest_name.replace("_path", "_dataset_path")
                manifest_path = Path(str(value))
                if not manifest_path.is_absolute():
                    manifest_path = run_dir / manifest_path
                try:
                    row[index_name] = str(
                        manifest_path.resolve().relative_to(root)
                    )
                except ValueError:
                    row[index_name] = str(manifest_path.resolve())
    return row


def scan_dataset(dataset_root: str | Path) -> list[dict[str, Any]]:
    """Build episode index rows without mutating the dataset."""

    root = Path(dataset_root).expanduser().resolve()
    return [read_episode_index_row(path, root) for path in discover_episode_shards(root)]


def _deduplicate(
    rows: Iterable[dict[str, Any]],
    key_fields: list[str],
    output_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        fields = output_fields or key_fields
        selected.setdefault(key, {field: row.get(field) for field in fields})
    return list(selected.values())


def build_dataset_index(
    dataset_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write episode, run, object, layout, and checkpoint Parquet tables."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency error is actionable
        raise ImportError("Parquet indexing requires pandas and pyarrow") from exc

    root = Path(dataset_root).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve() if output_dir else root / "index"
    target.mkdir(parents=True, exist_ok=True)
    rows = scan_dataset(root)
    tables: dict[str, list[dict[str, Any]]] = {
        "episodes": rows,
        "runs": _deduplicate(
            rows,
            ["run_id"],
            [
                "run_id",
                "object_id",
                "candidate_id",
                "physics_mode",
                "seed",
                "policy_stage",
                "policy_checkpoint_step",
                "sensor_manifest_dataset_path",
                "object_manifest_dataset_path",
            ],
        ),
        "objects": _deduplicate(
            rows,
            ["object_id", "physics_mode"],
            ["object_id", "physics_mode", "object_manifest_dataset_path"],
        ),
        "layouts": _deduplicate(
            rows,
            [
                "candidate_id",
                "sensor_count_total",
                "sensor_count_palm",
                "sensor_count_tip",
                "sensor_count_non_tip",
                "oracle_sensor_count",
                "alpha",
                "beta",
                "Rppx",
                "Rpt",
            ],
            [
                "candidate_id",
                "sensor_count_total",
                "sensor_count_palm",
                "sensor_count_tip",
                "sensor_count_non_tip",
                "oracle_sensor_count",
                "alpha",
                "beta",
                "Rppx",
                "Rpt",
                "sensor_manifest_dataset_path",
            ],
        ),
        "checkpoints": _deduplicate(
            rows,
            [
                "policy_type",
                "policy_stage",
                "policy_checkpoint_step",
                "policy_checkpoint_path",
            ],
        ),
    }
    table_paths: dict[str, str] = {}
    for name, table_rows in tables.items():
        path = target / f"{name}.parquet"
        pd.DataFrame(table_rows).to_parquet(path, index=False)
        table_paths[name] = str(path)

    summary = {
        "dataset_root": str(root),
        "episode_count": len(rows),
        "run_count": len(tables["runs"]),
        "object_count": len(tables["objects"]),
        "layout_count": len(tables["layouts"]),
        "checkpoint_count": len(tables["checkpoints"]),
        "timestep_count": sum(int(row["length"]) for row in rows),
        "total_shard_bytes": sum(int(row["shard_bytes"]) for row in rows),
        "empty_contact_episode_count": sum(not bool(row["has_contact"]) for row in rows),
        "tables": table_paths,
    }
    (target / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
