#!/usr/bin/env python3
"""Collect policy and passive oracle-touch trajectories from Shadow Hand tasks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_envs.dynamic_touch_env import DynamicXMLTouchEnv
from custom_wrappers.oracle_touch_wrapper import OracleTouchWrapper
from custom_wrappers.structured_observation_wrapper import StructuredObservationWrapper
from custom_wrappers.trajectory_recorder import TrajectoryRecorder
from world_model.dataset_index import build_dataset_index
from world_model.dataset_validation import validate_dataset, write_validation_report
from world_model.object_manifest import build_object_manifest, write_object_manifest
from world_model.schemas import RunMetadata
from world_model.sensor_manifest import (
    build_sensor_manifest,
    sensor_counts,
    write_sensor_manifest,
)


def _checkpoint_step(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"(?:model_)?(\d+)(?:_steps)?\.zip$", Path(path).name)
    return int(match.group(1)) if match else None


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML configs require pyyaml; JSON configs need no parser extra") from exc
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Collection config must contain a JSON/YAML object")
    return value


def _format_templates(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(context)
    if isinstance(value, dict):
        return {key: _format_templates(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_format_templates(item, context) for item in value]
    return value


def _expand_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand either flat jobs or a compact object/layout/stage pilot matrix."""

    defaults = dict(config.get("defaults", {}))
    jobs = config.get("jobs")
    if isinstance(jobs, list) and jobs:
        return [{**defaults, **job} for job in jobs]
    matrix = config.get("pilot_matrix")
    if not isinstance(matrix, dict):
        raise ValueError("Config requires either a non-empty jobs list or pilot_matrix")
    objects = matrix.get("objects", [])
    layouts = matrix.get("layouts", [])
    stages = matrix.get("stages", [])
    if not all(isinstance(items, list) and items for items in (objects, layouts, stages)):
        raise ValueError("pilot_matrix requires non-empty objects, layouts, and stages lists")
    expected = matrix.get("expected_counts", {"objects": 6, "layouts": 8, "stages": 5})
    actual = {"objects": len(objects), "layouts": len(layouts), "stages": len(stages)}
    for name, count in expected.items():
        if actual.get(name) != int(count):
            raise ValueError(
                f"Pilot matrix has {actual.get(name)} {name}; expected {int(count)}"
            )
    expanded: list[dict[str, Any]] = []
    matrix_defaults = {
        "episodes": int(matrix.get("episodes_per_combination", 100)),
        **matrix.get("defaults", {}),
    }
    for object_row in objects:
        for layout_row in layouts:
            for stage_row in stages:
                context = {
                    **defaults,
                    **matrix_defaults,
                    **object_row,
                    **layout_row,
                    **stage_row,
                }
                expanded.append(_format_templates(context, context))
    return expanded


def _absolute(value: str | None, config_dir: Path | None = None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and config_dir is not None:
        path = config_dir / path
    return str(path.resolve())


def _run_id(job: dict[str, Any]) -> str:
    explicit = job.get("run_id")
    if explicit:
        return str(explicit)
    parts = (
        job["object_id"],
        job["candidate_id"],
        job.get("physics_mode", "rigid"),
        job.get("policy_stage", "random"),
        f"seed{int(job.get('seed', 0))}",
    )
    return "__".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", str(part)) for part in parts)


def _build_env(job: dict[str, Any], sensors: list[Any]):
    from gymnasium.wrappers import TimeLimit
    from sb3_contrib.common.wrappers import TimeFeatureWrapper
    from stable_baselines3.common.monitor import Monitor

    env = DynamicXMLTouchEnv(
        xml_path=job["xml_path"],
        target_position=job.get("target_position", "random"),
        target_rotation=job.get("target_rotation", "xyz"),
        ignore_z_target_rotation=bool(job.get("ignore_z_rot", False)),
        touch_get_obs=job.get("touch_get_obs", "sensordata"),
        action_scale=float(job.get("action_scale", 1.0)),
        action_clip=job.get("action_clip"),
        action_smoothing=float(job.get("action_smoothing", 0.0)),
        reset_settle_steps=int(job.get("reset_settle_steps", 0)),
    )
    env = TimeLimit(env, max_episode_steps=int(job.get("max_episode_steps", 100)))
    env = Monitor(env)
    env = TimeFeatureWrapper(env)
    env = OracleTouchWrapper(
        env,
        sensors=sensors,
        contact_margin=float(job.get("contact_margin", 1.0e-3)),
        object_body_name=job.get("object_body_name", "object"),
    )
    env = StructuredObservationWrapper(env, sensors=sensors, preserve_original=True)
    return env


def _run_metadata(job: dict[str, Any], run_id: str, sensors: list[Any]) -> RunMetadata:
    checkpoint = _absolute(job.get("checkpoint"))
    policy_stage = str(job.get("policy_stage", "random"))
    checkpoint_step = job.get("policy_checkpoint_step", _checkpoint_step(checkpoint))
    known = {item.name for item in fields(RunMetadata)}
    layout_counts = sensor_counts(sensors, active_only=True)
    values: dict[str, Any] = {
        "run_id": run_id,
        "candidate_id": str(job["candidate_id"]),
        "object_id": str(job["object_id"]),
        "physics_mode": str(job["physics_mode"]),
        "seed": int(job.get("seed", 0)),
        "policy_type": "random" if policy_stage == "random" else "TQC",
        "policy_stage": policy_stage,
        "policy_checkpoint_step": checkpoint_step,
        "policy_checkpoint_path": checkpoint,
        "oracle_sensor_count": len(sensors),
        **layout_counts,
    }
    for name in ("alpha", "beta", "Rppx", "Rpt"):
        if job.get(name) is not None:
            values[name] = float(job[name])
    if values.get("alpha") is not None and values.get("beta") is not None:
        total = layout_counts["sensor_count_total"]
        palm = int(round(float(values["alpha"]) * total))
        non_palm = max(total - palm, 0)
        tip = int(round(float(values["beta"]) * non_palm))
        non_tip = max(non_palm - tip, 0)
        values.setdefault("Rppx", float(palm) / max(float(non_tip), 1.0e-6))
        values.setdefault("Rpt", float(palm) / max(float(tip), 1.0e-6))
    values["extra"] = {
        key: value
        for key, value in job.items()
        if key not in known and key not in {"checkpoint", "vecnormalize"}
    }
    return RunMetadata(**values)


def _write_run_files(
    run_dir: Path,
    metadata: RunMetadata,
    sensors: list[Any],
    job: dict[str, Any],
) -> None:
    write_sensor_manifest(run_dir / metadata.sensor_manifest_path, sensors)
    object_overrides = dict(job.get("object_metadata") or {})
    for name in (
        "base_object",
        "size",
        "aspect_ratio",
        "roughness",
        "macro_geometry",
        "geometry",
        "material",
    ):
        if job.get(name) is not None:
            object_overrides[name] = job[name]
    object_manifest = build_object_manifest(
        object_id=metadata.object_id,
        physics_mode=metadata.physics_mode,
        xml_path=job["xml_path"],
        manifest_csv=job.get("object_manifest_csv"),
        overrides=object_overrides,
    )
    write_object_manifest(run_dir / metadata.object_manifest_path, object_manifest)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _collect_random(recorder: TrajectoryRecorder, episodes: int, seed: int) -> None:
    observation, _ = recorder.reset(seed=seed)
    del observation
    start = recorder.episodes_written
    rng = np.random.default_rng(seed)
    while recorder.episodes_written - start < episodes:
        action = recorder.action_space.sample()
        # Sampling through the seeded NumPy generator makes behavior independent
        # of Gym space implementation details.
        if hasattr(recorder.action_space, "low"):
            action = rng.uniform(
                recorder.action_space.low, recorder.action_space.high
            ).astype(recorder.action_space.dtype)
        _, _, terminated, truncated, _ = recorder.step(action)
        if terminated or truncated:
            recorder.reset()


def _collect_checkpoint(
    recorder: TrajectoryRecorder,
    episodes: int,
    checkpoint: str,
    vecnormalize: str | None,
    seed: int,
    deterministic: bool,
    device: str,
) -> None:
    from sb3_contrib import TQC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vector_env = DummyVecEnv([lambda: recorder])
    if vecnormalize:
        vector_env = VecNormalize.load(vecnormalize, vector_env)
        vector_env.training = False
        vector_env.norm_reward = False
    vector_env.seed(seed)
    model = TQC.load(checkpoint, env=vector_env, device=device)
    start = recorder.episodes_written
    observation = vector_env.reset()
    while recorder.episodes_written - start < episodes:
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, _, _, _ = vector_env.step(action)
    vector_env.close()


def collect_job(job: dict[str, Any], config_dir: Path | None = None) -> Path:
    """Collect one object-layout-policy-stage job and return its run directory."""

    required = ("xml_path", "oracle_site_xml", "dataset_root", "object_id", "candidate_id")
    missing = [name for name in required if not job.get(name)]
    if missing:
        raise ValueError(f"Collection job is missing required fields: {missing}")
    job = dict(job)
    job.setdefault("physics_mode", "rigid")
    job.setdefault("policy_stage", "random")
    job.setdefault("episodes", 1)
    for name in (
        "xml_path",
        "oracle_site_xml",
        "oracle_sensor_xml",
        "dataset_root",
        "checkpoint",
        "vecnormalize",
        "object_manifest_csv",
    ):
        job[name] = _absolute(job.get(name), config_dir)
    if job.get("checkpoint") and not job.get("vecnormalize"):
        step = job.get("policy_checkpoint_step") or _checkpoint_step(job["checkpoint"])
        if step is not None:
            inferred = Path(job["checkpoint"]).with_name(f"vecnorm_{int(step)}.pkl")
            if inferred.is_file():
                job["vecnormalize"] = str(inferred)
    for name in ("xml_path", "oracle_site_xml", "checkpoint", "vecnormalize"):
        if job.get(name) and not Path(job[name]).is_file():
            raise FileNotFoundError(f"{name} does not exist: {job[name]}")

    run_id = _run_id(job)
    run_dir = Path(job["dataset_root"]) / "runs" / run_id
    existing_manifest: dict[str, Any] | None = None
    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.is_file():
        existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    sensors = build_sensor_manifest(
        active_xml_path=job["xml_path"],
        oracle_site_xml_path=job["oracle_site_xml"],
        oracle_sensor_xml_path=job.get("oracle_sensor_xml"),
        candidate_id=str(job["candidate_id"]),
    )
    structured_env = _build_env(job, sensors)
    metadata = _run_metadata(job, run_id, sensors)
    if existing_manifest is not None:
        identity_fields = (
            "run_id",
            "candidate_id",
            "object_id",
            "physics_mode",
            "seed",
            "policy_stage",
            "policy_checkpoint_step",
        )
        mismatched = [
            name
            for name in identity_fields
            if existing_manifest.get(name) != metadata.to_dict().get(name)
        ]
        if mismatched:
            raise ValueError(
                f"Existing run {run_dir} has incompatible metadata fields: {mismatched}"
            )
    recorder = TrajectoryRecorder(
        structured_env,
        run_dir=run_dir,
        run_metadata=metadata,
        compressed=not bool(job.get("uncompressed", False)),
        event_oversample_factor=float(job.get("event_oversample_factor", 2.0)),
    )
    # First reset populates global rest positions before the manifest is saved.
    recorder.reset(seed=metadata.seed)
    _write_run_files(run_dir, metadata, sensors, job)
    target_episodes = int(job["episodes"])
    if target_episodes <= 0:
        raise ValueError("episodes must be positive")
    episodes = max(0, target_episodes - recorder.episodes_written)
    if episodes == 0:
        recorder.close()
        return run_dir
    if metadata.policy_stage == "random":
        _collect_random(recorder, episodes, metadata.seed)
        recorder.close()
    else:
        checkpoint = job.get("checkpoint")
        if not checkpoint:
            raise ValueError(f"Policy stage {metadata.policy_stage!r} requires checkpoint")
        _collect_checkpoint(
            recorder,
            episodes,
            checkpoint,
            job.get("vecnormalize"),
            metadata.seed,
            bool(job.get("deterministic_policy", True)),
            str(job.get("device", "auto")),
        )
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect dense passive oracle touch alongside unchanged policy observations."
    )
    parser.add_argument("--config", type=Path, help="JSON/YAML with defaults and a flat jobs list")
    parser.add_argument("--xml-path")
    parser.add_argument("--oracle-site-xml")
    parser.add_argument("--oracle-sensor-xml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--run-id")
    parser.add_argument("--object-id")
    parser.add_argument("--candidate-id")
    parser.add_argument("--physics-mode", choices=["rigid", "deformable"], default="rigid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--policy-stage", default="random")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vecnormalize")
    parser.add_argument("--policy-checkpoint-step", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic-policy", action="store_true")
    parser.add_argument("--object-manifest-csv")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--Rppx", type=float)
    parser.add_argument("--Rpt", type=float)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--contact-margin", type=float, default=1.0e-3)
    parser.add_argument("--event-oversample-factor", type=float, default=2.0)
    parser.add_argument("--target-position", default="random", choices=["random", "ignore"])
    parser.add_argument("--ignore-z-rot", action="store_true")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--action-clip", type=float)
    parser.add_argument("--action-smoothing", type=float, default=0.0)
    parser.add_argument("--reset-settle-steps", type=int, default=0)
    parser.add_argument("--validate-after", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.config:
        config_path = args.config.expanduser().resolve()
        config = _load_config(config_path)
        merged_jobs = _expand_config(config)
        config_dir = config_path.parent
    else:
        merged_jobs = [
            {
                key: value
                for key, value in vars(args).items()
                if key not in {"config", "validate_after", "stochastic_policy"}
                and value is not None
            }
        ]
        merged_jobs[0]["deterministic_policy"] = not args.stochastic_policy
        config_dir = None

    run_dirs = []
    for job in merged_jobs:
        run_dir = collect_job(job, config_dir)
        run_dirs.append(run_dir)
        print(f"Collected {job.get('episodes', 1)} episode(s): {run_dir}")

    dataset_roots = {
        Path(_absolute(job["dataset_root"], config_dir)).resolve() for job in merged_jobs
    }
    if args.validate_after or bool(args.config and config.get("validate_after", False)):
        for root in sorted(dataset_roots):
            report = validate_dataset(root)
            write_validation_report(report, root / "index" / "validation_report.json")
            summary = build_dataset_index(root) if report.ready else None
            print(json.dumps({"index": summary, "validation": report.to_dict()}, indent=2))
            if not report.ready:
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
