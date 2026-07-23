import json
from pathlib import Path

import numpy as np

from world_model.dataset_index import build_dataset_index
from world_model.dataset_validation import validate_dataset
from world_model.events import derive_event_tags
from world_model.schemas import ORACLE_TOUCH_FIELDS, RunMetadata, episode_array_key
from world_model.sensor_manifest import build_sensor_manifest


def _write_manifest_fixture(root: Path) -> tuple[Path, Path, Path]:
    sites = root / "sites.xml"
    sites.write_text(
        """
<mujoco>
  <body name="robot0:palm">
    <site name="robot0:T_palm_a" type="box" pos="0 -0.01 0" size="0.01 0.002 0.01"/>
    <site name="robot0:T_palm_b" type="box" pos="0 -0.01 0.02" size="0.01 0.002 0.01"/>
  </body>
</mujoco>
""".strip()
    )
    active_sensors = root / "active_sensors.xml"
    active_sensors.write_text(
        """
<mujoco><sensor>
  <touch name="robot0:ST_Tch_tip" site="robot0:Tch_tip"/>
  <touch name="robot0:TS_palm_b" site="robot0:T_palm_b"/>
</sensor></mujoco>
""".strip()
    )
    active = root / "active.xml"
    active.write_text(
        '<mujoco><include file="active_sensors.xml"/></mujoco>', encoding="utf-8"
    )
    oracle = root / "oracle.xml"
    oracle.write_text(
        """
<mujoco><sensor>
  <touch name="robot0:TS_palm_a" site="robot0:T_palm_a"/>
  <touch name="robot0:TS_palm_b" site="robot0:T_palm_b"/>
</sensor></mujoco>
""".strip(),
        encoding="utf-8",
    )
    return active, sites, oracle


def test_sensor_manifest_tracks_dense_and_sparse_order(tmp_path):
    active, sites, oracle = _write_manifest_fixture(tmp_path)
    rows = build_sensor_manifest(active, sites, "layout-a", oracle)
    assert len(rows) == 2
    assert [row.active_in_layout for row in rows] == [False, True]
    assert rows[1].policy_touch_index == 0
    assert all(row.candidate_id == "layout-a" for row in rows)


def _write_synthetic_episode(dataset_root: Path) -> None:
    run_dir = dataset_root / "runs" / "run-a"
    episodes = run_dir / "episodes"
    episodes.mkdir(parents=True)
    sensor_rows = []
    for sensor_id in range(2):
        sensor_rows.append(
            {
                "sensor_id": sensor_id,
                "site_name": f"site-{sensor_id}",
                "touch_sensor_name": f"touch-{sensor_id}",
                "body_name": "robot0:palm",
                "finger_id": "palm",
                "link_id": "palm",
                "region_type": "palm",
                "local_position_xyz": [0.0, 0.0, 0.0],
                "local_normal_xyz": [0.0, -1.0, 0.0],
                "global_rest_position_xyz": [0.0, 0.0, 0.0],
                "active_in_layout": sensor_id == 1,
                "candidate_id": "layout-a",
            }
        )
    (run_dir / "sensor_manifest.json").write_text(
        json.dumps(sensor_rows), encoding="utf-8"
    )
    (run_dir / "object_manifest.json").write_text(
        json.dumps(
            {
                "object_id": "object-a",
                "base_object": "object-a",
                "size": "small",
                "aspect_ratio": "low",
                "roughness": "low",
                "rigid_deformable": "rigid",
            }
        ),
        encoding="utf-8",
    )
    metadata = RunMetadata(
        run_id="run-a",
        candidate_id="layout-a",
        object_id="object-a",
        physics_mode="rigid",
        seed=7,
        sensor_count_total=1,
        sensor_count_palm=1,
        oracle_sensor_count=2,
        policy_stage="random",
    )
    length = 3
    arrays = {
        episode_array_key("actions"): np.zeros((length, 20), np.float32),
        episode_array_key("rewards"): np.zeros(length, np.float32),
        episode_array_key("terminated"): np.array([False, False, False]),
        episode_array_key("truncated"): np.array([False, False, True]),
        episode_array_key("success"): np.zeros(length, np.float32),
    }
    for prefix in ("observations", "next_observations"):
        arrays[episode_array_key(f"{prefix}/proprioception")] = np.zeros(
            (length, 48), np.float32
        )
        arrays[episode_array_key(f"{prefix}/policy_touch")] = np.zeros(
            (length, 1), np.float32
        )
        arrays[episode_array_key(f"{prefix}/achieved_goal")] = np.zeros(
            (length, 7), np.float32
        )
        arrays[episode_array_key(f"{prefix}/desired_goal")] = np.zeros(
            (length, 7), np.float32
        )
        arrays[episode_array_key(f"{prefix}/sensor_mask")] = np.array(
            [[False, True]] * length
        )
        for name in ORACLE_TOUCH_FIELDS:
            dtype = np.dtype(ORACLE_TOUCH_FIELDS[name])
            suffix = (
                (3,)
                if name in {"contact_position", "contact_normal", "site_world_position"}
                else (2,)
                if name == "tangential_force"
                else ()
            )
            value = np.zeros((length, 2) + suffix, dtype=dtype)
            if name == "contacting_geom_id":
                value.fill(-1)
            arrays[episode_array_key(f"{prefix}/oracle_touch/{name}")] = value
    arrays[
        episode_array_key("next_observations/oracle_touch/contact_active")
    ][1, 0] = True
    for key, value in metadata.to_dict().items():
        if value is not None and not isinstance(value, dict):
            arrays[f"metadata__{key}"] = np.asarray(value)
    arrays["metadata__episode_id"] = np.asarray(0)
    arrays["metadata__event_score"] = np.asarray(1)
    arrays["metadata__sampling_weight"] = np.asarray(3.0)
    np.savez_compressed(episodes / "episode_000000.npz", **arrays)


def test_dataset_index_and_validation(tmp_path):
    _write_synthetic_episode(tmp_path)
    report = validate_dataset(tmp_path)
    assert report.ready
    assert report.episode_count == 1
    assert report.timestep_count == 3
    assert report.empty_contact_episode_count == 0
    summary = build_dataset_index(tmp_path)
    assert summary["episode_count"] == 1
    assert summary["object_count"] == 1
    assert Path(summary["tables"]["episodes"]).is_file()
    saved = json.loads((tmp_path / "index" / "summary.json").read_text())
    assert saved["layout_count"] == 1


def test_event_annotations_include_contact_transitions():
    active = np.array([[False], [True], [True], [False], [True]])
    tangent = np.zeros((5, 1, 2), dtype=np.float32)
    success = np.array([0, 0, 0, 0, 1], dtype=np.float32)
    events = derive_event_tags(active, tangent, success)
    assert events["contact_onset"].tolist() == [False, True, False, False, True]
    assert events["contact_release"].tolist() == [False, False, False, True, False]
    assert events["regrasp_event"][-1]
    assert events["success_transition"][-1]
