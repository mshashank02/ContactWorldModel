import tempfile
import unittest
from pathlib import Path

from study_common import (
    build_candidate_grid,
    load_cluster_config,
    load_study_manifest,
    sobol_initial_candidates,
)


class StudyCommonTests(unittest.TestCase):
    def test_candidate_grid_has_expected_size(self):
        grid = build_candidate_grid()
        self.assertEqual(len(grid), 486)
        self.assertEqual(len({candidate.candidate_id for candidate in grid}), 486)

    def test_sobol_initial_candidates_are_unique_and_on_grid(self):
        candidates = sobol_initial_candidates(12)
        allowed_ids = {candidate.candidate_id for candidate in build_candidate_grid()}
        self.assertEqual(len(candidates), 12)
        self.assertEqual(len({candidate.candidate_id for candidate in candidates}), 12)
        self.assertTrue(all(candidate.candidate_id in allowed_ids for candidate in candidates))

    def test_load_study_manifest_validates_expected_combos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.csv"
            rows = [
                ("obj_a_low_small", "obj_a_low_small.msh", "obj_a", "low", "small"),
                ("obj_a_low_large", "obj_a_low_large.msh", "obj_a", "low", "large"),
                ("obj_a_high_small", "obj_a_high_small.msh", "obj_a", "high", "small"),
                ("obj_a_high_large", "obj_a_high_large.msh", "obj_a", "high", "large"),
                ("obj_b_low_small", "obj_b_low_small.msh", "obj_b", "low", "small"),
                ("obj_b_low_large", "obj_b_low_large.msh", "obj_b", "low", "large"),
                ("obj_b_high_small", "obj_b_high_small.msh", "obj_b", "high", "small"),
                ("obj_b_high_large", "obj_b_high_large.msh", "obj_b", "high", "large"),
            ]
            manifest_path.write_text(
                "object_id,msh_file,base_object,aspect_ratio,size\n"
                + "\n".join(",".join(row) for row in rows)
                + "\n",
                encoding="utf-8",
            )
            for _, mesh_name, *_ in rows:
                (root / mesh_name).write_text("mesh", encoding="utf-8")

            objects = load_study_manifest(str(root), expected_base_objects=2)
            self.assertEqual(len(objects), 8)
            self.assertEqual(objects[0].aspect_ratio, "high")

    def test_load_cluster_config_computes_repo_path_and_python_bin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "cluster_hosts.yaml"
            config_path.write_text(
                """
coordinator:
  host: coordinator
  ssh_target: user@coordinator
  gpu_count: 0
  work_root: /tmp/work
  python_root: /tmp/work/envs/shadowhand
  priority: 100
hosts:
  - host: pc1
    ssh_target: user@pc1
    gpu_count: 4
    work_root: /tmp/work
    python_root: /tmp/work/envs/shadowhand
    priority: 90
""".strip(),
                encoding="utf-8",
            )
            cluster_cfg = load_cluster_config(str(config_path), repo_dirname="ShadowHand-TQC")
            self.assertEqual(cluster_cfg.coordinator.repo_path, "/tmp/work/ShadowHand-TQC")
            self.assertTrue(cluster_cfg.coordinator.python_bin.endswith("/bin/python"))
            self.assertEqual(cluster_cfg.hosts[0].host, "pc1")

    def test_coordinator_can_also_be_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "cluster_hosts.yaml"
            config_path.write_text(
                """
coordinator:
  host: pc2
  ssh_target: user@pc2
  gpu_count: 2
  work_root: /tmp/work
  python_root: /tmp/work/envs/shadowhand
  priority: 100
  role: coordinator
  run_worker: true
hosts:
  - host: pc1
    ssh_target: user@pc1
    gpu_count: 4
    work_root: /tmp/work
    python_root: /tmp/work/envs/shadowhand
    priority: 90
""".strip(),
                encoding="utf-8",
            )
            cluster_cfg = load_cluster_config(str(config_path), repo_dirname="ShadowHand-TQC")
            worker_names = [host.host for host in cluster_cfg.worker_hosts()]
            self.assertEqual(worker_names, ["pc2", "pc1"])


if __name__ == "__main__":
    unittest.main()
