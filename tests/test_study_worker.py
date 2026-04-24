import unittest
from pathlib import Path

from study_common import HostConfig
from study_worker import (
    FreeGpuTracker,
    build_job_command,
    free_gpu_ids,
    parse_compute_query_output,
    parse_gpu_query_output,
)


class StudyWorkerTests(unittest.TestCase):
    def _host_cfg(self, num_envs_per_job=11):
        return HostConfig(
            host="pc1",
            ssh_target="user@pc1",
            gpu_count=4,
            cpu_cores=48,
            num_envs_per_job=num_envs_per_job,
            work_root="/tmp/work",
            python_root="/tmp/work/envs/shadowhand",
            priority=100,
            repo_path="/tmp/work/ShadowHand-TQC",
            python_bin="/tmp/work/envs/shadowhand/bin/python",
        )

    def test_parse_gpu_query_output(self):
        rows = parse_gpu_query_output("0, GPU-aaa, 100, 5\n1, GPU-bbb, 2048, 90\n")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["index"], 0)
        self.assertEqual(rows[1]["utilization_gpu"], 90)

    def test_parse_compute_query_output(self):
        uuids = parse_compute_query_output("GPU-aaa, 1234\nGPU-bbb, 5678\n")
        self.assertEqual(uuids, {"GPU-aaa", "GPU-bbb"})

    def test_free_gpu_ids_requires_low_usage_and_no_foreign_process(self):
        snapshot = [
            {"index": 0, "uuid": "GPU-aaa", "memory_used_mib": 100, "utilization_gpu": 5, "has_compute_process": False},
            {"index": 1, "uuid": "GPU-bbb", "memory_used_mib": 2000, "utilization_gpu": 5, "has_compute_process": False},
            {"index": 2, "uuid": "GPU-ccc", "memory_used_mib": 100, "utilization_gpu": 50, "has_compute_process": False},
            {"index": 3, "uuid": "GPU-ddd", "memory_used_mib": 100, "utilization_gpu": 5, "has_compute_process": True},
        ]
        free = free_gpu_ids(snapshot, memory_threshold_mib=512, utilization_threshold=10, allow_foreign_processes=False)
        self.assertEqual(free, {0})

    def test_free_gpu_tracker_requires_consecutive_polls(self):
        tracker = FreeGpuTracker(required_free_polls=2)
        self.assertEqual(tracker.update({0}), set())
        self.assertEqual(tracker.update({0}), {0})
        self.assertEqual(tracker.update(set()), set())

    def test_build_job_command_forwards_object_size_label_and_host_num_envs(self):
        job = {
            "artifact_relpath": "generated/run",
            "object_msh_relpath": "objects/example_medium.msh",
            "base_xml_relpath": "assets/hand_base.xml",
            "candidate_id": "n0010_a0p1_b0p1",
            "object_id": "obj_a_low_medium",
            "size": "medium",
            "physics_mode": "rigid",
            "Ntotal": 10,
            "Rppx": 0.1,
            "Rpt": 0.1,
            "seed": 0,
            "eval_episodes": 2,
            "trainer_args": [],
        }

        cmd = build_job_command(job, Path("/tmp/repo"), self._host_cfg())
        self.assertIn("--object-size", cmd)
        self.assertIn("medium", cmd)
        self.assertIn("--num-envs", cmd)
        self.assertIn("11", cmd)

    def test_build_job_command_respects_explicit_num_envs_override(self):
        job = {
            "artifact_relpath": "generated/run",
            "object_msh_relpath": "objects/example_medium.msh",
            "base_xml_relpath": "assets/hand_base.xml",
            "candidate_id": "n0010_a0p1_b0p1",
            "object_id": "obj_a_low_medium",
            "size": "medium",
            "physics_mode": "rigid",
            "Ntotal": 10,
            "Rppx": 0.1,
            "Rpt": 0.1,
            "seed": 0,
            "eval_episodes": 2,
            "trainer_args": ["--num-envs", "3"],
        }

        cmd = build_job_command(job, Path("/tmp/repo"), self._host_cfg())
        num_envs_idx = cmd.index("--num-envs")
        self.assertEqual(cmd[num_envs_idx + 1], "3")


if __name__ == "__main__":
    unittest.main()
