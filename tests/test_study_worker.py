import unittest
from pathlib import Path

from study_worker import (
    FreeGpuTracker,
    build_job_command,
    free_gpu_ids,
    parse_compute_query_output,
    parse_gpu_query_output,
)


class StudyWorkerTests(unittest.TestCase):
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

    def test_build_job_command_forwards_object_size_label(self):
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

        cmd = build_job_command(job, Path("/tmp/repo"))
        self.assertIn("--object-size", cmd)
        self.assertIn("medium", cmd)


if __name__ == "__main__":
    unittest.main()
