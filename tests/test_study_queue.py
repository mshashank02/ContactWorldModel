import tempfile
import unittest
from pathlib import Path

from study_common import Candidate
from study_queue import StudyQueue


class StudyQueueTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tempdir.name) / "study.db")
        self.queue = StudyQueue(self.db_path)
        self.queue.register_hosts(
            [
                {
                    "host": "pc1",
                    "ssh_target": "user@pc1",
                    "gpu_count": 4,
                    "work_root": "/tmp/work",
                    "python_root": "/tmp/work/envs/shadowhand",
                    "repo_path": "/tmp/work/ShadowHand-TQC",
                    "python_bin": "/tmp/work/envs/shadowhand/bin/python",
                    "priority": 100,
                    "role": "worker",
                }
            ]
        )
        self.candidate = Candidate(candidate_id="n0010_a0p1_b0p1", N=10, alpha=0.1, beta=0.1)
        self.queue.register_candidates([self.candidate])

    def tearDown(self):
        self.queue.close()
        self.tempdir.cleanup()

    def _jobs(self):
        return [
            {
                "object_id": "obj_a_low_small",
                "physics_mode": "deformable",
                "base_object": "obj_a",
                "aspect_ratio": "low",
                "size": "small",
                "seed": 0,
                "priority": 0,
                "artifact_relpath": "generated/studies/test/candidates/n0010_a0p1_b0p1/obj_a_low_small/deformable",
                "metrics_relpath": "generated/studies/test/candidates/n0010_a0p1_b0p1/obj_a_low_small/deformable/metrics.json",
                "stdout_relpath": "generated/studies/test/candidates/n0010_a0p1_b0p1/obj_a_low_small/deformable/stdout.txt",
                "stderr_relpath": "generated/studies/test/candidates/n0010_a0p1_b0p1/obj_a_low_small/deformable/stderr.txt",
                "payload": {"candidate_id": "n0010_a0p1_b0p1"},
            },
            {
                "object_id": "obj_a_low_small",
                "physics_mode": "rigid",
                "base_object": "obj_a",
                "aspect_ratio": "low",
                "size": "small",
                "seed": 0,
                "priority": 1,
                "artifact_relpath": "generated/studies/test/candidates/n0010_a0p1_b0p1/obj_a_low_small/rigid",
                "metrics_relpath": "generated/studies/test/candidates/n0010_a0p1_b0p1/obj_a_low_small/rigid/metrics.json",
                "stdout_relpath": "generated/studies/test/candidates/n0010_a0p1_b0p1/obj_a_low_small/rigid/stdout.txt",
                "stderr_relpath": "generated/studies/test/candidates/n0010_a0p1_b0p1/obj_a_low_small/rigid/stderr.txt",
                "payload": {"candidate_id": "n0010_a0p1_b0p1"},
            },
        ]

    def test_candidate_completes_after_all_jobs_succeed(self):
        self.queue.enqueue_candidate_jobs(self.candidate, self._jobs(), source="sobol")
        self.queue.worker_heartbeat("pc1", "worker-1", 12345, details={})

        first_job = self.queue.claim_job("pc1", "worker-1", 0)
        second_job = self.queue.claim_job("pc1", "worker-1", 1)
        self.assertIsNotNone(first_job)
        self.assertIsNotNone(second_job)

        self.queue.finish_job(int(first_job["job_id"]), 0.6)
        self.queue.finish_job(int(second_job["job_id"]), 0.4)

        completed = self.queue.completed_candidates()
        self.assertEqual(len(completed), 1)
        self.assertAlmostEqual(float(completed[0]["score"]), 0.5, places=6)
        self.assertAlmostEqual(float(completed[0]["rigid_mean"]), 0.4, places=6)
        self.assertAlmostEqual(float(completed[0]["deformable_mean"]), 0.6, places=6)

    def test_retryable_failures_return_job_to_pending(self):
        self.queue.enqueue_candidate_jobs(self.candidate, self._jobs(), source="sobol")
        self.queue.worker_heartbeat("pc1", "worker-1", 12345, details={})

        first_job = self.queue.claim_job("pc1", "worker-1", 0)
        result = self.queue.fail_job(int(first_job["job_id"]), "temporary issue", max_attempts=3)
        self.assertEqual(result["status"], "pending")

        reclaimed = self.queue.claim_job("pc1", "worker-1", 0)
        self.assertEqual(int(reclaimed["job_id"]), int(first_job["job_id"]))

    def test_requeue_stale_jobs(self):
        self.queue.enqueue_candidate_jobs(self.candidate, self._jobs(), source="sobol")
        self.queue.worker_heartbeat("pc1", "worker-1", 12345, details={})
        first_job = self.queue.claim_job("pc1", "worker-1", 0, lease_seconds=1)
        self.queue.conn.execute("UPDATE workers SET last_seen_at = '2000-01-01T00:00:00+00:00' WHERE worker_id = 'worker-1'")
        self.queue.conn.commit()

        requeued = self.queue.requeue_stale_jobs(heartbeat_timeout_seconds=1)
        self.assertIn(int(first_job["job_id"]), requeued)
        row = self.queue.conn.execute("SELECT status FROM jobs WHERE id = ?", (int(first_job["job_id"]),)).fetchone()
        self.assertEqual(row["status"], "pending")


if __name__ == "__main__":
    unittest.main()
