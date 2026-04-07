import argparse
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from study_common import Candidate, summarize_candidate_scores


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True)


class StudyQueue:
    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.init_schema()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self):
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hosts (
                host TEXT PRIMARY KEY,
                ssh_target TEXT NOT NULL,
                gpu_count INTEGER NOT NULL,
                work_root TEXT NOT NULL,
                python_root TEXT NOT NULL,
                repo_path TEXT NOT NULL,
                python_bin TEXT NOT NULL,
                priority INTEGER NOT NULL,
                role TEXT NOT NULL DEFAULT 'worker',
                last_seen_at TEXT
            );

            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                host TEXT NOT NULL,
                pid INTEGER,
                status TEXT NOT NULL DEFAULT 'active',
                last_seen_at TEXT,
                details_json TEXT,
                FOREIGN KEY(host) REFERENCES hosts(host)
            );

            CREATE TABLE IF NOT EXISTS heartbeats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                host TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                details_json TEXT
            );

            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                N INTEGER NOT NULL,
                alpha REAL NOT NULL,
                beta REAL NOT NULL,
                source TEXT,
                selection_order INTEGER,
                status TEXT NOT NULL DEFAULT 'new',
                queued_at TEXT,
                completed_at TEXT,
                score REAL,
                rigid_mean REAL,
                deformable_mean REAL,
                base_object_means_json TEXT,
                aspect_ratio_means_json TEXT,
                size_means_json TEXT,
                last_error TEXT
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                object_id TEXT NOT NULL,
                physics_mode TEXT NOT NULL,
                base_object TEXT NOT NULL,
                aspect_ratio TEXT NOT NULL,
                size TEXT NOT NULL,
                seed INTEGER NOT NULL DEFAULT 0,
                priority INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                host TEXT,
                worker_id TEXT,
                gpu_id INTEGER,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                lease_expires_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                artifact_relpath TEXT NOT NULL,
                metrics_relpath TEXT NOT NULL,
                stdout_relpath TEXT NOT NULL,
                stderr_relpath TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                score REAL,
                last_error TEXT,
                FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id),
                UNIQUE(candidate_id, object_id, physics_mode, seed)
            );

            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                attempt_number INTEGER NOT NULL,
                host TEXT,
                worker_id TEXT,
                gpu_id INTEGER,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                error_text TEXT,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            );
            """
        )

    def set_metadata(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO metadata(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, to_json(value)),
        )
        self.conn.commit()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    def register_hosts(self, hosts: Iterable[Dict[str, Any]]) -> None:
        with self.transaction():
            for host in hosts:
                self.conn.execute(
                    """
                    INSERT INTO hosts(host, ssh_target, gpu_count, work_root, python_root, repo_path, python_bin, priority, role, last_seen_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(host) DO UPDATE SET
                        ssh_target=excluded.ssh_target,
                        gpu_count=excluded.gpu_count,
                        work_root=excluded.work_root,
                        python_root=excluded.python_root,
                        repo_path=excluded.repo_path,
                        python_bin=excluded.python_bin,
                        priority=excluded.priority,
                        role=excluded.role
                    """,
                    (
                        host["host"],
                        host["ssh_target"],
                        int(host["gpu_count"]),
                        host["work_root"],
                        host["python_root"],
                        host["repo_path"],
                        host["python_bin"],
                        int(host["priority"]),
                        host.get("role", "worker"),
                        host.get("last_seen_at"),
                    ),
                )

    def register_candidates(self, candidates: Sequence[Candidate]) -> None:
        with self.transaction():
            for candidate in candidates:
                self.conn.execute(
                    """
                    INSERT INTO candidates(candidate_id, N, alpha, beta, status)
                    VALUES(?, ?, ?, ?, 'new')
                    ON CONFLICT(candidate_id) DO UPDATE SET
                        N=excluded.N,
                        alpha=excluded.alpha,
                        beta=excluded.beta
                    """,
                    (candidate.candidate_id, candidate.N, candidate.alpha, candidate.beta),
                )

    def list_candidates(self, statuses: Optional[Sequence[str]] = None) -> List[sqlite3.Row]:
        query = "SELECT * FROM candidates"
        params: List[Any] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY selection_order IS NULL, selection_order, candidate_id"
        return list(self.conn.execute(query, params))

    def completed_candidates(self) -> List[sqlite3.Row]:
        return self.list_candidates(statuses=["completed"])

    def available_candidates(self) -> List[sqlite3.Row]:
        return self.list_candidates(statuses=["new"])

    def active_candidates(self) -> List[sqlite3.Row]:
        return self.list_candidates(statuses=["queued", "running"])

    def has_candidate_jobs(self, candidate_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM jobs WHERE candidate_id = ? LIMIT 1", (candidate_id,)).fetchone()
        return row is not None

    def enqueue_candidate_jobs(self, candidate: Candidate, jobs: Sequence[Dict[str, Any]], source: str) -> None:
        if not jobs:
            raise ValueError("Candidate must enqueue at least one job.")

        with self.transaction():
            current = self.conn.execute(
                "SELECT selection_order FROM candidates WHERE candidate_id = ?", (candidate.candidate_id,)
            ).fetchone()
            if current is None:
                raise ValueError(f"Unknown candidate {candidate.candidate_id}")

            if self.has_candidate_jobs(candidate.candidate_id):
                return

            max_order_row = self.conn.execute("SELECT COALESCE(MAX(selection_order), 0) AS value FROM candidates").fetchone()
            next_order = (max_order_row["value"] or 0) + 1
            self.conn.execute(
                """
                UPDATE candidates
                SET source = ?, selection_order = COALESCE(selection_order, ?), status = 'queued', queued_at = ?, last_error = NULL
                WHERE candidate_id = ?
                """,
                (source, next_order, utcnow(), candidate.candidate_id),
            )

            for job in jobs:
                self.conn.execute(
                    """
                    INSERT INTO jobs(
                        candidate_id, object_id, physics_mode, base_object, aspect_ratio, size, seed,
                        priority, status, artifact_relpath, metrics_relpath, stdout_relpath,
                        stderr_relpath, payload_json
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.candidate_id,
                        job["object_id"],
                        job["physics_mode"],
                        job["base_object"],
                        job["aspect_ratio"],
                        job["size"],
                        int(job.get("seed", 0)),
                        int(job.get("priority", 0)),
                        job["artifact_relpath"],
                        job["metrics_relpath"],
                        job["stdout_relpath"],
                        job["stderr_relpath"],
                        to_json(job["payload"]),
                    ),
                )

    def worker_heartbeat(self, host: str, worker_id: str, pid: Optional[int], details: Optional[Dict[str, Any]] = None) -> None:
        now = utcnow()
        details_json = to_json(details or {})
        with self.transaction():
            self.conn.execute("UPDATE hosts SET last_seen_at = ? WHERE host = ?", (now, host))
            self.conn.execute(
                """
                INSERT INTO workers(worker_id, host, pid, last_seen_at, details_json)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    host=excluded.host,
                    pid=excluded.pid,
                    last_seen_at=excluded.last_seen_at,
                    status='active',
                    details_json=excluded.details_json
                """,
                (worker_id, host, pid, now, details_json),
            )
            self.conn.execute(
                "INSERT INTO heartbeats(host, worker_id, created_at, details_json) VALUES(?, ?, ?, ?)",
                (host, worker_id, now, details_json),
            )

    def refresh_job_lease(self, job_id: int, lease_seconds: int) -> None:
        expires = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        self.conn.execute("UPDATE jobs SET lease_expires_at = ? WHERE id = ?", (expires, int(job_id)))
        self.conn.commit()

    def claim_job(self, host: str, worker_id: str, gpu_id: int, lease_seconds: int = 7200, max_attempts: int = 3) -> Optional[Dict[str, Any]]:
        expires = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        with self.transaction():
            row = self.conn.execute(
                """
                SELECT jobs.*, candidates.selection_order
                FROM jobs
                JOIN candidates ON candidates.candidate_id = jobs.candidate_id
                WHERE jobs.status = 'pending'
                  AND jobs.attempt_count < ?
                  AND candidates.status IN ('queued', 'running')
                ORDER BY candidates.selection_order, jobs.priority, jobs.id
                LIMIT 1
                """,
                (int(max_attempts),),
            ).fetchone()
            if row is None:
                return None

            next_attempt = int(row["attempt_count"]) + 1
            started_at = utcnow()
            self.conn.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    host = ?,
                    worker_id = ?,
                    gpu_id = ?,
                    attempt_count = ?,
                    lease_expires_at = ?,
                    started_at = COALESCE(started_at, ?),
                    last_error = NULL
                WHERE id = ?
                """,
                (host, worker_id, int(gpu_id), next_attempt, expires, started_at, int(row["id"])),
            )
            self.conn.execute(
                """
                INSERT INTO attempts(job_id, attempt_number, host, worker_id, gpu_id, started_at, status)
                VALUES(?, ?, ?, ?, ?, ?, 'running')
                """,
                (int(row["id"]), next_attempt, host, worker_id, int(gpu_id), started_at),
            )
            self.conn.execute(
                "UPDATE candidates SET status = 'running' WHERE candidate_id = ?",
                (row["candidate_id"],),
            )

            payload = json.loads(row["payload_json"])
            payload.update(
                {
                    "job_id": int(row["id"]),
                    "candidate_id": row["candidate_id"],
                    "object_id": row["object_id"],
                    "physics_mode": row["physics_mode"],
                    "base_object": row["base_object"],
                    "aspect_ratio": row["aspect_ratio"],
                    "size": row["size"],
                    "seed": int(row["seed"]),
                    "artifact_relpath": row["artifact_relpath"],
                    "metrics_relpath": row["metrics_relpath"],
                    "stdout_relpath": row["stdout_relpath"],
                    "stderr_relpath": row["stderr_relpath"],
                    "lease_expires_at": expires,
                }
            )
            return payload

    def job_rows_for_candidate(self, candidate_id: str) -> List[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM jobs WHERE candidate_id = ? ORDER BY priority, object_id, physics_mode",
                (candidate_id,),
            )
        )

    def reconcile_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        job_rows = self.job_rows_for_candidate(candidate_id)
        if not job_rows:
            return None

        statuses = {row["status"] for row in job_rows}
        if "failed" in statuses:
            self.conn.execute(
                "UPDATE candidates SET status = 'failed', completed_at = ?, last_error = ? WHERE candidate_id = ?",
                (utcnow(), "One or more jobs exhausted retry budget.", candidate_id),
            )
            self.conn.commit()
            return {"candidate_id": candidate_id, "status": "failed"}

        if statuses == {"succeeded"}:
            summary = summarize_candidate_scores([dict(row) for row in job_rows])
            self.conn.execute(
                """
                UPDATE candidates
                SET status = 'completed',
                    completed_at = ?,
                    score = ?,
                    rigid_mean = ?,
                    deformable_mean = ?,
                    base_object_means_json = ?,
                    aspect_ratio_means_json = ?,
                    size_means_json = ?,
                    last_error = NULL
                WHERE candidate_id = ?
                """,
                (
                    utcnow(),
                    summary["score"],
                    summary["physics_means"].get("rigid"),
                    summary["physics_means"].get("deformable"),
                    to_json(summary["base_object_means"]),
                    to_json(summary["aspect_ratio_means"]),
                    to_json(summary["size_means"]),
                    candidate_id,
                ),
            )
            self.conn.commit()
            return {"candidate_id": candidate_id, "status": "completed", **summary}

        status = "queued" if statuses <= {"pending"} else "running"
        self.conn.execute("UPDATE candidates SET status = ? WHERE candidate_id = ?", (status, candidate_id))
        self.conn.commit()
        return {"candidate_id": candidate_id, "status": status}

    def finish_job(
        self,
        job_id: int,
        score: float,
        metrics_relpath: Optional[str] = None,
        stdout_relpath: Optional[str] = None,
        stderr_relpath: Optional[str] = None,
        artifact_relpath: Optional[str] = None,
    ) -> Dict[str, Any]:
        finished_at = utcnow()
        with self.transaction():
            row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (int(job_id),)).fetchone()
            if row is None:
                raise ValueError(f"Unknown job id {job_id}")

            self.conn.execute(
                """
                UPDATE jobs
                SET status = 'succeeded',
                    score = ?,
                    completed_at = ?,
                    lease_expires_at = NULL,
                    metrics_relpath = COALESCE(?, metrics_relpath),
                    stdout_relpath = COALESCE(?, stdout_relpath),
                    stderr_relpath = COALESCE(?, stderr_relpath),
                    artifact_relpath = COALESCE(?, artifact_relpath),
                    last_error = NULL
                WHERE id = ?
                """,
                (
                    float(score),
                    finished_at,
                    metrics_relpath,
                    stdout_relpath,
                    stderr_relpath,
                    artifact_relpath,
                    int(job_id),
                ),
            )
            self.conn.execute(
                """
                UPDATE attempts
                SET finished_at = ?, status = 'succeeded'
                WHERE job_id = ? AND attempt_number = ?
                """,
                (finished_at, int(job_id), int(row["attempt_count"])),
            )

        result = self.reconcile_candidate(row["candidate_id"])
        return {"job_id": int(job_id), "status": "succeeded", "candidate": result}

    def fail_job(self, job_id: int, reason: str, max_attempts: int = 3) -> Dict[str, Any]:
        finished_at = utcnow()
        with self.transaction():
            row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (int(job_id),)).fetchone()
            if row is None:
                raise ValueError(f"Unknown job id {job_id}")

            attempts_used = int(row["attempt_count"])
            if attempts_used >= int(max_attempts):
                next_status = "failed"
                completed_at = finished_at
            else:
                next_status = "pending"
                completed_at = None

            self.conn.execute(
                """
                UPDATE jobs
                SET status = ?,
                    lease_expires_at = NULL,
                    host = NULL,
                    worker_id = NULL,
                    gpu_id = NULL,
                    completed_at = ?,
                    last_error = ?
                WHERE id = ?
                """,
                (next_status, completed_at, reason, int(job_id)),
            )
            self.conn.execute(
                """
                UPDATE attempts
                SET finished_at = ?, status = ?, error_text = ?
                WHERE job_id = ? AND attempt_number = ?
                """,
                (finished_at, "failed" if next_status == "failed" else "retryable", reason, int(job_id), attempts_used),
            )

        result = self.reconcile_candidate(row["candidate_id"])
        return {"job_id": int(job_id), "status": next_status, "candidate": result}

    def requeue_stale_jobs(self, heartbeat_timeout_seconds: int = 600) -> List[int]:
        stale_cutoff = (datetime.now(timezone.utc) - timedelta(seconds=heartbeat_timeout_seconds)).isoformat()
        stale_job_ids: List[int] = []
        with self.transaction():
            rows = self.conn.execute(
                """
                SELECT jobs.id, jobs.candidate_id, jobs.attempt_count
                FROM jobs
                LEFT JOIN workers ON workers.worker_id = jobs.worker_id
                WHERE jobs.status = 'running'
                  AND (
                      jobs.lease_expires_at IS NOT NULL AND jobs.lease_expires_at < ?
                      OR workers.last_seen_at IS NULL
                      OR workers.last_seen_at < ?
                  )
                """,
                (utcnow(), stale_cutoff),
            ).fetchall()

            for row in rows:
                stale_job_ids.append(int(row["id"]))
                self.conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'pending',
                        host = NULL,
                        worker_id = NULL,
                        gpu_id = NULL,
                        lease_expires_at = NULL,
                        last_error = 'Lease expired or worker heartbeat lost.'
                    WHERE id = ?
                    """,
                    (int(row["id"]),),
                )
                self.conn.execute(
                    """
                    UPDATE attempts
                    SET finished_at = ?, status = 'lost', error_text = 'Worker heartbeat lost or lease expired.'
                    WHERE job_id = ? AND attempt_number = ?
                    """,
                    (utcnow(), int(row["id"]), int(row["attempt_count"])),
                )

        touched_candidates = {
            row["candidate_id"]
            for row in self.conn.execute("SELECT DISTINCT candidate_id FROM jobs WHERE id IN ({})".format(
                ",".join("?" for _ in stale_job_ids)
            ), stale_job_ids).fetchall()
        } if stale_job_ids else set()
        for candidate_id in touched_candidates:
            self.reconcile_candidate(candidate_id)
        return stale_job_ids

    def summary(self) -> Dict[str, Any]:
        candidate_counts = {
            row["status"]: row["count"]
            for row in self.conn.execute(
                "SELECT status, COUNT(*) AS count FROM candidates GROUP BY status"
            ).fetchall()
        }
        job_counts = {
            row["status"]: row["count"]
            for row in self.conn.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
        }
        return {
            "db_path": self.db_path,
            "candidates": candidate_counts,
            "jobs": job_counts,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study queue helper for distributed GPBO runs.")
    parser.add_argument("--db", required=True, help="Path to the study sqlite database.")
    sub = parser.add_subparsers(dest="command", required=True)

    heartbeat = sub.add_parser("heartbeat")
    heartbeat.add_argument("--host", required=True)
    heartbeat.add_argument("--worker-id", required=True)
    heartbeat.add_argument("--pid", type=int, default=None)
    heartbeat.add_argument("--details-json", default="{}")

    claim = sub.add_parser("claim")
    claim.add_argument("--host", required=True)
    claim.add_argument("--worker-id", required=True)
    claim.add_argument("--gpu-id", type=int, required=True)
    claim.add_argument("--lease-seconds", type=int, default=7200)
    claim.add_argument("--max-attempts", type=int, default=3)

    finish = sub.add_parser("finish")
    finish.add_argument("--job-id", type=int, required=True)
    finish.add_argument("--score", type=float, required=True)
    finish.add_argument("--metrics-relpath", default=None)
    finish.add_argument("--stdout-relpath", default=None)
    finish.add_argument("--stderr-relpath", default=None)
    finish.add_argument("--artifact-relpath", default=None)

    fail = sub.add_parser("fail")
    fail.add_argument("--job-id", type=int, required=True)
    fail.add_argument("--reason", required=True)
    fail.add_argument("--max-attempts", type=int, default=3)

    requeue = sub.add_parser("requeue-stale")
    requeue.add_argument("--heartbeat-timeout-seconds", type=int, default=600)

    sub.add_parser("summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue = StudyQueue(args.db)
    try:
        if args.command == "heartbeat":
            details = json.loads(args.details_json)
            queue.worker_heartbeat(args.host, args.worker_id, args.pid, details=details)
            print(json.dumps({"ok": True}))
        elif args.command == "claim":
            job = queue.claim_job(
                args.host,
                args.worker_id,
                args.gpu_id,
                lease_seconds=args.lease_seconds,
                max_attempts=args.max_attempts,
            )
            print(json.dumps(job or {}))
        elif args.command == "finish":
            print(
                json.dumps(
                    queue.finish_job(
                        args.job_id,
                        args.score,
                        metrics_relpath=args.metrics_relpath,
                        stdout_relpath=args.stdout_relpath,
                        stderr_relpath=args.stderr_relpath,
                        artifact_relpath=args.artifact_relpath,
                    )
                )
            )
        elif args.command == "fail":
            print(json.dumps(queue.fail_job(args.job_id, args.reason, max_attempts=args.max_attempts)))
        elif args.command == "requeue-stale":
            print(json.dumps({"requeued_job_ids": queue.requeue_stale_jobs(args.heartbeat_timeout_seconds)}))
        elif args.command == "summary":
            print(json.dumps(queue.summary(), indent=2))
        else:  # pragma: no cover
            raise ValueError(f"Unsupported command {args.command}")
    finally:
        queue.close()


if __name__ == "__main__":
    main()
