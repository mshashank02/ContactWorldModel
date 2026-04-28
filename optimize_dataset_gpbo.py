import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from botorch.acquisition import qLogNoisyExpectedImprovement as qLogNEI
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from study_common import (
    Candidate,
    build_candidate_grid,
    build_run_artifact_relpath,
    load_cluster_config,
    load_study_manifest,
    map_alpha_beta_to_ratios,
    resolve_repo_root,
    resolve_study_root,
    sanitize_identifier,
    sobol_initial_candidates,
)
from study_queue import StudyQueue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed discrete GPBO coordinator for dataset-scale sensor search.")
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--objects-root", required=True, help="Path to the study object folder containing manifest.csv.")
    parser.add_argument("--cluster-config", default="cluster_hosts.yaml")
    parser.add_argument("--db", default=None, help="Optional explicit path to the sqlite study DB.")
    parser.add_argument("--base", default="assets/hand_base.xml", help="Base hand XML used by generate_and_train.py.")
    parser.add_argument("--init-candidates", type=int, default=4)
    parser.add_argument("--bo-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--heartbeat-timeout-seconds", type=int, default=900)
    parser.add_argument("--loop-sleep-seconds", type=int, default=60)
    parser.add_argument("--expected-base-objects", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="Forward --force into generate_and_train.py jobs.")
    parser.add_argument("--once", action="store_true", help="Run one coordinator iteration and exit.")
    parser.add_argument("--report-only", action="store_true", help="Export reports from current DB state and exit.")
    parser.add_argument(
        "--preview-initial-only",
        action="store_true",
        help="Initialize the study, print the initial Sobol candidates, export reports, and exit without enqueuing jobs.",
    )
    parser.add_argument("--trainer-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def row_to_candidate(row) -> Candidate:
    return Candidate(candidate_id=row["candidate_id"], N=int(row["N"]), alpha=float(row["alpha"]), beta=float(row["beta"]))


def initialize_study(
    queue: StudyQueue,
    args: argparse.Namespace,
    cluster_cfg,
    study_objects,
    study_root: Path,
    repo_root: Path,
) -> Dict[str, Any]:
    queue.register_hosts([host.to_dict() for host in cluster_cfg.all_hosts()])
    queue.register_candidates(build_candidate_grid())

    objects_root = Path(args.objects_root).expanduser().resolve()
    base_xml = Path(args.base).expanduser().resolve()
    if not base_xml.is_file():
        raise FileNotFoundError(base_xml)

    trainer_args = args.trainer_args[1:] if (args.trainer_args and args.trainer_args[0] == "--") else list(args.trainer_args)
    spec = queue.get_metadata("study_spec")
    if spec is None:
        initial_candidates = sobol_initial_candidates(args.init_candidates)
        spec = {
            "study_name": args.study_name,
            "objects_root_relpath": os.path.relpath(objects_root, repo_root),
            "base_xml_relpath": os.path.relpath(base_xml, repo_root),
            "artifact_root_relpath": os.path.relpath(study_root, repo_root),
            "repo_root": str(repo_root),
            "seed": int(args.seed),
            "eval_episodes": int(args.eval_episodes),
            "force": bool(args.force),
            "trainer_args": trainer_args,
            "init_candidates": int(args.init_candidates),
            "bo_candidates": int(args.bo_candidates),
            "budget_total": int(args.init_candidates + args.bo_candidates),
            "expected_base_objects": int(args.expected_base_objects),
            "initial_candidate_ids": [candidate.candidate_id for candidate in initial_candidates],
        }
        queue.set_metadata("study_spec", spec)
    return spec


def build_jobs_for_candidate(
    candidate: Candidate,
    study_name: str,
    study_objects,
    objects_root_relpath: str,
    base_xml_relpath: str,
    trainer_args: Sequence[str],
    seed: int,
    eval_episodes: int,
    force: bool,
) -> List[Dict[str, Any]]:
    _, _, _, _, Rppx, Rpt = map_alpha_beta_to_ratios(candidate.N, candidate.alpha, candidate.beta)
    jobs: List[Dict[str, Any]] = []
    for obj in study_objects:
        for physics_mode in ("deformable", "rigid"):
            artifact_relpath = build_run_artifact_relpath(study_name, candidate.candidate_id, obj.object_id, physics_mode)
            payload = {
                "study_name": study_name,
                "candidate_id": candidate.candidate_id,
                "candidate": candidate.to_dict(),
                "object_id": obj.object_id,
                "base_object": obj.base_object,
                "aspect_ratio": obj.aspect_ratio,
                "size": obj.size,
                "physics_mode": physics_mode,
                "seed": int(seed),
                "base_xml_relpath": base_xml_relpath,
                "object_msh_relpath": os.path.join(objects_root_relpath, obj.msh_file),
                "artifact_relpath": artifact_relpath,
                "metrics_relpath": os.path.join(artifact_relpath, "metrics.json"),
                "stdout_relpath": os.path.join(artifact_relpath, "stdout.txt"),
                "stderr_relpath": os.path.join(artifact_relpath, "stderr.txt"),
                "Ntotal": int(candidate.N),
                "Rppx": float(Rppx),
                "Rpt": float(Rpt),
                "force": bool(force),
                "trainer_args": list(trainer_args),
                "eval_episodes": int(eval_episodes),
            }
            jobs.append(
                {
                    "object_id": obj.object_id,
                    "physics_mode": physics_mode,
                    "base_object": obj.base_object,
                    "aspect_ratio": obj.aspect_ratio,
                    "size": obj.size,
                    "seed": int(seed),
                    "priority": 0 if physics_mode == "deformable" else 1,
                    "artifact_relpath": artifact_relpath,
                    "metrics_relpath": payload["metrics_relpath"],
                    "stdout_relpath": payload["stdout_relpath"],
                    "stderr_relpath": payload["stderr_relpath"],
                    "payload": payload,
                }
            )
    return jobs


def choose_bo_candidate(completed_rows, available_rows) -> Candidate:
    if len(completed_rows) < 2:
        return row_to_candidate(available_rows[0])

    train_X = torch.tensor(
        [[float(row["N"]), float(row["alpha"]), float(row["beta"])] for row in completed_rows],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[float(row["score"])] for row in completed_rows], dtype=torch.double)
    candidate_X = torch.tensor(
        [[float(row["N"]), float(row["alpha"]), float(row["beta"])] for row in available_rows],
        dtype=torch.double,
    )

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=3),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acq = qLogNEI(model=model, X_baseline=train_X, prune_baseline=True)
    acquisition_values = acq(candidate_X.unsqueeze(1))
    best_idx = int(torch.argmax(acquisition_values).item())
    return row_to_candidate(available_rows[best_idx])


def choose_next_candidate(queue: StudyQueue, spec: Dict[str, Any]) -> Tuple[Optional[Candidate], Optional[str]]:
    selected_rows = queue.list_candidates(statuses=["queued", "running", "completed", "failed"])
    selected_ids = {row["candidate_id"] for row in selected_rows}

    for candidate_id in spec["initial_candidate_ids"]:
        if candidate_id in selected_ids:
            continue
        row = queue.conn.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
        if row is not None:
            return row_to_candidate(row), "sobol"

    completed_rows = queue.completed_candidates()
    available_rows = queue.available_candidates()
    if not available_rows:
        return None, None
    return choose_bo_candidate(completed_rows, available_rows), "bo"


def export_reports(queue: StudyQueue, study_root: Path) -> None:
    study_root.mkdir(parents=True, exist_ok=True)
    spec = queue.get_metadata("study_spec", {})
    candidate_rows = queue.list_candidates()
    candidate_history_path = study_root / "candidate_history.json"
    leaderboard_path = study_root / "leaderboard.csv"
    per_condition_path = study_root / "per_condition_scores.csv"
    study_spec_path = study_root / "study_spec.json"
    initial_candidates_path = study_root / "initial_candidates.json"

    history = []
    for row in candidate_rows:
        history.append(
            {
                "candidate_id": row["candidate_id"],
                "N": int(row["N"]),
                "alpha": float(row["alpha"]),
                "beta": float(row["beta"]),
                "status": row["status"],
                "source": row["source"],
                "selection_order": row["selection_order"],
                "score": row["score"],
                "rigid_mean": row["rigid_mean"],
                "deformable_mean": row["deformable_mean"],
            }
        )
    with candidate_history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    with study_spec_path.open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)

    initial_candidate_rows = []
    for candidate_id in spec.get("initial_candidate_ids", []):
        row = queue.conn.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
        if row is None:
            continue
        initial_candidate_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "N": int(row["N"]),
                "alpha": float(row["alpha"]),
                "beta": float(row["beta"]),
                "status": row["status"],
                "source": row["source"],
                "selection_order": row["selection_order"],
            }
        )
    with initial_candidates_path.open("w", encoding="utf-8") as handle:
        json.dump(initial_candidate_rows, handle, indent=2)

    completed = [row for row in candidate_rows if row["status"] == "completed"]
    completed.sort(key=lambda row: float(row["score"]), reverse=True)
    with leaderboard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_id", "N", "alpha", "beta", "score", "rigid_mean", "deformable_mean", "source"],
        )
        writer.writeheader()
        for row in completed:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "N": int(row["N"]),
                    "alpha": float(row["alpha"]),
                    "beta": float(row["beta"]),
                    "score": row["score"],
                    "rigid_mean": row["rigid_mean"],
                    "deformable_mean": row["deformable_mean"],
                    "source": row["source"],
                }
            )

    with per_condition_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "object_id",
                "physics_mode",
                "base_object",
                "aspect_ratio",
                "size",
                "status",
                "score",
                "artifact_relpath",
                "metrics_relpath",
            ],
        )
        writer.writeheader()
        for row in queue.conn.execute(
            """
            SELECT candidate_id, object_id, physics_mode, base_object, aspect_ratio, size,
                   status, score, artifact_relpath, metrics_relpath
            FROM jobs
            ORDER BY candidate_id, object_id, physics_mode
            """
        ).fetchall():
            writer.writerow(dict(row))


def coordinator_iteration(
    queue: StudyQueue,
    spec: Dict[str, Any],
    study_objects,
    study_root: Path,
) -> Dict[str, Any]:
    requeued = queue.requeue_stale_jobs()
    active = queue.active_candidates()
    completed = queue.completed_candidates()
    status = {
        "requeued_jobs": requeued,
        "active_candidates": [row["candidate_id"] for row in active],
        "completed_candidates": len(completed),
        "budget_total": int(spec["budget_total"]),
    }

    if not active and len(completed) < int(spec["budget_total"]):
        next_candidate, source = choose_next_candidate(queue, spec)
        if next_candidate is not None and source is not None:
            jobs = build_jobs_for_candidate(
                candidate=next_candidate,
                study_name=spec["study_name"],
                study_objects=study_objects,
                objects_root_relpath=spec["objects_root_relpath"],
                base_xml_relpath=spec["base_xml_relpath"],
                trainer_args=spec["trainer_args"],
                seed=int(spec["seed"]),
                eval_episodes=int(spec["eval_episodes"]),
                force=bool(spec["force"]),
            )
            queue.enqueue_candidate_jobs(next_candidate, jobs, source=source)
            status["enqueued_candidate"] = next_candidate.candidate_id
            status["enqueued_source"] = source
        else:
            status["enqueued_candidate"] = None

    export_reports(queue, study_root)
    status["summary"] = queue.summary()
    return status


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    study_root = resolve_study_root(args.study_name, repo_root=repo_root)
    db_path = os.path.abspath(args.db or str(study_root / "study.db"))

    cluster_cfg = load_cluster_config(args.cluster_config, repo_dirname=repo_root.name)
    study_objects = load_study_manifest(args.objects_root, expected_base_objects=args.expected_base_objects)

    queue = StudyQueue(db_path)
    try:
        spec = initialize_study(queue, args, cluster_cfg, study_objects, study_root, repo_root)
        if args.preview_initial_only:
            export_reports(queue, study_root)
            preview = []
            for candidate_id in spec.get("initial_candidate_ids", []):
                row = queue.conn.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
                if row is None:
                    continue
                preview.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "N": int(row["N"]),
                        "alpha": float(row["alpha"]),
                        "beta": float(row["beta"]),
                        "status": row["status"],
                    }
                )
            print(
                json.dumps(
                    {
                        "study_name": spec["study_name"],
                        "study_root": str(study_root),
                        "db_path": db_path,
                        "initial_candidates": preview,
                    },
                    indent=2,
                )
            )
            return
        if args.report_only:
            export_reports(queue, study_root)
            print(json.dumps(queue.summary(), indent=2))
            return

        while True:
            status = coordinator_iteration(queue, spec, study_objects, study_root)
            print(json.dumps(status, indent=2))

            completed_count = len(queue.completed_candidates())
            if completed_count >= int(spec["budget_total"]):
                break
            if args.once:
                break
            time.sleep(max(1, int(args.loop_sleep_seconds)))
    finally:
        queue.close()


if __name__ == "__main__":
    main()
