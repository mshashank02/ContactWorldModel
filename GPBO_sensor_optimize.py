#!/usr/bin/env python3
"""
GPU Bayesian Optimization (BoTorch qNEI) over (N, alpha, beta), aggregating across tasks.

- For each candidate (N, alpha, beta):
    * Loops over tasks (default: block, egg, pen)
    * For each task, loops over seeds
    * Runs your `generate_train.py` -> trainer (which writes --metrics-json or prints FINAL_SCORE)
    * Computes a per-task scalar
    * Returns a weighted average across tasks (equal weights by default)

- Objective per run = mean over seeds of (0.5*Final + 0.3*AULC + 0.2*Speed − 0.1*Dispersion if present)
  using the normalized checkpoints you log. AULC uses trapezoid (robust to non-uniform spacing).
"""

import argparse, os, sys, json, subprocess, re, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

# ------------------ small utils ------------------

def even_round(n: int) -> int:
    return int(round(n / 2) * 2)

def sha(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def map_alpha_beta_to_ratios(N: int, alpha: float, beta: float):
    """alpha: palm fraction; beta: fraction of non-palm going to fingertips"""
    N = int(N)
    Np = int(round(alpha * N))
    N_non = max(N - Np, 0)
    Nt = int(round(beta * N_non))
    Nx = max(N_non - Nt, 0)
    eps = 1e-6
    Rppx = float(Np) / float(max(Nx, eps))  # palm:phalanx
    Rpt  = float(Np) / float(max(Nt, eps))  # palm:tip
    return N, Np, Nt, Nx, Rppx, Rpt

# ------------------ metrics handling ------------------

_PATS = [
    r"FINAL_SCORE[:=]\s*([0-9]*\.?[0-9]+)",
    r"mean[_\s-]?success[^0-9]*([0-9]*\.?[0-9]+)",
    r"success[_\s-]*rate[^0-9]*([0-9]*\.?[0-9]+)",
    r"eval[/\s]*success[^0-9]*([0-9]*\.?[0-9]+)",
]

def parse_score_from_text(txt: str) -> Optional[float]:
    for pat in _PATS:
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None

def load_json(path: str) -> Optional[dict]:
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def compute_scalar_from_metrics(
    m: dict,
    theta: float = 0.8,
    w_final: float = 0.5,
    w_aulc: float = 0.3,
    w_speed: float = 0.2,
    w_disp: float = 0.1,
    penalty_lambda_time: float = 0.0,
    penalty_lambda_N: float = 0.0,
    N_ref: Optional[int] = None,
) -> float:
    """
    Expected JSON:
      {
        "tasks": ["block","egg","pen"] or ["block"],
        "checkpoints": [..fractions..],
        "success": {"block":[..K..], ...},
        "final_success": {"block": x, ...},
        # optional:
        "final_success_seeds": {"block":[..], ...},
        "step_time_ratio": 1.03,
        "N": 92
      }
    If only {"score": x} is present, returns x directly.
    """
    if "score" in m and not m.get("tasks"):
        return float(m["score"])

    tasks = m.get("tasks", [])
    if not tasks:
        raise ValueError("metrics.json missing 'tasks' or 'score'.")

    checkpoints = m.get("checkpoints", None)
    finals, aulcs, speeds, disps = [], [], [], []

    for t in tasks:
        f = float(m["final_success"][t]); finals.append(f)

        if checkpoints is not None and "success" in m and t in m["success"]:
            curve = [float(v) for v in m["success"][t]]
            x = [float(u) for u in checkpoints]  # fractions 0..1
            if len(curve) >= 2 and x[-1] > x[0]:
                aulc = float(np.trapz(curve, x=x) / (x[-1] - x[0]))
            else:
                aulc = float(np.mean(curve)) if curve else f
            # fraction of budget remaining when threshold first reached
            try:
                idx = next(i for i, v in enumerate(curve) if v >= theta)
                speed = 1.0 - x[idx]
            except StopIteration:
                speed = 0.0
        else:
            aulc = f
            speed = 0.0

        aulcs.append(aulc)
        speeds.append(speed)

        if "final_success_seeds" in m and t in m["final_success_seeds"]:
            vals = [float(v) for v in m["final_success_seeds"][t]]
            disp = float(np.std(vals)) if len(vals) >= 2 else 0.0
        else:
            disp = 0.0
        disps.append(disp)

    perf = np.mean(w_final*np.array(finals) + w_aulc*np.array(aulcs) + w_speed*np.array(speeds))
    robustness_pen = w_disp * np.mean(disps)

    pen_time = penalty_lambda_time * max(0.0, float(m.get("step_time_ratio", 1.0)) - 1.0) if penalty_lambda_time > 0.0 else 0.0
    pen_N = 0.0
    if penalty_lambda_N > 0.0 and N_ref and "N" in m:
        pen_N = penalty_lambda_N * max(0.0, (float(m["N"]) - float(N_ref)) / float(N_ref))

    return float(perf - robustness_pen - pen_time - pen_N)

# ------------------ one (task, seed) evaluation ------------------

def run_once_generate_train(
    generate_script: str,
    base: str, task: str, out_root: str,
    Ap: float, Apx: float, At: float, Ap1: float, Ap2: float,
    N: int, alpha: float, beta: float,
    force: bool,
    trainer_args: List[str],
    seed: Optional[int],
    trial_dir: str,
) -> float:
    """Build XML + launch trainer for ONE (task, seed). Returns scalar score."""
    N = even_round(int(N))
    Ntotal, Np, Nt, Nx, Rppx, Rpt = map_alpha_beta_to_ratios(N, alpha, beta)

    task_dir = ensure_dir(os.path.join(trial_dir, task))
    metrics_path = os.path.join(task_dir, f"metrics_{task}_seed{seed if seed is not None else 0}.json")

    cmd = [
        sys.executable, generate_script,
        "--base", base, "--task", task,
        "--Ntotal", str(Ntotal),
        "--Rppx", f"{Rppx:.6f}", "--Rpt", f"{Rpt:.6f}",
        "--Ap", str(Ap), "--Apx", str(Apx), "--At", str(At), "--Ap1", str(Ap1), "--Ap2", str(Ap2),
        "--out-root", out_root,
    ]
    if force:
        cmd.append("--force")
    cmd += ["--"]  # pass-through to trainer
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    # ensure trainer writes JSON with per-task label
    cmd += ["--metrics-json", metrics_path, "--task-name", task]
    cmd += trainer_args

    proc = subprocess.run(cmd, text=True, capture_output=True)

    # Prefer JSON → richer scalar; else parse stdout/stderr
    score = None
    m = load_json(metrics_path)
    if m is not None:
        score = compute_scalar_from_metrics(m)
    if score is None:
        score = parse_score_from_text(proc.stdout or "") or parse_score_from_text(proc.stderr or "")
    if score is None:
        score = float("nan")  # mark failure

    # Persist logs
    with open(os.path.join(task_dir, f"stdout_{task}_seed{seed if seed is not None else 0}.txt"), "w") as f:
        f.write(proc.stdout or "")
    with open(os.path.join(task_dir, f"stderr_{task}_seed{seed if seed is not None else 0}.txt"), "w") as f:
        f.write(proc.stderr or "")

    return float(score)

# ------------------ candidate evaluation across tasks ------------------

def eval_candidate(
    generate_script: str,
    base: str, tasks: List[str], task_weights: List[float],
    out_root: str,
    Ap: float, Apx: float, At: float, Ap1: float, Ap2: float,
    N: int, alpha: float, beta: float,
    force: bool,
    trainer_args: List[str],
    seeds: List[int],
    bo_root: str,
) -> Tuple[float, Dict]:
    """
    Evaluate (N, alpha, beta) across tasks and seeds.
    Returns weighted mean score + meta including per-task details.
    """
    tag = f"N{even_round(N)}_a{alpha:.4f}_b{beta:.4f}"
    trial_dir = ensure_dir(os.path.join(bo_root, f"{tag}-{sha(tag)}"))

    per_task_scores = {}
    per_task_seed_scores = {}
    for t in tasks:
        seed_scores = []
        for s in (seeds or [None]):
            seed_scores.append(run_once_generate_train(
                generate_script, base, t, out_root, Ap, Apx, At, Ap1, Ap2,
                N, alpha, beta, force, trainer_args, s, trial_dir
            ))
        per_task_seed_scores[t] = [float(x) for x in seed_scores]
        per_task_scores[t] = float(np.nanmean(seed_scores))

    # Weighted average across tasks (ignore NaN tasks by re-normalizing weights)
    w = np.array(task_weights, dtype=float)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    task_vals = np.array([per_task_scores[t] for t in tasks], dtype=float)

    valid = ~np.isnan(task_vals)
    if valid.any():
        agg = float(np.dot(w[valid], task_vals[valid]) / w[valid].sum())
    else:
        agg = float("nan")

    meta = {
        "N": even_round(N),
        "alpha": float(alpha),
        "beta": float(beta),
        "per_task_seed_scores": per_task_seed_scores,
        "per_task_scores": per_task_scores,
        "task_weights": {t: float(wi) for t, wi in zip(tasks, w.tolist())},
        "aggregated_score": agg,
        "trial_dir": trial_dir,
    }
    with open(os.path.join(trial_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return agg, meta

# ------------------ BoTorch qNEI loop ------------------

def round_N_in_tensor(X: torch.Tensor, nmin: int, nmax: int) -> torch.Tensor:
    Xr = X.clone()
    Ne = torch.round(Xr[..., 0] / 2.0) * 2.0
    Xr[..., 0] = torch.clamp(Ne, nmin, nmax)
    return Xr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate-script", default="generate_train.py")
    ap.add_argument("--base", required=True)

    # TASKS: list and weights
    ap.add_argument("--tasks", type=str, default="block,egg,pen",
                    help="comma-separated list of tasks to evaluate")
    ap.add_argument("--task-weights", type=str, default=None,
                    help="comma-separated weights aligned with --tasks (defaults to equal weights)")

    # layout generator weights (kept constant during BO)
    ap.add_argument("--Ap", type=float, required=True)
    ap.add_argument("--Apx", type=float, required=True)
    ap.add_argument("--At", type=float, required=True)
    ap.add_argument("--Ap1", type=float, required=True)
    ap.add_argument("--Ap2", type=float, required=True)

    ap.add_argument("--out-root", default="generated")
    ap.add_argument("--force", action="store_true")

    # search space
    ap.add_argument("--N-min", type=int, default=40)
    ap.add_argument("--N-max", type=int, default=140)
    ap.add_argument("--alpha-min", type=float, default=0.05)
    ap.add_argument("--alpha-max", type=float, default=0.95)
    ap.add_argument("--beta-min", type=float, default=0.05)
    ap.add_argument("--beta-max", type=float, default=0.95)

    # BO config
    ap.add_argument("--init", type=int, default=16, help="Sobol initial points")
    ap.add_argument("--iters", type=int, default=40, help="BO iterations")
    ap.add_argument("--q", type=int, default=8, help="batch size per BO step")
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--raw-samples", type=int, default=256)
    ap.add_argument("--bo-root", default="generated/bo_runs")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # seeds + pass-through trainer args
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--trainer-args", nargs=argparse.REMAINDER, default=[])

    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError("No tasks provided.")
    if args.task_weights is None:
        task_weights = [1.0] * len(tasks)
    else:
        task_weights = [float(x) for x in args.task_weights.split(",")]
        if len(task_weights) != len(tasks):
            raise ValueError("--task-weights must match the number of --tasks")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()!=""]

    # strip leading "--" if present (we add it ourselves in the subprocess)
    trainer_args = args.trainer_args[1:] if (args.trainer_args and args.trainer_args[0] == "--") else args.trainer_args

    torch.set_default_dtype(torch.double)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    bo_root = ensure_dir(args.bo_root)

    # bounds in ORIGINAL scale (we normalize inside the model)
    lb = torch.tensor([args.N_min, args.alpha_min, args.beta_min], device=device, dtype=torch.double)
    ub = torch.tensor([args.N_max, args.alpha_max, args.beta_max], device=device, dtype=torch.double)
    bounds = torch.stack([lb, ub])

    # ---------- Sobol init ----------
    X_hist, Y_hist = [], []
    X0 = draw_sobol_samples(bounds=bounds, n=1, q=args.init).squeeze(0)
    X0 = round_N_in_tensor(X0, args.N_min, args.N_max)

    for i in range(X0.shape[0]):
        Ni, ai, bi = int(X0[i,0].item()), float(X0[i,1].item()), float(X0[i,2].item())
        yi, meta = eval_candidate(
            generate_script=args.generate_script,
            base=args.base, tasks=tasks, task_weights=task_weights,
            out_root=args.out_root,
            Ap=args.Ap, Apx=args.Apx, At=args.At, Ap1=args.Ap1, Ap2=args.Ap2,
            N=Ni, alpha=ai, beta=bi,
            force=args.force,
            trainer_args=trainer_args,
            seeds=seeds,
            bo_root=bo_root,
        )
        X_hist.append([Ni, ai, bi]); Y_hist.append(yi)
        with open(os.path.join(bo_root, "bo_summary.json"), "w") as f:
            json.dump({"X": X_hist, "Y": Y_hist, "best": float(np.nanmax(Y_hist))}, f, indent=2)

    train_X = torch.tensor(np.array(X_hist), device=device, dtype=torch.double)
    train_Y = torch.tensor(np.array(Y_hist)[:, None], device=device, dtype=torch.double)

    # ---------- BO loop ----------
    for it in range(args.iters):
        model = SingleTaskGP(
            train_X, train_Y,
            input_transform=Normalize(d=3),
            outcome_transform=Standardize(m=1),
        ).to(device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acq = qNoisyExpectedImprovement(model=model, X_baseline=train_X, prune_baseline=True)

        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=args.q,
            num_restarts=args.restarts,
            raw_samples=args.raw_samples,
        )
        cand = round_N_in_tensor(cand, args.N_min, args.N_max)
        cand_np = cand.detach().cpu().numpy()

        # Parallel eval of q candidates
        new_X, new_Y = [], []
        with ProcessPoolExecutor(max_workers=args.q) as ex:
            futures = []
            for j in range(args.q):
                Ni, ai, bi = int(cand_np[j,0]), float(cand_np[j,1]), float(cand_np[j,2])
                futures.append(ex.submit(
                    eval_candidate,
                    args.generate_script, args.base, tasks, task_weights,
                    args.out_root,
                    args.Ap, args.Apx, args.At, args.Ap1, args.Ap2,
                    Ni, ai, bi,
                    args.force,
                    trainer_args,
                    seeds,
                    bo_root
                ))
            for fut in as_completed(futures):
                yi, meta = fut.result()
                new_X.append([meta["N"], meta["alpha"], meta["beta"]])
                new_Y.append(yi)

        train_X = torch.cat([train_X, torch.tensor(np.array(new_X), device=device, dtype=torch.double)], dim=0)
        train_Y = torch.cat([train_Y, torch.tensor(np.array(new_Y)[:, None], device=device, dtype=torch.double)], dim=0)

        best_idx = int(torch.nanargmax(train_Y).item())
        best = {
            "N": int(train_X[best_idx,0].item()),
            "alpha": float(train_X[best_idx,1].item()),
            "beta": float(train_X[best_idx,2].item()),
            "score": float(train_Y[best_idx,0].item()),
        }
        with open(os.path.join(bo_root, "bo_summary.json"), "w") as f:
            json.dump({
                "X": train_X.detach().cpu().tolist(),
                "Y": train_Y.detach().cpu().view(-1).tolist(),
                "best": best,
            }, f, indent=2)
        with open(os.path.join(bo_root, "bo_best.json"), "w") as f:
            json.dump(best, f, indent=2)

        print(f"[BO][{it+1}/{args.iters}] best so far: {best}")

    print("\n[BO] DONE.")

if __name__ == "__main__":
    main()
