# save as run_sobol_manual.py
import argparse, json, os, sys, subprocess, math

def map_alpha_beta_to_ratios(N, alpha, beta):
    N = int(N)
    Np = int(round(alpha * N))
    N_non = max(N - Np, 0)
    Nt = int(round(beta * N_non))
    Nx = max(N_non - Nt, 0)
    eps = 1e-6
    Rppx = float(Np) / float(max(Nx, eps))
    Rpt  = float(Np) / float(max(Nt, eps))
    return N, Np, Nt, Nx, Rppx, Rpt

def even_round(n: int) -> int:
    return int(round(n / 2) * 2)

ap = argparse.ArgumentParser()
ap.add_argument("--points", default="sobol_points.json")
ap.add_argument("--generate-script", default="generate_and_train.py")
ap.add_argument("--base", required=True)
ap.add_argument("--tasks", default="block,egg,pen")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--out-root", default="generated")
ap.add_argument("--bo-root", default="generated/bo_runs")
ap.add_argument("--Ap",   type=float, default= 6557, help="Area weight: Palm")
ap.add_argument("--Apx",  type=float, default=26885, help="Area weight: Phalanx")
ap.add_argument("--At",   type=float, default=7193, help="Area weight: Tips")
ap.add_argument("--Ap1",  type=float, default=5557, help="Palm sub-area 1 (palm)")
ap.add_argument("--Ap2",  type=float, default=1000, help="Palm sub-area 2 (lfmetacarpal)")
ap.add_argument("--force", action="store_true")
ap.add_argument("--eval-episodes", type=int, default=50)
ap.add_argument("--trainer-extra", nargs=argparse.REMAINDER, default=[])
ap.add_argument("--max-concurrency", type=int, default=2)
args = ap.parse_args()

with open(args.points) as f:
    pts = json.load(f)

tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
seeds = [int(s) for s in args.seeds.split(",") if s.strip()!=""]

os.makedirs(args.bo_root, exist_ok=True)

# simple concurrency limiter
from concurrent.futures import ThreadPoolExecutor, as_completed
def run_one(pt, task, seed):
    N, alpha, beta = even_round(pt["N"]), float(pt["alpha"]), float(pt["beta"])
    Ntotal, *_ , Rppx, Rpt = map_alpha_beta_to_ratios(N, alpha, beta)
    tag = f"N{N}_a{alpha:.4f}_b{beta:.4f}"
    trial_dir = os.path.join(args.bo_root, tag)  # you can add a hash if you like
    task_dir = os.path.join(trial_dir, task)
    os.makedirs(task_dir, exist_ok=True)
    metrics_path = os.path.abspath(os.path.join(task_dir, f"metrics_{task}_seed{seed}.json"))

    cmd = [
        sys.executable, args.generate_script,
        "--base", args.base,
        "--task", task,
        "--Ntotal", str(Ntotal),
        "--Rppx", f"{Rppx:.6f}",
        "--Rpt",  f"{Rpt:.6f}",
        "--Ap", str(args.Ap),
        "--Apx", str(args.Apx),
        "--At", str(args.At),
        "--Ap1", str(args.Ap1),
        "--Ap2", str(args.Ap2),
        "--out-root", args.out_root,
    ]
    if args.force:
        cmd.append("--force")
    # pass-through to trainer
    wandb_name = f"{task}-N{N}_a{alpha:.4f}_b{beta:.4f}-seed{seed}"
    cmd += ["--",
            "--seed", str(seed),
            "--metrics-json", metrics_path,
            "--task-name", task,
            "--eval-episodes", str(args.eval_episodes),
            "--wandb-name", wandb_name]
    if args.trainer_extra:
        cmd += args.trainer_extra

    # log files
    stdout_path = os.path.join(task_dir, f"stdout_{task}_seed{seed}.txt")
    stderr_path = os.path.join(task_dir, f"stderr_{task}_seed{seed}.txt")

    proc = subprocess.run(cmd, text=True, capture_output=True)
    with open(stdout_path, "w") as f: f.write(proc.stdout or "")
    with open(stderr_path, "w") as f: f.write(proc.stderr or "")
    return (pt, task, seed, metrics_path)

jobs = []
with ThreadPoolExecutor(max_workers=max(1, args.max_concurrency)) as ex:
    for pt in pts:
        for task in tasks:
            for seed in seeds:
                jobs.append(ex.submit(run_one, pt, task, seed))

    # wait + collect (optional: check metrics exist)
    for fut in as_completed(jobs):
        pt, task, seed, mpath = fut.result()
        print(f"Done: N={pt['N']} a={pt['alpha']:.3f} b={pt['beta']:.3f} | {task} seed{seed} → {mpath}")
