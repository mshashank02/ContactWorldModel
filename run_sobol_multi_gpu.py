# save as run_sobol_manual.py
import argparse, json, os, sys, subprocess, math, threading, queue
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="sobol_points.json")
    ap.add_argument("--generate-script", default="generate_and_train.py")
    ap.add_argument("--base", required=True)
    ap.add_argument("--tasks", default="block,egg,pen")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out-root", default="generated")
    ap.add_argument("--bo-root", default="generated/bo_runs")

    # area weights (now with defaults so you don't have to pass them every time)
    ap.add_argument("--Ap",   type=float, default=6557,  help="Area weight: Palm")
    ap.add_argument("--Apx",  type=float, default=26885, help="Area weight: Phalanx")
    ap.add_argument("--At",   type=float, default=7193,  help="Area weight: Tips")
    ap.add_argument("--Ap1",  type=float, default=5557,  help="Palm sub-area 1 (palm)")
    ap.add_argument("--Ap2",  type=float, default=1000,  help="Palm sub-area 2 (lfmetacarpal)")

    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--trainer-extra", nargs=argparse.REMAINDER, default=[])

    ap.add_argument("--max-concurrency", type=int, default=2)

    # GPU distribution & logging controls
    ap.add_argument("--gpu-ids", type=str, default=None,
                    help="Comma-separated physical GPU ids to use (e.g. '0,1,2,3'). "
                         "If omitted, uses CUDA_VISIBLE_DEVICES from the shell if set; "
                         "otherwise no pinning (single GPU or CPU).")
    ap.add_argument("--stream-logs", action="store_true",
                    help="Stream child stdout/stderr live to terminal (still tee to files).")
    ap.add_argument("--print-cmd", action="store_true",
                    help="Print the full trainer command for each job.")
    return ap.parse_args()

def build_gpu_pool(args):
    if args.gpu_ids:
        gpu_list = [g.strip() for g in args.gpu_ids.split(",") if g.strip() != ""]
    else:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd:
            gpu_list = [g.strip() for g in cvd.split(",") if g.strip() != ""]
        else:
            gpu_list = []
    q = queue.Queue()
    for g in gpu_list:
        q.put(g)
    return gpu_list, q

def tee_stream(pipe, sink_file, to_stderr=False):
    for line in iter(pipe.readline, ''):
        if to_stderr:
            print(line, end='', file=sys.stderr)
        else:
            print(line, end='')
        sink_file.write(line)
        sink_file.flush()
    pipe.close()

def run_one(pt, task, seed, args, gpu_list, gpu_pool):
    N, alpha, beta = even_round(pt["N"]), float(pt["alpha"]), float(pt["beta"])
    Ntotal, *_ , Rppx, Rpt = map_alpha_beta_to_ratios(N, alpha, beta)
    tag = f"N{N}_a{alpha:.4f}_b{beta:.4f}"

    trial_dir = os.path.join(args.bo_root, tag)
    task_dir  = os.path.join(trial_dir, task)
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

    # pass-through to trainer (runner adds the `--` separator for you)
    wandb_name = f"{task}-N{N}_a{alpha:.4f}_b{beta:.4f}-seed{seed}"
    cmd += ["--",
            "--seed", str(seed),
            "--metrics-json", metrics_path,
            "--task-name", task,
            "--eval-episodes", str(args.eval_episodes),
            "--wandb-name", wandb_name]
    if args.trainer_extra:
        cmd += args.trainer_extra

    stdout_path = os.path.join(task_dir, f"stdout_{task}_seed{seed}.txt")
    stderr_path = os.path.join(task_dir, f"stderr_{task}_seed{seed}.txt")

    # --- Environment for child process ---
    env = os.environ.copy()
    # Headless MuJoCo + friends
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    # Mirror console to W&B as well
    env.setdefault("WANDB_CONSOLE", "wrap")

    # GPU pinning: each job grabs one GPU (if provided) and returns it when done
    assigned_gpu = None
    if gpu_list:
        assigned_gpu = gpu_pool.get()  # blocks until a GPU id is available
        env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
        # After restricting visibility to a single physical GPU, that GPU is logical "0"
        env["MUJOCO_EGL_DEVICE_ID"] = "0"

    if args.print_cmd:
        print("CMD:", " ".join(cmd))

    try:
        if args.stream_logs:
            # Live streaming (tee) + files
            cmd0 = cmd
            # force unbuffered python for child if calling via python
            if cmd0[0].endswith("python") or cmd0[0].endswith("python3") or cmd0[0] == sys.executable:
                cmd0 = [cmd0[0], "-u"] + cmd0[1:]
                env["PYTHONUNBUFFERED"] = "1"

            with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
                proc = subprocess.Popen(cmd0, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1, env=env)
                t_out = threading.Thread(target=tee_stream, args=(proc.stdout, out_f, False))
                t_err = threading.Thread(target=tee_stream, args=(proc.stderr, err_f, True))
                t_out.start(); t_err.start()
                rc = proc.wait()
                t_out.join(); t_err.join()
        else:
            # Capture, then write to files afterwards
            proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
            rc = proc.returncode
            with open(stdout_path, "w") as f: f.write(proc.stdout or "")
            with open(stderr_path, "w") as f: f.write(proc.stderr or "")

        # Status tagging
        if rc != 0:
            status = f"FAIL(rc={rc})"
        elif not os.path.isfile(metrics_path) or os.path.getsize(metrics_path) == 0:
            status = "NO_METRICS"
        else:
            status = "OK"

        return (pt, task, seed, metrics_path, status, stdout_path, stderr_path)
    finally:
        if assigned_gpu is not None:
            gpu_pool.put(assigned_gpu)

def main():
    args = parse_args()

    with open(args.points) as f:
        pts = json.load(f)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    os.makedirs(args.bo_root, exist_ok=True)

    gpu_list, gpu_pool = build_gpu_pool(args)
    if args.gpu_ids and args.max_concurrency > len(gpu_list):
        print(f"Warning: --max-concurrency ({args.max_concurrency}) > number of GPUs provided ({len(gpu_list)}). "
              f"Multiple jobs may share a GPU.", file=sys.stderr)

    jobs = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_concurrency)) as ex:
        for pt in pts:
            for task in tasks:
                for seed in seeds:
                    jobs.append(ex.submit(run_one, pt, task, seed, args, gpu_list, gpu_pool))

        for fut in as_completed(jobs):
            pt, task, seed, mpath, status, outp, errp = fut.result()
            print(f"{status}: N={pt['N']} a={pt['alpha']:.3f} b={pt['beta']:.3f} | {task} seed{seed} → {mpath}")
            if status != "OK":
                print(f"  ↳ Logs: stdout={outp}  stderr={errp}")

if __name__ == "__main__":
    main()
