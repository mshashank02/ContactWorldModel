# save as summarize_sobol.py
import argparse, json, os, numpy as np

def load_json(path):
    try:
        with open(path) as f: return json.load(f)
    except Exception:
        return None

def compute_scalar_from_metrics(m, theta=0.8):
    if "score" in m and not m.get("tasks"):
        return float(m["score"])
    tasks = m.get("tasks", [])
    if not tasks: return float("nan")
    checkpoints = m.get("checkpoints")
    finals, aulcs, speeds, disps = [], [], [], []
    for t in tasks:
        f = float(m["final_success"][t]); finals.append(f)
        if checkpoints is not None and "success" in m and t in m["success"]:
            curve = [float(v) for v in m["success"][t]]
            x = [float(u) for u in checkpoints]
            if len(curve) >= 2 and x[-1] > x[0]:
                aulc = float(np.trapz(curve, x=x) / (x[-1] - x[0]))
            else:
                aulc = float(np.mean(curve)) if curve else f
            try:
                idx = next(i for i, v in enumerate(curve) if v >= theta)
                speed = 1.0 - x[idx]
            except StopIteration:
                speed = 0.0
        else:
            aulc, speed = f, 0.0
        aulcs.append(aulc); speeds.append(speed)
        if "final_success_seeds" in m and t in m["final_success_seeds"]:
            vals = [float(v) for v in m["final_success_seeds"][t]]
            disp = float(np.std(vals)) if len(vals) >= 2 else 0.0
        else:
            disp = 0.0
        disps.append(disp)
    perf = np.mean(0.5*np.array(finals) + 0.3*np.array(aulcs) + 0.2*np.array(speeds))
    robustness_pen = 0.1 * np.mean(disps)
    return float(perf - robustness_pen)

ap = argparse.ArgumentParser()
ap.add_argument("--bo-root", default="generated/bo_runs_init")
ap.add_argument("--tasks", default="block,egg,pen")
ap.add_argument("--out", default="bo_summary_init.json")
args = ap.parse_args()
tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

X_hist, Y_hist = [], []

for tag in sorted(os.listdir(args.bo_root)):
    trial_dir = os.path.join(args.bo_root, tag)
    if not os.path.isdir(trial_dir): continue

    per_task = []
    for t in tasks:
        tdir = os.path.join(trial_dir, t)
        if not os.path.isdir(tdir): continue
        # average over all seed metrics in this task dir
        seed_vals = []
        for fn in os.listdir(tdir):
            if fn.startswith("metrics_") and fn.endswith(".json"):
                m = load_json(os.path.join(tdir, fn))
                if m is not None:
                    seed_vals.append(compute_scalar_from_metrics(m))
        if seed_vals:
            per_task.append(float(np.nanmean(seed_vals)))

    if per_task:
        Y = float(np.nanmean(per_task))  # equal weights across tasks
        Y_hist.append(Y)
        # recover (N,α,β) from tag "N{N}_a{α}_b{β}"
        # if you used a hash, parse meta.json instead
        try:
            parts = tag.split("_")
            N = int(parts[0][1:])
            a = float(parts[1][1:])
            b = float(parts[2][1:])
            X_hist.append([N, a, b])
        except Exception:
            # fallback to meta.json if present
            meta = load_json(os.path.join(trial_dir, "meta.json")) or {}
            X_hist.append([meta.get("N"), meta.get("alpha"), meta.get("beta")])

with open(args.out, "w") as f:
    json.dump({"X": X_hist, "Y": Y_hist, "best": (max(Y_hist) if Y_hist else None)}, f, indent=2)
print(f"Wrote {len(X_hist)} initial points to {args.out}")
