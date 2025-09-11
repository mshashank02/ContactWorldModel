# save as summarize_sobol.py
import argparse, json, os, numpy as np

def load_json(path):
    try:
        with open(path) as f: 
            return json.load(f)
    except Exception:
        return None

def compute_scalar_from_metrics(m):
    tasks = m.get("tasks", [])
    if not tasks: 
        return float("nan")

    finals, aulcs, speeds, disps = [], [], [], []

    for t in tasks:
        # -------------------
        # 1️⃣ Best success
        # -------------------
        if "success" in m and t in m["success"]:
            curve = [float(v) for v in m["success"][t]]
            best_success = max(curve) if curve else float(m["final_success"].get(t, 0.0))
        else:
            curve = []
            best_success = float(m["final_success"].get(t, 0.0))
        finals.append(best_success)

        # -------------------
        # 2️⃣ Speed to best
        # -------------------
        checkpoints = m.get("checkpoints")
        if checkpoints is not None and curve:
            x = [float(u) for u in checkpoints]
            try:
                idx = curve.index(best_success)  # first occurrence of best success
                speed = 1.0 - x[idx]  # larger value = faster
            except ValueError:
                speed = 0.0
        else:
            speed = 0.0

        # -------------------
        # 3️⃣ Area under curve
        # -------------------
        if checkpoints is not None and curve and len(curve) >= 2:
            aulc = float(np.trapz(curve, x=x) / (x[-1] - x[0]))
        else:
            aulc = best_success
        aulcs.append(aulc)

        # -------------------
        # 4️⃣ Robustness: variance of success curve
        # -------------------
        if curve:
            disp = float(np.std(curve))  # penalize fluctuating curves
        else:
            disp = 0.0
        disps.append(disp)

        speeds.append(speed)

    perf = np.mean(0.5*np.array(finals) + 0.3*np.array(aulcs) + 0.2*np.array(speeds))
    robustness_pen = 0.1 * np.mean(disps)
    return float(perf - robustness_pen)

# -------------------
# Parse arguments
# -------------------
ap = argparse.ArgumentParser()
ap.add_argument("--bo-root", default="generated/bo_runs_init")
ap.add_argument("--tasks", default="block,egg,pen")
ap.add_argument("--out", default="bo_summary_init.json")
args = ap.parse_args()
tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

X_hist, Y_hist = [], []

# -------------------
# Iterate over trials
# -------------------
for tag in sorted(os.listdir(args.bo_root)):
    trial_dir = os.path.join(args.bo_root, tag)
    if not os.path.isdir(trial_dir): continue

    # Check all tasks exist and have at least one metrics file
    valid_trial = True
    per_task = []

    for t in tasks:
        tdir = os.path.join(trial_dir, t)
        if not os.path.isdir(tdir):
            valid_trial = False
            break
        metrics_files = [fn for fn in os.listdir(tdir) if fn.startswith("metrics_") and fn.endswith(".json")]
        if not metrics_files:
            valid_trial = False
            break

        # Compute average score for this task
        seed_vals = []
        for fn in metrics_files:
            m = load_json(os.path.join(tdir, fn))
            if m is not None:
                seed_vals.append(compute_scalar_from_metrics(m))
        if seed_vals:
            per_task.append(float(np.nanmean(seed_vals)))
        else:
            valid_trial = False
            break

    if not valid_trial:
        continue  # skip entire trial

    # -------------------
    # Aggregate across tasks
    # -------------------
    Y = float(np.nanmean(per_task))  # equal weights across tasks
    Y_hist.append(Y)

    # Recover (N, α, β) from tag "N{N}_a{α}_b{β}"
    try:
        parts = tag.split("_")
        N = int(parts[0][1:])
        a = float(parts[1][1:])
        b = float(parts[2][1:])
        X_hist.append([N, a, b])
    except Exception:
        meta = load_json(os.path.join(trial_dir, "meta.json")) or {}
        X_hist.append([meta.get("N"), meta.get("alpha"), meta.get("beta")])

# -------------------
# Save summary
# -------------------
with open(args.out, "w") as f:
    json.dump({"X": X_hist, "Y": Y_hist, "best": (max(Y_hist) if Y_hist else None)}, f, indent=2)

print(f"Wrote {len(X_hist)} valid trials to {args.out}")
