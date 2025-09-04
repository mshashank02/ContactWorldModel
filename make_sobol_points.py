# save as make_sobol_points.py
import argparse, json
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--N-min", type=int, default=40)
ap.add_argument("--N-max", type=int, default=180)
ap.add_argument("--alpha-min", type=float, default=0.05)
ap.add_argument("--alpha-max", type=float, default=0.95)
ap.add_argument("--beta-min", type=float, default=0.05)
ap.add_argument("--beta-max", type=float, default=0.95)
ap.add_argument("--out", type=str, default="sobol_points.json")
args = ap.parse_args()

torch.set_default_dtype(torch.double)
lb = torch.tensor([args.N_min, args.alpha_min, args.beta_min], dtype=torch.double)
ub = torch.tensor([args.N_max, args.alpha_max, args.beta_max], dtype=torch.double)
bounds = torch.stack([lb, ub])
X0 = torch.quasirandom.SobolEngine(3).draw(args.k).double()
X = bounds[0] + X0 * (bounds[1] - bounds[0])

# round N to even
N = torch.round(X[:,0] / 2.0) * 2.0
X[:,0] = torch.clamp(N, args.N_min, args.N_max)

pts = [{"N": int(X[i,0].item()),
        "alpha": float(X[i,1].item()),
        "beta":  float(X[i,2].item())} for i in range(X.shape[0])]

with open(args.out, "w") as f:
    json.dump(pts, f, indent=2)
print(f"Wrote {len(pts)} points to {args.out}")
