#!/usr/bin/env python3
"""
suggest_next_candidates.py

Usage:
  python suggest_next_candidates.py --input bo_summary.json --q 8 --device cpu --nmin 40 --nmax 140 --restarts 10 --raw-samples 256

Assumes your JSON has fields:
  { "X": [[N, alpha, beta], ...], "Y": [y1, y2, ...] }
If your file has a different shape, adapt the load section.
"""

import argparse, json
import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement as qLogNEI
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

def round_N_in_tensor(X: torch.Tensor, nmin: int, nmax: int) -> torch.Tensor:
    Xr = X.clone()
    Ne = torch.round(Xr[..., 0] / 2.0) * 2.0
    Xr[..., 0] = torch.clamp(Ne, nmin, nmax)
    return Xr

def even_round_int(n: int) -> int:
    return int(round(n/2.0)*2)

def load_history(path):
    j = json.load(open(path, "r"))
    X = np.array(j.get("X", []), dtype=float)
    Y = np.array(j.get("Y", []), dtype=float)
    # Some bo_summary.json store Y as a single "best" and "X" only (unlikely). Validate:
    if X.size == 0:
        raise ValueError("No X found in JSON.")
    if Y.size == 0:
        # Try to read Y inside nested structure (compatibility)
        if "Y" in j and isinstance(j["Y"], list):
            Y = np.array(j["Y"], dtype=float)
        else:
            raise ValueError("No Y found in JSON.")
    return X, Y

def suggest_next(input_json, q=8, device="cpu", nmin=40, nmax=140, restarts=10, raw_samples=256):
    device = torch.device(device)
    X_np, Y_np = load_history(input_json)

    # enforce shapes
    X = torch.tensor(X_np, dtype=torch.double, device=device)
    Y = torch.tensor(Y_np, dtype=torch.double, device=device).view(-1,1)

    # remove NaNs from training set
    mask = ~torch.isnan(Y.view(-1))
    X = X[mask]
    Y = Y[mask].view(-1,1)

    if X.shape[0] < 2:
        raise RuntimeError("Need at least 2 non-NaN training points to fit a GP.")

    # build and fit GP (same transforms as your script)
    torch.set_default_dtype(torch.double)
    model = SingleTaskGP(train_X=X, train_Y=Y, input_transform=Normalize(d=3), outcome_transform=Standardize(m=1)).to(device)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    lb = torch.tensor([nmin, 0.05, 0.05], dtype=torch.double, device=device)
    ub = torch.tensor([nmax, 0.95, 0.95], dtype=torch.double, device=device)
    bounds = torch.stack([lb, ub])

    acq = qLogNEI(model=model, X_baseline=X, prune_baseline=True)

    # optimize acquisition for q points
    cand, acq_val = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=q,
        num_restarts=restarts,
        raw_samples=raw_samples,
    )
    cand = round_N_in_tensor(cand, nmin, nmax)
    cand_np = cand.detach().cpu().numpy()

    suggestions = []
    for i in range(cand_np.shape[0]):
        Ni = even_round_int(int(round(cand_np[i,0])))
        ai = float(cand_np[i,1])
        bi = float(cand_np[i,2])
        suggestions.append({"N": Ni, "alpha": ai, "beta": bi})

    return suggestions

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to bo_summary.json (or your sobol results JSON)")
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--nmin", type=int, default=40)
    ap.add_argument("--nmax", type=int, default=200)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--raw-samples", type=int, default=256)
    args = ap.parse_args()

    sug = suggest_next(args.input, q=args.q, device=args.device, nmin=args.nmin, nmax=args.nmax, restarts=args.restarts, raw_samples=args.raw_samples)
    print(json.dumps({"suggestions": sug}, indent=2))
