"""
Rosenbaum bounds sensitivity analysis on real datasets (Lalonde, STAR, Yelp if available).
Computes Wilcoxon signed-rank upper/lower p-value bounds over a grid of Gamma values.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from run_realdata_case_study import load_lalonde, load_star


def rosenbaum_bounds(df: pd.DataFrame, gamma: float) -> dict:
    """
    Compute Rosenbaum bounds (Wilcoxon signed-rank) for matched pairs.

    df must contain columns: pair_id, T (0/1), Y.
    """
    wide = df.pivot(index="pair_id", columns="T", values="Y").dropna()
    diff = (wide[1] - wide[0]).to_numpy()
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return {"Gamma": gamma, "Upper_p": np.nan, "Lower_p": np.nan}
    ranks = np.argsort(np.argsort(np.abs(diff))) + 1
    lam = gamma / (1 + gamma)
    W = ranks[diff > 0].sum()
    mu = lam * n * (n + 1) / 2
    var = lam * (1 - lam) * n * (n + 1) * (2 * n + 1) / 6
    z = (W - mu) / np.sqrt(var + 1e-8)
    upper_p = norm.sf(z)
    lower_p = norm.cdf(z)
    return {"Gamma": gamma, "Upper_p": upper_p, "Lower_p": lower_p}


def match_pairs(X: np.ndarray, y: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    """Simple nearest-neighbor matching on Euclidean distance."""
    treated_idx = np.where(t == 1)[0]
    control_idx = np.where(t == 0)[0]
    treated = X[treated_idx]
    control = X[control_idx]
    matched_rows = []
    for i, xi in zip(treated_idx, treated):
        dists = np.linalg.norm(control - xi, axis=1)
        j_rel = int(np.argmin(dists))
        j = control_idx[j_rel]
        matched_rows.append({"pair_id": len(matched_rows), "T": 1, "Y": y[i]})
        matched_rows.append({"pair_id": len(matched_rows) - 1, "T": 0, "Y": y[j]})
        control_idx = np.delete(control_idx, j_rel)
        control = np.delete(control, j_rel, axis=0)
        if len(control_idx) == 0:
            break
    return pd.DataFrame(matched_rows)


def run_dataset(name: str, gamma_grid: list[float]) -> pd.DataFrame:
    if name.lower() == "lalonde":
        X, y, t, _ = load_lalonde()
    elif name.lower() == "star":
        X, y, t, _ = load_star()
    elif name.lower() == "yelp":
        ate_path = Path("realdata_yelp_smd.csv")
        if not ate_path.exists():
            raise FileNotFoundError("realdata_yelp_smd.csv not found; run run_real_multimodal_yelp.py first.")
        # Use tabular features from the balance file (no raw covariates stored), so skip Yelp if absent.
        raise NotImplementedError("Yelp matching requires raw covariates; rerun with saved X if needed.")
    else:
        raise ValueError(f"Unknown dataset {name}")

    pairs = match_pairs(X, y, t)
    rows = []
    for g in gamma_grid:
        rows.append(rosenbaum_bounds(pairs, g))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Rosenbaum bounds sensitivity analysis.")
    parser.add_argument("--dataset", type=str, choices=["lalonde", "star"], default="lalonde")
    parser.add_argument("--gamma-max", type=float, default=2.0)
    parser.add_argument("--gamma-step", type=float, default=0.1)
    args = parser.parse_args()

    grid = [1.0 + i * args.gamma_step for i in range(int((args.gamma_max - 1.0) / args.gamma_step) + 1)]
    df = run_dataset(args.dataset, grid)
    out_path = Path(f"rosenbaum_{args.dataset}.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
