"""
TWINS dataset loader (CEVAE/DoWhy version).

Source CSVs (public):
- X: https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_X_3years_samesex.csv
- Y: https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv
- T: https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_T_3years_samesex.csv

Preprocessing (matches DoWhy notebook):
- Same-sex twins with both birth weights < 2000g (already filtered in the CEVAE CSVs).
- Expand each twin pair into two rows: the heavier twin gets T=1, the lighter T=0.
- Outcome Y: one-year mortality indicator for that twin.
- Covariates X: shared parental/pregnancy/birth covariates (46 dims).

Returns numpy arrays X, T, Y; no ground-truth CATE is available (observational).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests

ROOT = Path("data/twins")
ROOT.mkdir(parents=True, exist_ok=True)

URLS = {
    "X": "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_X_3years_samesex.csv",
    "Y": "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv",
    "T": "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_T_3years_samesex.csv",
}


def _download(name: str) -> Path:
    out = ROOT / f"{name}.csv"
    if out.exists():
        return out
    url = URLS[name]
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    return out


def load_twins(return_df: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and expand TWINS to individual-level rows."""
    paths = {k: _download(k) for k in URLS}
    X_pairs = pd.read_csv(paths["X"]).drop(columns=["Unnamed: 0"], errors="ignore")
    Y_pairs = pd.read_csv(paths["Y"]).drop(columns=["Unnamed: 0"], errors="ignore")
    T_pairs = pd.read_csv(paths["T"]).drop(columns=["Unnamed: 0"], errors="ignore")

    # Each row has two twins: columns 0/1 for lighter/heavier order unknown
    X_list, T_list, Y_list = [], [], []
    for idx in range(len(X_pairs)):
        w = T_pairs.iloc[idx].to_numpy(dtype=float)
        y = Y_pairs.iloc[idx].to_numpy(dtype=float)
        x = X_pairs.iloc[idx].to_numpy(dtype=float)
        if np.any(np.isnan(w)) or np.any(np.isnan(y)):
            continue
        # Filter out if either twin exceeds 2000g
        if np.any(w >= 2000):
            continue
        heavy_idx = int(np.argmax(w))
        light_idx = 1 - heavy_idx

        # Heavy twin (T=1)
        X_list.append(x)
        T_list.append(1.0)
        Y_list.append(float(y[heavy_idx]))

        # Light twin (T=0)
        X_list.append(x)
        T_list.append(0.0)
        Y_list.append(float(y[light_idx]))

    X = np.vstack(X_list).astype(np.float32)
    # Replace any residual NaNs in covariates with column means
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    T = np.array(T_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    if return_df:
        df = pd.DataFrame(X)
        df["T"] = T
        df["Y"] = Y
        return df
    return X, T, Y


if __name__ == "__main__":
    X, T, Y = load_twins()
    print(f"Loaded TWINS expanded: X={X.shape}, T mean={T.mean():.3f}, Y mean={Y.mean():.3f}")
