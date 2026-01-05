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

IMPORTANT: Duplicate X Issue
-----------------------------
Each twin pair generates TWO rows with IDENTICAL X values (one for heavy twin T=1,
one for light twin T=0). This means:

1. Each covariate vector X appears exactly twice in the dataset
2. For each X, we observe BOTH Y(0) and Y(1) (the twin outcomes)
3. This provides a special form of ground truth where ITE = Y(1) - Y(0) per pair

This is BY DESIGN for the TWINS benchmark, as twins share parental/pregnancy
covariates. However, users should be aware:
- Standard overlap assumptions may not apply
- The model sees the same X with different treatments
- Evaluation should account for this paired structure

For methods sensitive to covariate overlap, consider using only one twin per pair
or adjusting the evaluation accordingly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union
import hashlib

import numpy as np
import pandas as pd
import requests

# Get project root
_THIS_DIR = Path(__file__).parent
ROOT = _THIS_DIR.parents[1] / "data" / "twins"
ROOT.mkdir(parents=True, exist_ok=True)

URLS = {
    "X": "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_X_3years_samesex.csv",
    "Y": "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv",
    "T": "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_T_3years_samesex.csv",
}
CHECKSUMS = {
    "X": "6ec6263e678f67205d584ebda083173c1ec44fb46ea7cfaafe3c87a31f612b86",
    "Y": "d03428d23ed308673f9151e3934aa7712df27c1fed244ed44ecc80a98a60836b",
    "T": "fbe6eb1cad27794c61cbbd8b11b72a26e96549e59d64fb88b07b30aaf2bae07d",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(path: Path, expected: str, label: str) -> None:
    if not expected:
        return
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} checksum mismatch: expected {expected}, got {actual}")


def _download(name: str) -> Path:
    """Download a dataset file if not already cached."""
    out = ROOT / f"{name}.csv"
    if out.exists():
        _verify_checksum(out, CHECKSUMS.get(name, ""), f"TWINS {name}.csv")
        return out
    url = URLS[name]
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    _verify_checksum(out, CHECKSUMS.get(name, ""), f"TWINS {name}.csv")
    return out


def load_twins(
    return_df: bool = False,
    max_weight: float = 2000.0
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]:
    """
    Load and expand TWINS to individual-level rows.

    Args:
        return_df: If True, return DataFrame instead of arrays
        max_weight: Maximum birth weight threshold

    Returns:
        If return_df=False: Tuple of (X, T, Y) numpy arrays
        If return_df=True: DataFrame with X, T, Y columns
    """
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

        # Filter out if either twin exceeds weight threshold
        if np.any(w >= max_weight):
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


def compute_twins_ite(Y: np.ndarray) -> np.ndarray:
    """
    Compute Individual Treatment Effects from TWINS paired structure.

    Since TWINS data has paired rows (T=1 followed by T=0 for each twin pair),
    we can compute ground truth ITE = Y(1) - Y(0).

    Args:
        Y: Outcome array from load_twins (length 2N for N twin pairs)

    Returns:
        ITE array of length N (one per twin pair)
    """
    n_pairs = len(Y) // 2
    Y_treated = Y[0::2]    # Heavy twins (T=1)
    Y_control = Y[1::2]    # Light twins (T=0)
    return Y_treated - Y_control


def compute_twins_pehe(tau_pred: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute PEHE using TWINS paired ground truth.

    Args:
        tau_pred: Predicted CATE for treated twins (length N for N pairs)
        Y: Outcome array from load_twins (length 2N)

    Returns:
        PEHE score (sqrt of MSE against paired ITE)
    """
    ite_true = compute_twins_ite(Y)
    return float(np.sqrt(np.mean((tau_pred - ite_true) ** 2)))


if __name__ == "__main__":
    X, T, Y = load_twins()
    print(f"Loaded TWINS expanded: X={X.shape}, T mean={T.mean():.3f}, Y mean={Y.mean():.3f}")
    ite = compute_twins_ite(Y)
    print(f"Computed ITE: mean={ite.mean():.4f}, std={ite.std():.4f}")
