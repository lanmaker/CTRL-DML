"""
ACIC 2019 data challenge loader.

Fetches ZIP archives from the public Dropbox links and caches them locally.
Each CSV inside a ZIP is a single simulated dataset with columns:
    Y (outcome), A (binary treatment), V1...Vp (covariates).

Example usage:
    from data_acic2019 import load_acic
    X, T, Y = load_acic("low1")  # loads low_dimensional_datasets.zip and reads low1.csv
"""
from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests

ROOT = Path("data/acic2019")
ROOT.mkdir(parents=True, exist_ok=True)

URLS = {
    "low": "https://www.dropbox.com/s/g0elnbfmhbf7rr3/low_dimensional_datasets.zip?dl=1",
    "high": "https://www.dropbox.com/s/k2k1cs42i3pzkuu/high_dimensional_datasets.zip?dl=1",
    "test_low": "https://www.dropbox.com/s/qaj6fjbzorzmwpp/TestDatasets_lowD_Dec28.zip?dl=1",
}


def _download_zip(kind: str) -> Path:
    out = ROOT / f"{kind}.zip"
    if out.exists():
        return out
    url = URLS[kind]
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    return out


def list_datasets(kind: str = "low") -> list[str]:
    """List CSV names available in the chosen ZIP."""
    zpath = _download_zip(kind)
    with zipfile.ZipFile(zpath, "r") as zf:
        return sorted([n for n in zf.namelist() if n.endswith(".csv")])


def load_acic(dataset: str = "low1", kind: str = "low") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a specific ACIC 2019 dataset by filename without extension (e.g., 'low1').
    """
    zpath = _download_zip(kind)
    filename = f"{dataset}.csv" if not dataset.endswith(".csv") else dataset
    with zipfile.ZipFile(zpath, "r") as zf:
        if filename not in zf.namelist():
            raise ValueError(f"{filename} not found in {zpath.name}. Available: {list_datasets(kind)[:5]} ...")
        with zf.open(filename) as f:
            df = pd.read_csv(f)
    if not {"Y", "A"}.issubset(df.columns):
        raise ValueError(f"{filename} missing required columns Y/A.")
    y = df["Y"].to_numpy().astype(np.float32)
    t = df["A"].to_numpy().astype(np.float32)
    X = df.drop(columns=["Y", "A"]).to_numpy().astype(np.float32)
    return X, t, y


if __name__ == "__main__":
    print("Listing low-d datasets (first 5):", list_datasets("low")[:5])
    X, T, Y = load_acic("low1", kind="low")
    print(f"Loaded ACIC low1: X={X.shape}, T mean={T.mean():.3f}, Y mean={Y.mean():.3f}")
