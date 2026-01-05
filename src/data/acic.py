"""
ACIC 2019 data challenge loader.

Fetches ZIP archives from the public Dropbox links and caches them locally.
Each CSV inside a ZIP is a single simulated dataset with columns:
    Y (outcome), A (binary treatment), V1...Vp (covariates).

Example usage:
    from src.data.acic import load_acic
    X, T, Y = load_acic("low1")  # loads low_dimensional_datasets.zip and reads low1.csv
"""
from __future__ import annotations

import warnings
import zipfile
from pathlib import Path
from typing import Tuple, List
import hashlib

import numpy as np
import pandas as pd
import requests

# Get project root
_THIS_DIR = Path(__file__).parent
ROOT = _THIS_DIR.parents[1] / "data" / "acic2019"
ROOT.mkdir(parents=True, exist_ok=True)

URLS = {
    "low": "https://www.dropbox.com/s/g0elnbfmhbf7rr3/low_dimensional_datasets.zip?dl=1",
    "high": "https://www.dropbox.com/s/k2k1cs42i3pzkuu/high_dimensional_datasets.zip?dl=1",
    "test_low": "https://www.dropbox.com/s/qaj6fjbzorzmwpp/TestDatasets_lowD_Dec28.zip?dl=1",
}
# SHA256 checksums for integrity verification.
# To compute: python -c "from src.data.acic import _sha256; print(_sha256(Path('data/acic2019/high.zip')))"
ZIP_SHA256 = {
    "low": "c728f9b9fd23fc5f2ee7bcdbb23b84ae49c59106d3466c3c0955e9ef3bcbaba4",
    "high": "",  # TODO: compute after first download
    "test_low": "",  # TODO: compute after first download
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(path: Path, expected: str, label: str) -> None:
    if not expected:
        warnings.warn(
            f"{label}: no checksum available - integrity not verified. "
            f"Computed SHA256: {_sha256(path)}",
            UserWarning,
            stacklevel=3,
        )
        return
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} checksum mismatch: expected {expected}, got {actual}")


def _download_zip(kind: str) -> Path:
    """Download and cache a ZIP archive."""
    out = ROOT / f"{kind}.zip"
    if out.exists():
        _verify_checksum(out, ZIP_SHA256.get(kind, ""), f"ACIC {kind} zip")
        return out
    url = URLS[kind]
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    _verify_checksum(out, ZIP_SHA256.get(kind, ""), f"ACIC {kind} zip")
    return out


def list_acic_datasets(kind: str = "low") -> List[str]:
    """
    List CSV names available in the chosen ZIP.

    Args:
        kind: One of "low", "high", "test_low"

    Returns:
        Sorted list of CSV filenames
    """
    zpath = _download_zip(kind)
    with zipfile.ZipFile(zpath, "r") as zf:
        return sorted([n for n in zf.namelist() if n.endswith(".csv")])


def load_acic(
    dataset: str = "low1",
    kind: str = "low"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a specific ACIC 2019 dataset by filename without extension (e.g., 'low1').

    Args:
        dataset: Dataset name (e.g., "low1", "high3")
        kind: ZIP archive kind ("low", "high", "test_low")

    Returns:
        Tuple of (X, T, Y) numpy arrays
    """
    zpath = _download_zip(kind)
    filename = f"{dataset}.csv" if not dataset.endswith(".csv") else dataset

    with zipfile.ZipFile(zpath, "r") as zf:
        if filename not in zf.namelist():
            available = list_acic_datasets(kind)[:5]
            raise ValueError(f"{filename} not found in {zpath.name}. Available: {available} ...")
        with zf.open(filename) as f:
            df = pd.read_csv(f)

    if not {"Y", "A"}.issubset(df.columns):
        raise ValueError(f"{filename} missing required columns Y/A.")

    y = df["Y"].to_numpy().astype(np.float32)
    t = df["A"].to_numpy().astype(np.float32)
    X = df.drop(columns=["Y", "A"]).to_numpy().astype(np.float32)

    return X, t, y


def compute_and_update_checksums() -> None:
    """
    Download all ACIC datasets and print checksums for updating ZIP_SHA256.

    Usage:
        python -m src.data.acic --compute-checksums

    After running, copy the printed checksums into ZIP_SHA256 dict above.
    """
    print("Computing checksums for ACIC datasets...")
    print("(This may take several minutes for large files)\n")

    for kind in ["low", "high", "test_low"]:
        print(f"Processing {kind}...")
        try:
            path = _download_zip(kind)
            checksum = _sha256(path)
            current = ZIP_SHA256.get(kind, "")
            status = "✓ matches" if current == checksum else ("⚠ MISSING" if not current else "✗ MISMATCH")
            print(f'  "{kind}": "{checksum}",  # {status}')
        except Exception as e:
            print(f'  "{kind}": "",  # Error: {e}')

    print("\nCopy the checksums above into ZIP_SHA256 in src/data/acic.py")


if __name__ == "__main__":
    import sys

    if "--compute-checksums" in sys.argv:
        compute_and_update_checksums()
    else:
        print("Listing low-d datasets (first 5):", list_acic_datasets("low")[:5])
        X, T, Y = load_acic("low1", kind="low")
        print(f"Loaded ACIC low1: X={X.shape}, T mean={T.mean():.3f}, Y mean={Y.mean():.3f}")
        print("\nTo compute checksums for all datasets, run:")
        print("  python -m src.data.acic --compute-checksums")
