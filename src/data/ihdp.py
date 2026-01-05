"""
IHDP (Infant Health and Development Program) dataset loader.

The IHDP dataset is a semi-synthetic benchmark for causal inference.
Ground-truth potential outcomes are available.

Sources:
- CSV (1-10 replicates): CEVAE repository
- NPZ (1-100 replicates): Fredjo files
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import urllib.request
import os
import tempfile
import hashlib

# Column names for the IHDP CSV files
COLUMNS = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]

# CSV source (only 1-10)
CSV_URL = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_{}.csv"

# NPZ source (1-100 replicates)
NPZ_TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
NPZ_TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"
NPZ_TRAIN_SHA256 = "750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240"
NPZ_TEST_SHA256 = "a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c"

# Local data directory (relative to project root)
LOCAL_DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Cache directory (fallback to temp)
CACHE_DIR = Path(tempfile.gettempdir()) / "ihdp_cache"

# Global cache for NPZ data
_npz_cache = {"train": None, "test": None}


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


def _find_npz_files() -> Tuple[Optional[Path], Optional[Path]]:
    """Find NPZ files in local data dir or cache."""
    # Check local data directory first
    local_train = LOCAL_DATA_DIR / "ihdp_npci_1-100.train.npz"
    local_test = LOCAL_DATA_DIR / "ihdp_npci_1-100.test.npz"

    if local_train.exists() and local_test.exists():
        return local_train, local_test

    # Fallback to cache directory
    cache_train = CACHE_DIR / "ihdp_train.npz"
    cache_test = CACHE_DIR / "ihdp_test.npz"

    if cache_train.exists() and cache_test.exists():
        return cache_train, cache_test

    return None, None


def _download_npz():
    """Download and cache the NPZ files with 100 replicates."""
    global _npz_cache

    if _npz_cache["train"] is not None:
        return

    # Check for local files first
    train_path, test_path = _find_npz_files()

    if train_path is None or test_path is None:
        # Need to download
        CACHE_DIR.mkdir(exist_ok=True)
        train_path = CACHE_DIR / "ihdp_train.npz"
        test_path = CACHE_DIR / "ihdp_test.npz"

        if not train_path.exists():
            print("Downloading IHDP train data (100 replicates)...")
            urllib.request.urlretrieve(NPZ_TRAIN_URL, train_path)

        if not test_path.exists():
            print("Downloading IHDP test data (100 replicates)...")
            urllib.request.urlretrieve(NPZ_TEST_URL, test_path)

    _verify_checksum(train_path, NPZ_TRAIN_SHA256, "IHDP train NPZ")
    _verify_checksum(test_path, NPZ_TEST_SHA256, "IHDP test NPZ")

    # Load into cache
    _npz_cache["train"] = np.load(train_path)
    _npz_cache["test"] = np.load(test_path)


def load_ihdp_replicate(rep: int = 1) -> pd.DataFrame:
    """
    Load a single IHDP semi-synthetic replicate with ground-truth potential outcomes.

    Checks local data directory first, then downloads if necessary.

    Args:
        rep: Replicate number (1-100)

    Returns:
        DataFrame with columns: treatment, y_factual, y_cfactual, mu0, mu1, x1..x25, replicate
    """
    # Use NPZ (local first, then download if needed)
    _download_npz()

    # NPZ format: 'x', 't', 'yf', 'ycf', 'mu0', 'mu1' with shape (n_samples, n_replicates)
    train = _npz_cache["train"]
    test = _npz_cache["test"]

    rep_idx = rep - 1  # 0-indexed

    # Combine train and test
    X = np.vstack([train['x'][:, :, rep_idx], test['x'][:, :, rep_idx]])
    T = np.concatenate([train['t'][:, rep_idx], test['t'][:, rep_idx]])
    Y_f = np.concatenate([train['yf'][:, rep_idx], test['yf'][:, rep_idx]])
    Y_cf = np.concatenate([train['ycf'][:, rep_idx], test['ycf'][:, rep_idx]])
    mu0 = np.concatenate([train['mu0'][:, rep_idx], test['mu0'][:, rep_idx]])
    mu1 = np.concatenate([train['mu1'][:, rep_idx], test['mu1'][:, rep_idx]])

    # Build DataFrame
    data = {
        "treatment": T,
        "y_factual": Y_f,
        "y_cfactual": Y_cf,
        "mu0": mu0,
        "mu1": mu1,
    }
    for i in range(25):
        data[f"x{i+1}"] = X[:, i]
    data["replicate"] = rep

    return pd.DataFrame(data)


def load_ihdp_replicates(reps: List[int]) -> pd.DataFrame:
    """
    Load multiple IHDP replicates and concatenate.

    Args:
        reps: List of replicate numbers

    Returns:
        Concatenated DataFrame
    """
    dfs = [load_ihdp_replicate(r) for r in reps]
    return pd.concat(dfs, ignore_index=True)


def load_ihdp_arrays(
    rep: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load IHDP as numpy arrays.

    Args:
        rep: Replicate number (1-100)

    Returns:
        Tuple of (X, T, Y, true_cate)
            X: Covariates (n_samples, 25)
            T: Treatment indicator (n_samples,)
            Y: Observed outcome (n_samples,)
            true_cate: Ground-truth CATE = mu1 - mu0 (n_samples,)
    """
    df = load_ihdp_replicate(rep)

    # Extract covariates
    x_cols = [f"x{i}" for i in range(1, 26)]
    X = df[x_cols].values.astype(np.float32)

    # Treatment and outcome
    T = df["treatment"].values.astype(np.float32)
    Y = df["y_factual"].values.astype(np.float32)

    # Ground-truth CATE
    true_cate = (df["mu1"] - df["mu0"]).values.astype(np.float32)

    return X, T, Y, true_cate


def load_ihdp_train_test(
    rep: int = 1,
    train_frac: float = 0.8,
    seed: int = 42
) -> dict:
    """
    Load IHDP with the official train/test split from the NPZ release.

    Args:
        rep: Replicate number
        train_frac: Unused when using the official split (kept for backwards compatibility)
        seed: Random seed

    Returns:
        Dictionary with train/test arrays
    """
    _download_npz()
    train = _npz_cache["train"]
    test = _npz_cache["test"]

    rep_idx = rep - 1
    X_train = train["x"][:, :, rep_idx].astype(np.float32)
    T_train = train["t"][:, rep_idx].astype(np.float32)
    Y_train = train["yf"][:, rep_idx].astype(np.float32)
    cate_train = (train["mu1"][:, rep_idx] - train["mu0"][:, rep_idx]).astype(np.float32)

    X_test = test["x"][:, :, rep_idx].astype(np.float32)
    T_test = test["t"][:, rep_idx].astype(np.float32)
    Y_test = test["yf"][:, rep_idx].astype(np.float32)
    cate_test = (test["mu1"][:, rep_idx] - test["mu0"][:, rep_idx]).astype(np.float32)

    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(Y_train))
    test_idx = rng.permutation(len(Y_test))

    return {
        "X_train": X_train[train_idx],
        "T_train": T_train[train_idx],
        "Y_train": Y_train[train_idx],
        "cate_train": cate_train[train_idx],
        "X_test": X_test[test_idx],
        "T_test": T_test[test_idx],
        "Y_test": Y_test[test_idx],
        "cate_test": cate_test[test_idx],
    }


if __name__ == "__main__":
    # Test loading replicates 1 and 50
    for rep in [1, 50, 100]:
        try:
            X, T, Y, cate = load_ihdp_arrays(rep)
            print(f"IHDP replicate {rep}: X={X.shape}, T mean={T.mean():.3f}, Y mean={Y.mean():.3f}")
            print(f"  True CATE: mean={cate.mean():.3f}, std={cate.std():.3f}")
        except Exception as e:
            print(f"Rep {rep} failed: {e}")
