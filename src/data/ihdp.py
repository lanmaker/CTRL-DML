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

# Column names for the IHDP CSV files
COLUMNS = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]

# CSV source (only 1-10)
CSV_URL = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_{}.csv"

# NPZ source (1-100 replicates)
NPZ_TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
NPZ_TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"

# Cache directory
CACHE_DIR = Path(tempfile.gettempdir()) / "ihdp_cache"

# Global cache for NPZ data
_npz_cache = {"train": None, "test": None}


def _download_npz():
    """Download and cache the NPZ files with 100 replicates."""
    global _npz_cache

    if _npz_cache["train"] is not None:
        return

    CACHE_DIR.mkdir(exist_ok=True)

    train_path = CACHE_DIR / "ihdp_train.npz"
    test_path = CACHE_DIR / "ihdp_test.npz"

    # Download if not cached
    if not train_path.exists():
        print("Downloading IHDP train data (100 replicates)...")
        urllib.request.urlretrieve(NPZ_TRAIN_URL, train_path)

    if not test_path.exists():
        print("Downloading IHDP test data (100 replicates)...")
        urllib.request.urlretrieve(NPZ_TEST_URL, test_path)

    # Load into cache
    _npz_cache["train"] = np.load(train_path)
    _npz_cache["test"] = np.load(test_path)


def load_ihdp_replicate(rep: int = 1) -> pd.DataFrame:
    """
    Load a single IHDP semi-synthetic replicate with ground-truth potential outcomes.

    Args:
        rep: Replicate number (1-100)

    Returns:
        DataFrame with columns: treatment, y_factual, y_cfactual, mu0, mu1, x1..x25, replicate
    """
    if rep <= 10:
        # Try CSV first for backwards compatibility
        try:
            df = pd.read_csv(CSV_URL.format(rep), header=None)
            df.columns = COLUMNS
            df["replicate"] = rep
            return df
        except Exception:
            pass

    # Use NPZ for rep > 10 or if CSV fails
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
    Load IHDP with train/test split.

    Args:
        rep: Replicate number
        train_frac: Fraction for training
        seed: Random seed

    Returns:
        Dictionary with train/test arrays
    """
    X, T, Y, true_cate = load_ihdp_arrays(rep)

    rng = np.random.default_rng(seed)
    n = len(Y)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)

    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    return {
        "X_train": X[train_idx],
        "T_train": T[train_idx],
        "Y_train": Y[train_idx],
        "cate_train": true_cate[train_idx],
        "X_test": X[test_idx],
        "T_test": T[test_idx],
        "Y_test": Y[test_idx],
        "cate_test": true_cate[test_idx],
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
