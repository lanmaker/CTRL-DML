import argparse
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import os

from run_ablation import get_stress_data, stabilize_residuals

# Ensure libomp is discoverable on macOS Homebrew installs
_libomp_path = "/usr/local/opt/libomp/lib"
if os.path.isdir(_libomp_path):
    current = os.environ.get("DYLD_LIBRARY_PATH", "")
    if _libomp_path not in current:
        os.environ["DYLD_LIBRARY_PATH"] = f"{_libomp_path}:{current}"

def set_seed(seed: int):
    np.random.seed(seed)


def cross_fit_nuisance(
    X: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    n_splits: int,
    seed: int,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    m_hat = np.zeros_like(y, dtype=np.float32)
    e_hat = np.zeros_like(y, dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        set_seed(seed + fold)
        # Outcome model
        m_model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed + fold,
            n_jobs=4,
        )
        m_model.fit(X[train_idx], y[train_idx])
        m_hat[val_idx] = m_model.predict(X[val_idx])

        # Propensity model
        e_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=seed + fold,
            n_jobs=4,
        )
        e_model.fit(X[train_idx], T[train_idx])
        e_hat[val_idx] = e_model.predict_proba(X[val_idx])[:, 1]
    return m_hat, e_hat


def train_tau_xgb(
    X: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray,
    seed: int,
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.05,
) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=4,
        objective="reg:squarederror",
    )
    model.fit(X, Z, sample_weight=weights)
    return model


def evaluate_pehe(pred: np.ndarray, true_te: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true_te) ** 2)))


def run_one(seed: int, n_samples: int, n_noise: int, n_splits: int) -> float:
    X, T, y, true_te = get_stress_data(n_samples=n_samples, n_noise=n_noise, seed=seed)
    m_hat, e_hat = cross_fit_nuisance(X, y, T, n_splits=n_splits, seed=seed)
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R = y - m_hat
    W = T - e_hat
    Z, weights = stabilize_residuals(R, W, clip_w=0.05, z_clip=5.0)
    tau_model = train_tau_xgb(X, Z, weights, seed=seed)
    tau_pred = tau_model.predict(X)
    pehe = evaluate_pehe(tau_pred, true_te)
    print(f"[seed {seed}] PEHE XGB-R: {pehe:.3f}")
    return pehe


def run_seeds(seeds: List[int], n_samples: int, n_noise: int, n_splits: int) -> None:
    pehes = [run_one(seed, n_samples, n_noise, n_splits) for seed in seeds]
    mean_pehe = float(np.mean(pehes))
    std_pehe = float(np.std(pehes))
    print(f"\nSummary (N={n_samples}, noise={n_noise}, seeds={seeds}): mean PEHE={mean_pehe:.3f} ± {std_pehe:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost R-learner baseline on stress test.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1024, 2023])
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-noise", type=int, default=50)
    parser.add_argument("--k-folds", type=int, default=3)
    args = parser.parse_args()

    run_seeds(args.seeds, n_samples=args.n_samples, n_noise=args.n_noise, n_splits=args.k_folds)
