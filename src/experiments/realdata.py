"""
CTRL-DML Real Data Experiments.

Runs CTRL-DML on real-world observational datasets:
- LaLonde: NSW job training program
- STAR: Tennessee class size experiment
- Amazon: Product review sentiment (multimodal)

Usage:
    python -m src.experiments.realdata --dataset lalonde
    python -m src.experiments.realdata --dataset star
    python -m src.experiments.realdata --dataset all

Data Sources:
    - LaLonde: econml.datasets, causalml, or direct CSV download (fallback)
    - STAR: Harvard Dataverse (doi:10.7910/DVN/SIWH9F)
"""
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.models.orthogonal_learner import (
    train_plugin,
    cross_fit_nuisance,
    stabilize_residuals,
    train_rlearner,
    predict_tau_tarnet,
    predict_tau_rlearner,
    set_seed,
    CTRLConfig,
)
from src.utils.io import get_output_manager
from src.utils.metrics import compute_ate, bootstrap_mean, bootstrap_ci


FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"


def load_lalonde() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load LaLonde NSW dataset.

    Returns (X, T, Y) where:
    - X: Covariates (age, education, earnings, etc.)
    - T: Treatment (1=job training, 0=control)
    - Y: Outcome (re78 - earnings in 1978)
    """
    try:
        from econml.datasets import get_lalonde
        X, T, Y, _, _, _, _ = get_lalonde()
        Y = (Y.astype(np.float32) / 1000.0)
        return X.astype(np.float32), T.astype(np.float32), Y
    except ImportError:
        print("econml not installed. Using causalml fallback...")
        try:
            from causalml.dataset import load_lalonde
            df = load_lalonde()
            T = df["treat"].values.astype(np.float32)
            Y = df["re78"].values.astype(np.float32) / 1000.0
            X = df.drop(columns=["treat", "re78"]).values.astype(np.float32)
            return X, T, Y
        except ImportError:
            print("causalml not installed. Downloading Lalonde CSV...")
            data_root = Path(__file__).parents[2] / "data" / "lalonde"
            data_path = data_root / "lalonde.csv"
            if not data_path.exists():
                data_root.mkdir(parents=True, exist_ok=True)
                data_path = _download_lalonde(data_root)

            df = pd.read_csv(data_path)
            if "rownames" in df.columns:
                df = df.drop(columns=["rownames"])
            treat_col = next((c for c in ["treat", "treatment", "T"] if c in df.columns), None)
            outcome_col = next((c for c in ["re78", "outcome", "Y"] if c in df.columns), None)
            if treat_col is None or outcome_col is None:
                raise KeyError("Lalonde CSV missing treatment or outcome columns.")

            T = df[treat_col].values.astype(np.float32)
            Y = df[outcome_col].values.astype(np.float32) / 1000.0
            X_df = df.drop(columns=[treat_col, outcome_col])
            X = pd.get_dummies(X_df, drop_first=True).values.astype(np.float32)

            mask = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
            return X[mask], T[mask], Y[mask]


def load_star() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load STAR class size dataset.

    Returns (X, T, Y) where:
    - X: Student/school covariates
    - T: Treatment (1=small class, 0=regular class)
    - Y: Outcome (test scores)

    Note: Requires downloading data from Harvard Dataverse.
    """
    data_root = Path(__file__).parents[2] / "data" / "star"
    data_path = data_root / "STAR_Students.tab"

    if not data_path.exists():
        data_root.mkdir(parents=True, exist_ok=True)
        data_path = _download_star_students(data_root)

    df = pd.read_csv(data_path, sep="\t")

    # Small class (treatment) vs regular (control).
    # STAR uses class-type codes (1=small, 2=regular, 3=regular+aide).
    class_candidates = ["gkclasstype", "g1classtype", "g2classtype", "g3classtype"]
    score_candidates = ["gktreadss", "g1treadss", "g2treadss", "g3treadss"]

    class_col = next((c for c in class_candidates if c in df.columns), None)
    score_col = next((c for c in score_candidates if c in df.columns), None)
    if class_col is None or score_col is None:
        raise KeyError("STAR columns missing class type or reading score.")

    df = df[df[class_col].isin([1, 2])].copy()
    T = (df[class_col] == 1).astype(np.float32).values
    Y = df[score_col].astype(np.float32).values

    # Covariates: gender, race, free lunch, school urbanicity (grade-k preferred).
    cov_cols = ["gender", "race", "gkfreelunch", "gksurban"]
    cov_cols = [c for c in cov_cols if c in df.columns]
    cov_df = df[cov_cols].copy()
    for col in cov_df.columns:
        cov_df[col] = cov_df[col].fillna(-1).astype("category")
    X = pd.get_dummies(cov_df, drop_first=True).values.astype(np.float32)

    # Handle missing outcomes.
    mask = ~np.isnan(Y)
    return X[mask], T[mask], Y[mask]


def _download_star_students(data_root: Path) -> Path:
    """
    Download STAR_Students.tab from Harvard Dataverse if missing.

    Returns path to the downloaded file.
    """
    meta_url = (
        "https://dataverse.harvard.edu/api/datasets/:persistentId/"
        "?persistentId=doi:10.7910/DVN/SIWH9F"
    )
    resp = requests.get(meta_url, timeout=30)
    resp.raise_for_status()
    files = resp.json().get("data", {}).get("latestVersion", {}).get("files", [])

    file_id = None
    filename = None
    for item in files:
        data_file = item.get("dataFile", {})
        name = data_file.get("filename", "")
        if name.lower().startswith("star_students"):
            file_id = data_file.get("id")
            filename = name
            break

    if file_id is None:
        raise RuntimeError("STAR_Students file not found in Dataverse metadata.")

    download_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
    resp = requests.get(download_url, timeout=60)
    resp.raise_for_status()

    out_path = data_root / filename
    out_path.write_bytes(resp.content)
    return out_path


def _download_lalonde(data_root: Path) -> Path:
    """
    Download Lalonde dataset CSV if missing.

    Returns path to the downloaded file.
    """
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/MatchIt/lalonde.csv"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path = data_root / "lalonde.csv"
    out_path.write_bytes(resp.content)
    return out_path


def compute_smd(X: np.ndarray, T: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute standardized mean differences for covariate balance.

    Args:
        X: Covariates (n, d)
        T: Treatment indicator (n,)
        weights: Optional sample weights

    Returns:
        Dict with 'mean_smd' and 'n_imbalanced' (SMD > 0.1)
    """
    t1 = T == 1
    t0 = T == 0

    if weights is None:
        weights = np.ones(len(T))

    smds = []
    for j in range(X.shape[1]):
        x = X[:, j]
        if weights is not None:
            w1 = weights[t1]
            w0 = weights[t0]
            mean1 = np.average(x[t1], weights=w1)
            mean0 = np.average(x[t0], weights=w0)
        else:
            mean1 = x[t1].mean()
            mean0 = x[t0].mean()

        pooled_std = np.sqrt((x[t1].var() + x[t0].var()) / 2 + 1e-8)
        smd = abs(mean1 - mean0) / pooled_std
        smds.append(smd)

    smds = np.array(smds)
    return {
        "max_smd": float(smds.max()),
        "n_imbalanced": int((smds > 0.1).sum()),
    }


def run_ate_estimation(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    dataset_name: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run ATE estimation with multiple methods.
    """
    rows = []
    ctrl_config = CTRLConfig()

    if FAST_RUN:
        ctrl_config.plugin_epochs = 100
        ctrl_config.nuisance_epochs = 80
        ctrl_config.tau_epochs = 150

    set_seed(seed)

    # OLS baseline
    from sklearn.linear_model import LinearRegression
    X_with_t = np.column_stack([X, T])
    ols = LinearRegression().fit(X_with_t, Y)
    ate_ols = ols.coef_[-1]

    # Bootstrap CI for OLS
    def ols_ate(data):
        Xi, Ti, Yi = data["X"], data["T"], data["Y"]
        Xti = np.column_stack([Xi, Ti])
        return LinearRegression().fit(Xti, Yi).coef_[-1]

    _, ols_ci_lo, ols_ci_hi = bootstrap_ci(
        ols_ate,
        {"X": X, "T": T, "Y": Y},
        n_bootstrap=100,
        seed=seed,
    )
    rows.append({
        "dataset": dataset_name,
        "method": "OLS",
        "ate": ate_ols,
        "ci_lower": ols_ci_lo,
        "ci_upper": ols_ci_hi,
    })

    # CTRL-DML
    plugin_model = train_plugin(
        X, Y, T,
        use_gating=ctrl_config.use_gating,
        lambda_sparsity=ctrl_config.lambda_sparsity,
        seed=seed,
        dropout_p=ctrl_config.dropout_p,
        hidden_dim=ctrl_config.hidden_tau,
        batch_size=ctrl_config.batch_size,
        epochs=ctrl_config.plugin_epochs,
    )
    tau_plugin = predict_tau_tarnet(plugin_model, X)

    m_hat, e_hat = cross_fit_nuisance(
        X, Y, T,
        use_gating=ctrl_config.use_gating,
        lambda_sparsity=ctrl_config.lambda_sparsity,
        seed=seed,
        k_folds=ctrl_config.k_folds,
        dropout_p=ctrl_config.dropout_p,
        hidden_dim=ctrl_config.hidden_dim,
        batch_size=ctrl_config.batch_size,
        epochs=ctrl_config.nuisance_epochs,
    )
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R = Y - m_hat
    W = T - e_hat
    Z, weights = stabilize_residuals(R, W, w_clip=ctrl_config.w_clip, z_clip=ctrl_config.z_clip, eps=ctrl_config.eps)

    tau_model = train_rlearner(
        X, Z, weights,
        use_gating=ctrl_config.use_gating,
        lambda_tau=ctrl_config.lambda_tau,
        seed=seed,
        dropout_p=ctrl_config.dropout_p,
        hidden_dim=ctrl_config.hidden_tau,
        batch_size=ctrl_config.batch_size,
        epochs=ctrl_config.tau_epochs,
        lr=ctrl_config.lr_tau,
        grad_clip=ctrl_config.grad_clip,
        warm_start_from=plugin_model,
        teacher_tau=tau_plugin,
        aux_beta_start=ctrl_config.aux_beta_start,
        aux_beta_end=ctrl_config.aux_beta_end,
        aux_decay_epochs=ctrl_config.aux_decay_epochs,
    )
    tau_dml = predict_tau_rlearner(tau_model, X)

    ate_ctrl = compute_ate(tau_dml)
    _, ctrl_ci_lo, ctrl_ci_hi = bootstrap_mean(tau_dml, n_bootstrap=100, seed=seed)

    rows.append({
        "dataset": dataset_name,
        "method": "CTRL-DML",
        "ate": ate_ctrl,
        "ci_lower": ctrl_ci_lo,
        "ci_upper": ctrl_ci_hi,
    })

    # CausalForest (if available)
    try:
        from econml.dml import CausalForestDML
        cf = CausalForestDML(n_estimators=100, random_state=seed)
        cf.fit(Y, T, X=X)
        tau_cf = cf.effect(X)
        ate_cf = compute_ate(tau_cf)
        cf_ci = cf.effect_interval(X, alpha=0.05)
        rows.append({
            "dataset": dataset_name,
            "method": "CausalForest",
            "ate": ate_cf,
            "ci_lower": float(np.mean(cf_ci[0])),
            "ci_upper": float(np.mean(cf_ci[1])),
        })
    except ImportError:
        pass

    return pd.DataFrame(rows)


def run_balance_check(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    dataset_name: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Check covariate balance before/after weighting."""
    rows = []
    ctrl_config = CTRLConfig()
    set_seed(seed)

    # Unweighted balance
    unweighted = compute_smd(X, T)
    rows.append({
        "dataset": dataset_name,
        "weighting": "unweighted",
        **unweighted,
    })

    # Propensity-weighted balance
    m_hat, e_hat = cross_fit_nuisance(
        X, Y, T,
        use_gating=ctrl_config.use_gating,
        lambda_sparsity=ctrl_config.lambda_sparsity,
        seed=seed,
        k_folds=ctrl_config.k_folds,
        dropout_p=ctrl_config.dropout_p,
        hidden_dim=ctrl_config.hidden_dim,
        batch_size=ctrl_config.batch_size,
        epochs=ctrl_config.nuisance_epochs if not FAST_RUN else 80,
    )
    e_hat = np.clip(e_hat, 0.01, 0.99)

    # IPW weights
    ipw = T / e_hat + (1 - T) / (1 - e_hat)
    weighted = compute_smd(X, T, weights=ipw)
    rows.append({
        "dataset": dataset_name,
        "weighting": "CTRL-IPW",
        **weighted,
    })

    return pd.DataFrame(rows)


def print_latex_reminder():
    """Print reminder to regenerate LaTeX macros."""
    print("\nTo regenerate LaTeX macros, run: python -m src.utils.latex")


def main():
    parser = argparse.ArgumentParser(description="CTRL-DML Real Data Experiments")
    parser.add_argument("--dataset", choices=["lalonde", "star", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = get_output_manager()

    all_ate = []
    all_balance = []

    if args.dataset in ["lalonde", "all"]:
        try:
            print("=== LaLonde Dataset ===")
            X, T, Y = load_lalonde()
            print(f"  Loaded: N={len(Y)}, treated={T.sum():.0f}")

            ate_df = run_ate_estimation(X, T, Y, "lalonde", seed=args.seed)
            balance_df = run_balance_check(X, T, Y, "lalonde", seed=args.seed)

            # Save per-dataset files (matching existing naming)
            ate_df.to_csv(output.csv_path("realdata_lalonde_ate"), index=False)
            balance_df.to_csv(output.csv_path("realdata_lalonde_smd"), index=False)

            all_ate.append(ate_df)
            all_balance.append(balance_df)

            print("\nATE Estimates:")
            print(ate_df.to_string(index=False))
            print("\nBalance Check:")
            print(balance_df.to_string(index=False))
        except (ImportError, FileNotFoundError) as e:
            print(f"  Skipping LaLonde: {e}")

    if args.dataset in ["star", "all"]:
        try:
            print("\n=== STAR Dataset ===")
            X, T, Y = load_star()
            print(f"  Loaded: N={len(Y)}, treated={T.sum():.0f}")

            ate_df = run_ate_estimation(X, T, Y, "star", seed=args.seed)
            balance_df = run_balance_check(X, T, Y, "star", seed=args.seed)

            # Save per-dataset files (matching existing naming)
            ate_df.to_csv(output.csv_path("realdata_star_ate"), index=False)
            balance_df.to_csv(output.csv_path("realdata_star_smd"), index=False)

            all_ate.append(ate_df)
            all_balance.append(balance_df)

            print("\nATE Estimates:")
            print(ate_df.to_string(index=False))
            print("\nBalance Check:")
            print(balance_df.to_string(index=False))
        except (ImportError, FileNotFoundError) as e:
            print(f"  Skipping STAR: {e}")

    if all_ate:
        ate_combined = pd.concat(all_ate, ignore_index=True)
        csv_path = output.csv_path("realdata_ate")
        ate_combined.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    if all_balance:
        balance_combined = pd.concat(all_balance, ignore_index=True)
        csv_path = output.csv_path("realdata_balance")
        balance_combined.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        print_latex_reminder()


if __name__ == "__main__":
    main()
