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
    - LaLonde: econml.datasets or causalml
    - STAR: Harvard Dataverse (doi:10.7910/DVN/SIWH9F)
"""
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd

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
from src.utils.metrics import compute_ate, bootstrap_mean
from src.utils.latex import MacroGenerator


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
        return X.astype(np.float32), T.astype(np.float32), Y.astype(np.float32)
    except ImportError:
        print("econml not installed. Using causalml fallback...")
        try:
            from causalml.dataset import load_lalonde
            df = load_lalonde()
            T = df["treat"].values.astype(np.float32)
            Y = df["re78"].values.astype(np.float32)
            X = df.drop(columns=["treat", "re78"]).values.astype(np.float32)
            return X, T, Y
        except ImportError:
            raise ImportError(
                "Neither econml nor causalml installed. "
                "Install with: pip install econml or pip install causalml"
            )


def load_star() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load STAR class size dataset.

    Returns (X, T, Y) where:
    - X: Student/school covariates
    - T: Treatment (1=small class, 0=regular class)
    - Y: Outcome (test scores)

    Note: Requires downloading data from Harvard Dataverse.
    """
    data_path = Path(__file__).parents[2] / "data" / "star" / "STAR_Students.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"STAR data not found at {data_path}. "
            "Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SIWH9F"
        )

    df = pd.read_csv(data_path)

    # Small class (treatment) vs regular (control)
    df = df[df["stark"].isin([1, 2])]  # 1=small, 2=regular
    T = (df["stark"] == 1).astype(np.float32).values
    Y = df["treadssk"].values.astype(np.float32)  # Reading score

    # Covariates: gender, race, free lunch, school urbanicity
    cov_cols = ["gender", "race", "lunch", "school_type"]
    cov_cols = [c for c in cov_cols if c in df.columns]
    X = pd.get_dummies(df[cov_cols], drop_first=True).values.astype(np.float32)

    # Handle missing values
    mask = ~np.isnan(Y)
    return X[mask], T[mask], Y[mask]


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
        "mean_smd": float(smds.mean()),
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
    def ols_ate(idx):
        Xi, Ti, Yi = X[idx], T[idx], Y[idx]
        Xti = np.column_stack([Xi, Ti])
        return LinearRegression().fit(Xti, Yi).coef_[-1]

    ols_ci = bootstrap_mean(np.arange(len(Y)), ols_ate, n_bootstrap=100)
    rows.append({
        "dataset": dataset_name,
        "method": "OLS",
        "ate": ate_ols,
        "ci_lower": ols_ci[0],
        "ci_upper": ols_ci[1],
    })

    # CTRL-DML
    plugin_model = train_plugin(
        X, Y, T,
        use_gating=True,
        seed=seed,
        epochs=ctrl_config.plugin_epochs,
    )
    tau_plugin = predict_tau_tarnet(plugin_model, X)

    m_hat, e_hat = cross_fit_nuisance(
        X, Y, T,
        use_gating=True,
        seed=seed,
        epochs=ctrl_config.nuisance_epochs,
    )
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R = Y - m_hat
    W = T - e_hat
    Z, weights = stabilize_residuals(R, W)

    tau_model = train_rlearner(
        X, Z, weights,
        use_gating=True,
        seed=seed,
        epochs=ctrl_config.tau_epochs,
        warm_start_from=plugin_model,
        teacher_tau=tau_plugin,
    )
    tau_dml = predict_tau_rlearner(tau_model, X)

    ate_ctrl = compute_ate(tau_dml)
    ctrl_ci = bootstrap_mean(tau_dml, np.mean, n_bootstrap=100)

    rows.append({
        "dataset": dataset_name,
        "method": "CTRL-DML",
        "ate": ate_ctrl,
        "ci_lower": ctrl_ci[0],
        "ci_upper": ctrl_ci[1],
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
        use_gating=True,
        seed=seed,
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


def generate_latex_macros(ate_df: pd.DataFrame, balance_df: pd.DataFrame, gen: MacroGenerator):
    """Generate LaTeX macros."""
    for _, row in ate_df.iterrows():
        ds = row["dataset"].title()
        method = row["method"].replace("-", "").replace(" ", "")
        gen.add(f"{ds}{method}Ate", row["ate"], f"{ds} Real Data")
        gen.add(f"{ds}{method}CiLo", row["ci_lower"], f"{ds} Real Data")
        gen.add(f"{ds}{method}CiHi", row["ci_upper"], f"{ds} Real Data")

    for _, row in balance_df.iterrows():
        ds = row["dataset"].title()
        weighting = row["weighting"].replace("-", "").title()
        gen.add(f"{ds}{weighting}Smd", row["mean_smd"], f"{ds} Real Data")
        gen.add(f"{ds}{weighting}Count", row["n_imbalanced"], f"{ds} Real Data")


def main():
    parser = argparse.ArgumentParser(description="CTRL-DML Real Data Experiments")
    parser.add_argument("--dataset", choices=["lalonde", "star", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = get_output_manager()
    gen = MacroGenerator()

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

        generate_latex_macros(ate_combined, balance_combined, gen)


if __name__ == "__main__":
    main()
