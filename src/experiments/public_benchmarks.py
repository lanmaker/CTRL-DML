"""
Public benchmark baselines for TWINS and ACIC 2019.

Produces ATE estimates and confidence intervals for CausalForest and TARNet.
Outputs:
  - output/tables/public_benchmarks.csv
  - output/tables/tab_public_benchmarks.tex
"""
import argparse
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.data.twins import load_twins
from src.data.acic import load_acic
from src.models.orthogonal_learner import train_plugin, predict_tau_tarnet, set_seed, CTRLConfig
from src.utils.metrics import bootstrap_mean
from src.utils.io import get_output_manager
from src.utils.latex import df_to_latex_table


FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"


@dataclass
class PublicBenchmarkConfig:
    """Configuration for public baselines."""
    seed: int = 42
    twins_pairs: int = 5000
    acic_dataset: str = "low1"
    acic_kind: str = "low"
    n_bootstrap: int = 50 if FAST_RUN else 200


def _bootstrap_ci_from_tau(tau: np.ndarray, seed: int, n_bootstrap: int) -> tuple:
    """Compute mean and bootstrap CI for ATE from tau samples."""
    mean, ci_lo, ci_hi = bootstrap_mean(tau, n_bootstrap=n_bootstrap, seed=seed)
    return mean, ci_lo, ci_hi


def _estimate_tarnet_ate(X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int, n_bootstrap: int) -> tuple:
    """Estimate ATE using TARNet plug-in."""
    ctrl_config = CTRLConfig()
    if FAST_RUN:
        ctrl_config.plugin_epochs = 100

    set_seed(seed)
    model = train_plugin(
        X, Y, T,
        use_gating=ctrl_config.use_gating,
        lambda_sparsity=ctrl_config.lambda_sparsity,
        seed=seed,
        dropout_p=ctrl_config.dropout_p,
        hidden_dim=ctrl_config.hidden_tau,
        batch_size=ctrl_config.batch_size,
        epochs=ctrl_config.plugin_epochs,
    )
    tau = predict_tau_tarnet(model, X)
    return _bootstrap_ci_from_tau(tau, seed, n_bootstrap)


def _estimate_cf_ate(X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int, n_bootstrap: int) -> tuple:
    """Estimate ATE using CausalForestDML."""
    from econml.dml import CausalForestDML
    set_seed(seed)
    cf = CausalForestDML(n_estimators=200 if not FAST_RUN else 100, random_state=seed)
    cf.fit(Y, T, X=X)
    tau = cf.effect(X)
    return _bootstrap_ci_from_tau(tau, seed, n_bootstrap)


def run_public_benchmarks(config: PublicBenchmarkConfig) -> pd.DataFrame:
    """Run public baselines for TWINS and ACIC."""
    rows: List[dict] = []

    # === TWINS (5k pairs by default) ===
    X_full, T_full, Y_full = load_twins()
    n_pairs = len(Y_full) // 2
    rng = np.random.default_rng(config.seed)
    n_pairs_use = min(config.twins_pairs, n_pairs)
    pair_idx = rng.choice(n_pairs, size=n_pairs_use, replace=False)
    idx = np.sort(np.concatenate([pair_idx * 2, pair_idx * 2 + 1]))
    X_twins, T_twins, Y_twins = X_full[idx], T_full[idx], Y_full[idx]

    ate, ci_lo, ci_hi = _estimate_cf_ate(X_twins, T_twins, Y_twins, config.seed, config.n_bootstrap)
    rows.append({
        "dataset": "twins",
        "method": "CausalForest",
        "ate": ate,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n": len(Y_twins),
    })

    ate, ci_lo, ci_hi = _estimate_tarnet_ate(X_twins, T_twins, Y_twins, config.seed, config.n_bootstrap)
    rows.append({
        "dataset": "twins",
        "method": "TARNet",
        "ate": ate,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n": len(Y_twins),
    })

    # === ACIC (low1 by default) ===
    X_acic, T_acic, Y_acic = load_acic(config.acic_dataset, kind=config.acic_kind)

    ate, ci_lo, ci_hi = _estimate_cf_ate(X_acic, T_acic, Y_acic, config.seed, config.n_bootstrap)
    rows.append({
        "dataset": f"acic_{config.acic_dataset}",
        "method": "CausalForest",
        "ate": ate,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n": len(Y_acic),
    })

    ate, ci_lo, ci_hi = _estimate_tarnet_ate(X_acic, T_acic, Y_acic, config.seed, config.n_bootstrap)
    rows.append({
        "dataset": f"acic_{config.acic_dataset}",
        "method": "TARNet",
        "ate": ate,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n": len(Y_acic),
    })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Public benchmark baselines (TWINS/ACIC)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--twins-pairs", type=int, default=5000)
    parser.add_argument("--acic-dataset", type=str, default="low1")
    parser.add_argument("--acic-kind", choices=["low", "high"], default="low")
    parser.add_argument("--bootstrap", type=int, default=0, help="Override bootstrap iterations")
    args = parser.parse_args()

    config = PublicBenchmarkConfig(
        seed=args.seed,
        twins_pairs=args.twins_pairs,
        acic_dataset=args.acic_dataset,
        acic_kind=args.acic_kind,
    )
    if args.bootstrap > 0:
        config.n_bootstrap = args.bootstrap

    output = get_output_manager()
    df = run_public_benchmarks(config)
    csv_path = output.csv_path("public_benchmarks")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    table_df = df.copy()
    table_df["ATE (CI)"] = table_df.apply(
        lambda r: f"{r['ate']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]",
        axis=1,
    )
    table_df = table_df[["dataset", "method", "ATE (CI)"]]
    table_df["dataset"] = table_df["dataset"].str.replace("_", "-")

    df_to_latex_table(
        table_df,
        output.table_path("tab_public_benchmarks"),
        caption="Public benchmark ATE baselines (TWINS and ACIC).",
        label="tab:public_benchmarks",
        columns=["dataset", "method", "ATE (CI)"],
        column_format="llc",
        escape=False,
    )


if __name__ == "__main__":
    main()
