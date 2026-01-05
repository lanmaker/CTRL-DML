"""
Public benchmark baselines for TWINS and ACIC 2019.

Produces OOF ATE estimates and confidence intervals for CausalForest and TARNet.
CausalForest uses built-in ATE inference when available (fallback to refit bootstrap).
Outputs:
  - output/tables/public_benchmarks.csv
  - output/tables/tab_public_benchmarks.tex
"""
import argparse
import os
from dataclasses import dataclass
from typing import List, Callable, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.twins import load_twins
from src.data.acic import load_acic
from src.models.orthogonal_learner import train_plugin, predict_tau_tarnet, CTRLConfig
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
    k_folds: int = 5
    n_bootstrap: int = 30 if FAST_RUN else 100
    alpha: float = 0.05


def _resolve_k_folds(T: np.ndarray, k_folds: int) -> int:
    counts = [int(np.sum(T == 0)), int(np.sum(T == 1))]
    min_class = min(counts)
    if min_class < 2:
        raise ValueError("Need at least two samples per treatment group for cross-fitting.")
    return max(2, min(k_folds, min_class))


def _stratified_bootstrap_indices(T: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx_t = np.where(T == 1)[0]
    idx_c = np.where(T == 0)[0]
    if len(idx_t) == 0 or len(idx_c) == 0:
        return rng.choice(len(T), size=len(T), replace=True)
    boot_t = rng.choice(idx_t, size=len(idx_t), replace=True)
    boot_c = rng.choice(idx_c, size=len(idx_c), replace=True)
    idx = np.concatenate([boot_t, boot_c])
    return rng.permutation(idx)


def bootstrap_refit_ate(
    estimate_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, int], float],
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = estimate_fn(X, T, Y, seed)
    estimates = []
    for b in range(n_bootstrap):
        idx = _stratified_bootstrap_indices(T, rng)
        estimates.append(estimate_fn(X[idx], T[idx], Y[idx], seed + b + 1))
    estimates = np.array(estimates)
    ci_lower = np.percentile(estimates, 100 * alpha / 2)
    ci_upper = np.percentile(estimates, 100 * (1 - alpha / 2))
    return point, float(ci_lower), float(ci_upper)


def _estimate_tarnet_oof_ate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    ctrl_config: CTRLConfig,
    seed: int,
    k_folds: int,
) -> float:
    """Estimate ATE using OOF TARNet plug-in predictions."""
    k_folds = _resolve_k_folds(T, k_folds)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    tau_oof = np.zeros(len(Y), dtype=np.float32)

    for fold, (tr, val) in enumerate(skf.split(X, T)):
        fold_seed = seed + 100 * (fold + 1)
        model = train_plugin(
            X[tr], Y[tr], T[tr],
            use_gating=ctrl_config.use_gating,
            lambda_sparsity=ctrl_config.lambda_sparsity,
            seed=fold_seed,
            dropout_p=ctrl_config.dropout_p,
            hidden_dim=ctrl_config.hidden_tau,
            batch_size=ctrl_config.batch_size,
            epochs=ctrl_config.plugin_epochs,
        )
        tau_oof[val] = predict_tau_tarnet(model, X[val])
    return float(np.mean(tau_oof))


def _estimate_cf_ate_inference(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    seed: int,
    alpha: float,
) -> Tuple[float, Optional[float], Optional[float]]:
    from econml.dml import CausalForestDML
    cf = CausalForestDML(n_estimators=200 if not FAST_RUN else 100, random_state=seed)
    cf.fit(Y, T, X=X)

    if hasattr(cf, "ate_inference"):
        inf = cf.ate_inference(X)
        ate = float(getattr(inf, "mean_point", np.mean(cf.effect(X))))
        if hasattr(inf, "conf_int"):
            ci = inf.conf_int(alpha=alpha)
            return ate, float(ci[0]), float(ci[1])
        if hasattr(inf, "conf_int_mean"):
            ci = inf.conf_int_mean(alpha=alpha)
            return ate, float(ci[0]), float(ci[1])
        if hasattr(inf, "mean_ci_lower") and hasattr(inf, "mean_ci_upper"):
            return ate, float(inf.mean_ci_lower), float(inf.mean_ci_upper)

    if hasattr(cf, "effect_inference"):
        inf = cf.effect_inference(X)
        summary = inf.population_summary()
        ate = float(summary.mean_point)
        if hasattr(summary, "conf_int_mean"):
            ci = summary.conf_int_mean(alpha=alpha)
            return ate, float(ci[0]), float(ci[1])
        if hasattr(summary, "mean_ci_lower") and hasattr(summary, "mean_ci_upper"):
            return ate, float(summary.mean_ci_lower), float(summary.mean_ci_upper)

    return float(np.mean(cf.effect(X))), None, None


def run_public_benchmarks(config: PublicBenchmarkConfig) -> pd.DataFrame:
    """Run public baselines for TWINS and ACIC."""
    rows: List[dict] = []
    ctrl_config = CTRLConfig()
    ctrl_config.k_folds = config.k_folds
    if FAST_RUN:
        ctrl_config.plugin_epochs = 100
        ctrl_config.k_folds = min(3, ctrl_config.k_folds)
        config.k_folds = ctrl_config.k_folds
        config.n_bootstrap = min(50, config.n_bootstrap)

    # === TWINS (5k pairs by default) ===
    X_full, T_full, Y_full = load_twins()
    n_pairs = len(Y_full) // 2
    rng = np.random.default_rng(config.seed)
    n_pairs_use = min(config.twins_pairs, n_pairs)
    pair_idx = rng.choice(n_pairs, size=n_pairs_use, replace=False)
    idx = np.sort(np.concatenate([pair_idx * 2, pair_idx * 2 + 1]))
    X_twins, T_twins, Y_twins = X_full[idx], T_full[idx], Y_full[idx]

    ate, ci_lo, ci_hi = _estimate_cf_ate_inference(
        X_twins, T_twins, Y_twins, seed=config.seed, alpha=config.alpha
    )
    if ci_lo is None or ci_hi is None:
        def _cf_estimator(Xi, Ti, Yi, est_seed):
            return _estimate_cf_ate_inference(Xi, Ti, Yi, seed=est_seed, alpha=config.alpha)[0]
        ate, ci_lo, ci_hi = bootstrap_refit_ate(
            _cf_estimator, X_twins, T_twins, Y_twins,
            n_bootstrap=config.n_bootstrap, seed=config.seed, alpha=config.alpha
        )
    rows.append({
        "dataset": "twins",
        "method": "CausalForest",
        "ate": ate,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n": len(Y_twins),
    })

    def _tarnet_estimator(Xi, Ti, Yi, est_seed):
        return _estimate_tarnet_oof_ate(
            Xi, Ti, Yi, ctrl_config=ctrl_config, seed=est_seed, k_folds=ctrl_config.k_folds
        )
    ate, ci_lo, ci_hi = bootstrap_refit_ate(
        _tarnet_estimator, X_twins, T_twins, Y_twins,
        n_bootstrap=config.n_bootstrap, seed=config.seed, alpha=config.alpha
    )
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

    ate, ci_lo, ci_hi = _estimate_cf_ate_inference(
        X_acic, T_acic, Y_acic, seed=config.seed, alpha=config.alpha
    )
    if ci_lo is None or ci_hi is None:
        def _cf_estimator_acic(Xi, Ti, Yi, est_seed):
            return _estimate_cf_ate_inference(Xi, Ti, Yi, seed=est_seed, alpha=config.alpha)[0]
        ate, ci_lo, ci_hi = bootstrap_refit_ate(
            _cf_estimator_acic, X_acic, T_acic, Y_acic,
            n_bootstrap=config.n_bootstrap, seed=config.seed, alpha=config.alpha
        )
    rows.append({
        "dataset": f"acic_{config.acic_dataset}",
        "method": "CausalForest",
        "ate": ate,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n": len(Y_acic),
    })

    ate, ci_lo, ci_hi = bootstrap_refit_ate(
        _tarnet_estimator, X_acic, T_acic, Y_acic,
        n_bootstrap=config.n_bootstrap, seed=config.seed, alpha=config.alpha
    )
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
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=0, help="Override bootstrap iterations")
    args = parser.parse_args()

    config = PublicBenchmarkConfig(
        seed=args.seed,
        twins_pairs=args.twins_pairs,
        acic_dataset=args.acic_dataset,
        acic_kind=args.acic_kind,
        k_folds=args.k_folds,
        alpha=args.alpha,
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
