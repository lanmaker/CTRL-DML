"""
CTRL-DML Benchmark Experiments.

Runs CTRL-DML on standard causal inference benchmarks:
- IHDP: Semi-synthetic with ground-truth CATE
- TWINS: Paired observational data
- ACIC: ACIC 2019 competition datasets

Usage:
    python -m src.experiments.benchmarks --dataset ihdp
    python -m src.experiments.benchmarks --dataset twins
    python -m src.experiments.benchmarks --dataset acic
    python -m src.experiments.benchmarks --dataset all
"""
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ihdp import load_ihdp_train_test
from src.data.twins import load_twins, compute_twins_ite
from src.data.acic import load_acic, list_acic_datasets
from src.models.orthogonal_learner import (
    CTRLOrthogonalLearner,
    CTRLConfig,
    set_seed,
)
from src.utils.metrics import compute_pehe, compute_ate
from src.utils.io import get_output_manager
from src.utils.latex import df_to_latex_table


# Fast mode for quick testing
FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    n_replicates: int = 10 if FAST_RUN else 100
    seeds: List[int] = None

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 7] if FAST_RUN else [42, 7, 1024]


def run_ihdp_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    """
    Run CTRL-DML on IHDP benchmark.

    Args:
        config: Benchmark configuration

    Returns:
        DataFrame with results per replicate
    """
    print("=== IHDP Benchmark ===", flush=True)
    rows = []

    ctrl_config = CTRLConfig()
    if FAST_RUN:
        ctrl_config.plugin_epochs = 100
        ctrl_config.nuisance_epochs = 80
        ctrl_config.tau_epochs = 150
        ctrl_config.k_folds = min(3, ctrl_config.k_folds)

    for rep in range(1, config.n_replicates + 1):
        for seed in config.seeds:
            print(f"  Rep {rep}, seed {seed}...", flush=True)
            set_seed(seed)

            # Load data
            data = load_ihdp_train_test(rep=rep, seed=seed)

            # Train CTRL-DML
            model = CTRLOrthogonalLearner(ctrl_config)
            model.fit(
                data["X_train"],
                data["Y_train"],
                data["T_train"],
                seed=seed
            )

            # Evaluate on test set
            tau_pred = model.predict_tau(data["X_test"])
            tau_plugin = model.predict_plugin(data["X_test"])

            pehe_dml = compute_pehe(tau_pred, data["cate_test"])
            pehe_plugin = compute_pehe(tau_plugin, data["cate_test"])
            ate_bias = compute_ate(tau_pred) - compute_ate(data["cate_test"])

            rows.append({
                "dataset": "ihdp",
                "replicate": rep,
                "seed": seed,
                "pehe_dml": pehe_dml,
                "pehe_plugin": pehe_plugin,
                "ate_bias": ate_bias,
            })

    return pd.DataFrame(rows)


def run_twins_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    """
    Run CTRL-DML on TWINS benchmark.

    Note: TWINS has paired structure where each X appears twice (T=0 and T=1).
    We use the paired ITE as ground truth for evaluation.
    """
    print("=== TWINS Benchmark ===")
    rows = []

    ctrl_config = CTRLConfig()
    if FAST_RUN:
        ctrl_config.plugin_epochs = 100
        ctrl_config.nuisance_epochs = 80
        ctrl_config.tau_epochs = 150
        ctrl_config.k_folds = min(3, ctrl_config.k_folds)

    # Load full TWINS data
    X_full, T_full, Y_full = load_twins()

    # Get paired ground truth ITE
    ite_true = compute_twins_ite(Y_full)
    n_pairs = len(ite_true)

    for seed in config.seeds:
        print(f"  Seed {seed}...")
        set_seed(seed)

        # Use treated twins (every other row starting from 0) for evaluation
        X_treated = X_full[0::2]

        # Split into train/test
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n_pairs)
        n_train = int(n_pairs * 0.8)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        # Full training data (both twins)
        train_full_idx = np.concatenate([train_idx * 2, train_idx * 2 + 1])
        train_full_idx = np.sort(train_full_idx)

        X_train = X_full[train_full_idx]
        T_train = T_full[train_full_idx]
        Y_train = Y_full[train_full_idx]

        # Train CTRL-DML
        model = CTRLOrthogonalLearner(ctrl_config)
        model.fit(X_train, Y_train, T_train, seed=seed)

        # Evaluate on test treated twins
        X_test = X_treated[test_idx]
        ite_test = ite_true[test_idx]

        tau_pred = model.predict_tau(X_test)
        tau_plugin = model.predict_plugin(X_test)

        pehe_dml = compute_pehe(tau_pred, ite_test)
        pehe_plugin = compute_pehe(tau_plugin, ite_test)
        ate_bias = compute_ate(tau_pred) - compute_ate(ite_test)

        rows.append({
            "dataset": "twins",
            "seed": seed,
            "pehe_dml": pehe_dml,
            "pehe_plugin": pehe_plugin,
            "ate_bias": ate_bias,
        })

    return pd.DataFrame(rows)


def run_acic_benchmark(
    config: BenchmarkConfig,
    kind: str = "low",
    n_datasets: int = 5
) -> pd.DataFrame:
    """
    Run CTRL-DML on ACIC 2019 benchmark datasets.

    Note: ACIC doesn't have ground-truth CATE, so we only report ATE estimates.
    """
    print(f"=== ACIC Benchmark ({kind}) ===")
    rows = []

    ctrl_config = CTRLConfig()
    if FAST_RUN:
        ctrl_config.plugin_epochs = 100
        ctrl_config.nuisance_epochs = 80
        ctrl_config.tau_epochs = 150
        ctrl_config.k_folds = min(3, ctrl_config.k_folds)

    # Get available datasets
    datasets = list_acic_datasets(kind)[:n_datasets]

    for ds_name in datasets:
        for seed in config.seeds:
            print(f"  Dataset {ds_name}, seed {seed}...")
            set_seed(seed)

            # Load data
            X, T, Y = load_acic(ds_name.replace(".csv", ""), kind=kind)

            # Split into train/test
            rng = np.random.default_rng(seed)
            n = len(Y)
            idx = rng.permutation(n)
            n_train = int(n * 0.8)

            X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
            T_train, T_test = T[idx[:n_train]], T[idx[n_train:]]
            Y_train, Y_test = Y[idx[:n_train]], Y[idx[n_train:]]

            # Train CTRL-DML
            model = CTRLOrthogonalLearner(ctrl_config)
            model.fit(X_train, Y_train, T_train, seed=seed)

            # Predict
            tau_pred = model.predict_tau(X_test)
            tau_plugin = model.predict_plugin(X_test)

            rows.append({
                "dataset": f"acic_{ds_name}",
                "seed": seed,
                "ate_dml": compute_ate(tau_pred),
                "ate_plugin": compute_ate(tau_plugin),
            })

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame, group_col: str = "dataset") -> pd.DataFrame:
    """Summarize benchmark results."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in [group_col, "replicate", "seed"]]

    summary = df.groupby(group_col)[numeric_cols].agg(["mean", "std"]).round(4)
    summary.columns = ["_".join(col) for col in summary.columns]
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description="CTRL-DML Benchmark Experiments")
    parser.add_argument(
        "--dataset",
        choices=["ihdp", "twins", "acic", "all"],
        default="ihdp",
        help="Which benchmark to run"
    )
    parser.add_argument("--n-replicates", type=int, default=10 if FAST_RUN else 100)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--acic-kind", choices=["low", "high"], default="low")
    parser.add_argument("--acic-n-datasets", type=int, default=5)

    args = parser.parse_args()

    output = get_output_manager()
    seeds = args.seeds
    if seeds is None and args.n_seeds is not None:
        seeds = list(range(42, 42 + args.n_seeds))

    config = BenchmarkConfig(
        n_replicates=args.n_replicates,
        seeds=seeds,
    )

    all_results = []

    if args.dataset in ["ihdp", "all"]:
        df_ihdp = run_ihdp_benchmark(config)
        all_results.append(df_ihdp)
        print("\nIHDP Summary:")
        print(summarize_results(df_ihdp).to_string(index=False))

    if args.dataset in ["twins", "all"]:
        df_twins = run_twins_benchmark(config)
        all_results.append(df_twins)
        print("\nTWINS Summary:")
        print(summarize_results(df_twins).to_string(index=False))

    if args.dataset in ["acic", "all"]:
        df_acic = run_acic_benchmark(
            config,
            kind=args.acic_kind,
            n_datasets=args.acic_n_datasets
        )
        all_results.append(df_acic)
        print("\nACIC Summary:")
        print(summarize_results(df_acic).to_string(index=False))

    # Save all benchmark results to unified file
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        csv_path = output.csv_path("benchmark_results")
        df_all.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    # Generate summary table
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        summary = summarize_results(df_all)
        df_to_latex_table(
            summary,
            output.table_path("tab_benchmarks"),
            caption="Benchmark results (mean and std across seeds/replicates).",
            label="tab:benchmarks",
        )


if __name__ == "__main__":
    main()
