"""
CTRL-DML Robustness Experiments.

Tests robustness under nuisance misspecification and bias-variance analysis.

Usage:
    python -m src.experiments.robustness --test nuisance
    python -m src.experiments.robustness --test bias_variance
    python -m src.experiments.robustness --test all
"""
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.synthetic import get_stress_data
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
from src.utils.metrics import compute_pehe
from src.utils.latex import MacroGenerator


FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"


@dataclass
class RobustnessConfig:
    """Configuration for robustness experiments."""
    seeds: List[int] = None
    n_samples: int = 1000
    n_noise: int = 50

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 7] if FAST_RUN else [42, 7, 1024]


def run_nuisance_misspecification(config: RobustnessConfig) -> pd.DataFrame:
    """
    Test robustness to nuisance model misspecification.

    Compares performance when nuisance models are well-specified vs weakly specified.
    """
    print("=== Nuisance Misspecification ===")
    rows = []

    ctrl_config = CTRLConfig()
    if FAST_RUN:
        ctrl_config.plugin_epochs = 100
        ctrl_config.nuisance_epochs = 80
        ctrl_config.tau_epochs = 150

    for seed in config.seeds:
        print(f"  Seed {seed}...")
        set_seed(seed)

        # Generate data
        X_train, T_train, y_train, _ = get_stress_data(
            n_samples=config.n_samples,
            n_noise=config.n_noise,
            seed=seed
        )
        X_test, _, _, true_te_test = get_stress_data(
            n_samples=config.n_samples,
            n_noise=config.n_noise,
            seed=seed + 10000
        )

        for strength in ["strong", "weak"]:
            # Adjust nuisance model complexity
            hidden = ctrl_config.hidden_dim if strength == "strong" else 32
            epochs = ctrl_config.nuisance_epochs if strength == "strong" else 30

            # Train plugin
            plugin_model = train_plugin(
                X_train, y_train, T_train,
                use_gating=True,
                lambda_sparsity=0.05,
                seed=seed,
                hidden_dim=ctrl_config.hidden_tau,
                epochs=ctrl_config.plugin_epochs,
            )
            tau_plugin_test = predict_tau_tarnet(plugin_model, X_test)
            tau_plugin_train = predict_tau_tarnet(plugin_model, X_train)

            # Cross-fit nuisances
            m_hat, e_hat = cross_fit_nuisance(
                X_train, y_train, T_train,
                use_gating=True,
                lambda_sparsity=0.05,
                seed=seed,
                hidden_dim=hidden,
                epochs=epochs,
            )
            e_hat = np.clip(e_hat, 0.01, 0.99)

            # Compute pseudo-outcomes
            R = y_train - m_hat
            W = T_train - e_hat
            Z, weights = stabilize_residuals(R, W)

            # Train R-learner
            tau_model = train_rlearner(
                X_train, Z, weights,
                use_gating=True,
                lambda_tau=ctrl_config.lambda_tau,
                seed=seed,
                hidden_dim=ctrl_config.hidden_tau,
                epochs=ctrl_config.tau_epochs,
                warm_start_from=plugin_model,
                teacher_tau=tau_plugin_train,
            )
            tau_pred_test = predict_tau_rlearner(tau_model, X_test)

            pehe_plugin = compute_pehe(tau_plugin_test, true_te_test)
            pehe_dml = compute_pehe(tau_pred_test, true_te_test)

            rows.append({
                "seed": seed,
                "n_samples": config.n_samples,
                "n_noise": config.n_noise,
                "nuisance_strength": strength,
                "pehe_plugin": pehe_plugin,
                "pehe_dml": pehe_dml,
            })
            print(f"    {strength}: Plugin={pehe_plugin:.3f}, DML={pehe_dml:.3f}")

    return pd.DataFrame(rows)


def run_bias_variance(config: RobustnessConfig) -> pd.DataFrame:
    """
    Bias-variance decomposition experiment.

    Compares plugin vs orthogonal estimators with/without confounders.
    """
    print("=== Bias-Variance Analysis ===")
    rows = []

    ctrl_config = CTRLConfig()
    if FAST_RUN:
        ctrl_config.plugin_epochs = 100
        ctrl_config.nuisance_epochs = 80
        ctrl_config.tau_epochs = 150

    for seed in config.seeds:
        print(f"  Seed {seed}...")
        set_seed(seed)

        for include_conf in [True, False]:
            conf_label = "with_conf" if include_conf else "no_conf"
            n_conf = 5 if include_conf else 0

            # Generate data
            X_train, T_train, y_train, _ = get_stress_data(
                n_samples=config.n_samples,
                n_noise=config.n_noise,
                n_confounders=n_conf,
                seed=seed
            )
            X_test, _, _, true_te_test = get_stress_data(
                n_samples=config.n_samples,
                n_noise=config.n_noise,
                n_confounders=n_conf,
                seed=seed + 10000
            )

            # Train plugin
            plugin_model = train_plugin(
                X_train, y_train, T_train,
                use_gating=True,
                seed=seed,
                epochs=ctrl_config.plugin_epochs,
            )
            tau_plugin = predict_tau_tarnet(plugin_model, X_test)
            tau_plugin_train = predict_tau_tarnet(plugin_model, X_train)

            # Train DML
            m_hat, e_hat = cross_fit_nuisance(
                X_train, y_train, T_train,
                use_gating=True,
                seed=seed,
                epochs=ctrl_config.nuisance_epochs,
            )
            e_hat = np.clip(e_hat, 0.01, 0.99)
            R = y_train - m_hat
            W = T_train - e_hat
            Z, weights = stabilize_residuals(R, W)

            tau_model = train_rlearner(
                X_train, Z, weights,
                use_gating=True,
                seed=seed,
                epochs=ctrl_config.tau_epochs,
                warm_start_from=plugin_model,
                teacher_tau=tau_plugin_train,
            )
            tau_dml = predict_tau_rlearner(tau_model, X_test)

            pehe_plugin = compute_pehe(tau_plugin, true_te_test)
            pehe_dml = compute_pehe(tau_dml, true_te_test)

            rows.append({
                "seed": seed,
                "confounders": conf_label,
                "pehe_plugin": pehe_plugin,
                "pehe_dml": pehe_dml,
            })
            print(f"    {conf_label}: Plugin={pehe_plugin:.3f}, DML={pehe_dml:.3f}")

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Summarize results by group."""
    numeric = ["pehe_plugin", "pehe_dml"]
    numeric = [c for c in numeric if c in df.columns]
    summary = df.groupby(group_col)[numeric].agg(["mean", "std"]).round(4)
    summary.columns = ["_".join(col) for col in summary.columns]
    return summary.reset_index()


def generate_latex_macros(nuisance_df: pd.DataFrame, bv_df: pd.DataFrame, gen: MacroGenerator):
    """Generate LaTeX macros from results."""
    if nuisance_df is not None:
        for strength in ["strong", "weak"]:
            mask = nuisance_df["nuisance_strength"] == strength
            suffix = strength.title()
            gen.add(f"NuisPlugin{suffix}", nuisance_df.loc[mask, "pehe_plugin"].mean(),
                    "Nuisance Misspecification")
            gen.add(f"NuisDml{suffix}", nuisance_df.loc[mask, "pehe_dml"].mean(),
                    "Nuisance Misspecification")

    if bv_df is not None:
        for conf in ["with_conf", "no_conf"]:
            mask = bv_df["confounders"] == conf
            suffix = "".join(w.title() for w in conf.split("_"))
            gen.add(f"BvPlugin{suffix}", bv_df.loc[mask, "pehe_plugin"].mean(), "Bias-Variance")
            gen.add(f"BvOrth{suffix}", bv_df.loc[mask, "pehe_dml"].mean(), "Bias-Variance")


def main():
    parser = argparse.ArgumentParser(description="CTRL-DML Robustness Experiments")
    parser.add_argument("--test", choices=["nuisance", "bias_variance", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-noise", type=int, default=50)
    args = parser.parse_args()

    config = RobustnessConfig(
        seeds=args.seeds,
        n_samples=args.n_samples,
        n_noise=args.n_noise,
    )

    output = get_output_manager()
    gen = MacroGenerator()

    nuisance_df = None
    bv_df = None

    if args.test in ["nuisance", "all"]:
        nuisance_df = run_nuisance_misspecification(config)
        csv_path = output.csv_path("nuisance_misspec")
        nuisance_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        print("\nNuisance Misspecification Summary:")
        print(summarize_results(nuisance_df, "nuisance_strength").to_string(index=False))

    if args.test in ["bias_variance", "all"]:
        bv_df = run_bias_variance(config)
        csv_path = output.csv_path("bias_variance")
        bv_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        print("\nBias-Variance Summary:")
        print(summarize_results(bv_df, "confounders").to_string(index=False))

    generate_latex_macros(nuisance_df, bv_df, gen)


if __name__ == "__main__":
    main()
