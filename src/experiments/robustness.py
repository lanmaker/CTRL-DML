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
import matplotlib.pyplot as plt

from src.data.synthetic import get_stress_data, get_feature_roles
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
                use_gating=ctrl_config.use_gating,
                lambda_sparsity=ctrl_config.lambda_sparsity,
                seed=seed,
                dropout_p=ctrl_config.dropout_p,
                hidden_dim=ctrl_config.hidden_tau,
                batch_size=ctrl_config.batch_size,
                epochs=ctrl_config.plugin_epochs,
            )
            tau_plugin_test = predict_tau_tarnet(plugin_model, X_test)
            tau_plugin_train = predict_tau_tarnet(plugin_model, X_train)

            # Cross-fit nuisances
            m_hat, e_hat = cross_fit_nuisance(
                X_train, y_train, T_train,
                use_gating=ctrl_config.use_gating,
                lambda_sparsity=ctrl_config.lambda_sparsity,
                seed=seed,
                k_folds=ctrl_config.k_folds,
                dropout_p=ctrl_config.dropout_p,
                hidden_dim=hidden,
                batch_size=ctrl_config.batch_size,
                epochs=epochs,
            )
            e_hat = np.clip(e_hat, 0.01, 0.99)

            # Compute pseudo-outcomes
            R = y_train - m_hat
            W = T_train - e_hat
            Z, weights = stabilize_residuals(R, W, w_clip=ctrl_config.w_clip, z_clip=ctrl_config.z_clip, eps=ctrl_config.eps)

            # Train R-learner
            tau_model = train_rlearner(
                X_train, Z, weights,
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
                teacher_tau=tau_plugin_train,
                aux_beta_start=ctrl_config.aux_beta_start,
                aux_beta_end=ctrl_config.aux_beta_end,
                aux_decay_epochs=ctrl_config.aux_decay_epochs,
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
    Missing-confounder stress test.

    Compares plugin vs orthogonal estimators when confounders are present
    versus removed from observed features (DGP unchanged).
    """
    print("=== Missing Confounders ===")
    rows = []

    ctrl_config = CTRLConfig()
    if FAST_RUN:
        ctrl_config.plugin_epochs = 100
        ctrl_config.nuisance_epochs = 80
        ctrl_config.tau_epochs = 150

    for seed in config.seeds:
        print(f"  Seed {seed}...")
        set_seed(seed)

        # Generate data with confounders present (DGP fixed).
        X_train_full, T_train, y_train, _ = get_stress_data(
            n_samples=config.n_samples,
            n_noise=config.n_noise,
            n_confounders=5,
            seed=seed
        )
        X_test_full, _, _, true_te_test = get_stress_data(
            n_samples=config.n_samples,
            n_noise=config.n_noise,
            n_confounders=5,
            seed=seed + 10000
        )

        role_idx = get_feature_roles(
            n_confounders=5,
            n_instruments=5,
            n_prognostic=5,
            n_noise=config.n_noise,
        )
        conf_idx = role_idx["confounders"]

        for include_conf in [True, False]:
            conf_label = "with_conf" if include_conf else "no_conf"
            if include_conf:
                X_train = X_train_full
                X_test = X_test_full
            else:
                X_train = np.delete(X_train_full, conf_idx, axis=1)
                X_test = np.delete(X_test_full, conf_idx, axis=1)

            # Train plugin
            plugin_model = train_plugin(
                X_train, y_train, T_train,
                use_gating=ctrl_config.use_gating,
                lambda_sparsity=ctrl_config.lambda_sparsity,
                seed=seed,
                dropout_p=ctrl_config.dropout_p,
                hidden_dim=ctrl_config.hidden_tau,
                batch_size=ctrl_config.batch_size,
                epochs=ctrl_config.plugin_epochs,
            )
            tau_plugin = predict_tau_tarnet(plugin_model, X_test)
            tau_plugin_train = predict_tau_tarnet(plugin_model, X_train)

            # Train DML
            m_hat, e_hat = cross_fit_nuisance(
                X_train, y_train, T_train,
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
            R = y_train - m_hat
            W = T_train - e_hat
            Z, weights = stabilize_residuals(R, W, w_clip=ctrl_config.w_clip, z_clip=ctrl_config.z_clip, eps=ctrl_config.eps)

            tau_model = train_rlearner(
                X_train, Z, weights,
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
                teacher_tau=tau_plugin_train,
                aux_beta_start=ctrl_config.aux_beta_start,
                aux_beta_end=ctrl_config.aux_beta_end,
                aux_decay_epochs=ctrl_config.aux_decay_epochs,
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


def plot_nuisance_misspec(df: pd.DataFrame, output_path: Path) -> None:
    """Plot nuisance misspecification results."""
    summary = df.groupby("nuisance_strength")[["pehe_plugin", "pehe_dml"]].agg(["mean", "std"])
    order = [c for c in ["strong", "weak"] if c in summary.index]
    summary = summary.loc[order]
    x = np.arange(len(summary.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(x - width / 2, summary[("pehe_plugin", "mean")], width,
           yerr=summary[("pehe_plugin", "std")], label="Plug-in", capsize=3)
    ax.bar(x + width / 2, summary[("pehe_dml", "mean")], width,
           yerr=summary[("pehe_dml", "std")], label="CTRL-DML", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(summary.index.tolist())
    ax.set_ylabel("PEHE")
    ax.set_title("Nuisance Misspecification")
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_bias_variance(df: pd.DataFrame, output_path: Path) -> None:
    """Plot missing-confounder results."""
    summary = df.groupby("confounders")[["pehe_plugin", "pehe_dml"]].agg(["mean", "std"])
    order = [c for c in ["with_conf", "no_conf"] if c in summary.index]
    summary = summary.loc[order]
    x = np.arange(len(summary.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(x - width / 2, summary[("pehe_plugin", "mean")], width,
           yerr=summary[("pehe_plugin", "std")], label="Plug-in", capsize=3)
    ax.bar(x + width / 2, summary[("pehe_dml", "mean")], width,
           yerr=summary[("pehe_dml", "std")], label="CTRL-DML", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(["with confounders", "confounders dropped"])
    ax.set_ylabel("PEHE")
    ax.set_title("Missing Confounders")
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_latex_reminder():
    """Print reminder to regenerate LaTeX macros."""
    print("\nTo regenerate LaTeX macros, run: python -m src.utils.latex")


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

    if args.test in ["nuisance", "all"]:
        nuisance_df = run_nuisance_misspecification(config)
        csv_path = output.csv_path("nuisance_misspec")
        nuisance_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        print("\nNuisance Misspecification Summary:")
        print(summarize_results(nuisance_df, "nuisance_strength").to_string(index=False))
        plot_nuisance_misspec(nuisance_df, output.figure_path("nuisance_misspec"))

    if args.test in ["bias_variance", "all"]:
        bv_df = run_bias_variance(config)
        csv_path = output.csv_path("bias_variance")
        bv_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        print("\nBias-Variance Summary:")
        print(summarize_results(bv_df, "confounders").to_string(index=False))
        plot_bias_variance(bv_df, output.figure_path("bias_variance"))

    print_latex_reminder()


if __name__ == "__main__":
    main()
