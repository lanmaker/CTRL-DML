"""
CTRL-DML Noise Robustness Sweep.

Vary noise dimensionality and compare plug-in vs CTRL-DML PEHE.

Usage:
    python -m src.experiments.robustness_sweep
"""
import argparse
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"


@dataclass
class RobustnessSweepConfig:
    """Configuration for noise robustness sweep."""
    seeds: List[int] = None
    n_samples: int = 2000
    noise_levels: List[int] = None

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 7] if FAST_RUN else [42, 7, 1024]
        if self.noise_levels is None:
            self.noise_levels = [10, 50] if FAST_RUN else [10, 20, 50, 100]


def run_noise_sweep(config: RobustnessSweepConfig) -> pd.DataFrame:
    """Run noise-dimension sweep."""
    rows = []
    ctrl_config = CTRLConfig()
    if FAST_RUN:
        ctrl_config.plugin_epochs = 100
        ctrl_config.nuisance_epochs = 80
        ctrl_config.tau_epochs = 150

    for n_noise in config.noise_levels:
        for seed in config.seeds:
            print(f"  n_noise={n_noise}, seed={seed}...")
            set_seed(seed)

            X_train, T_train, y_train, _ = get_stress_data(
                n_samples=config.n_samples,
                n_noise=n_noise,
                seed=seed,
            )
            X_test, _, _, true_te_test = get_stress_data(
                n_samples=config.n_samples,
                n_noise=n_noise,
                seed=seed + 10000,
            )

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
            Z, weights = stabilize_residuals(
                R, W,
                w_clip=ctrl_config.w_clip,
                z_clip=ctrl_config.z_clip,
                eps=ctrl_config.eps,
            )

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
            tau_dml_test = predict_tau_rlearner(tau_model, X_test)

            rows.append({
                "n_noise": n_noise,
                "seed": seed,
                "pehe_plugin": compute_pehe(tau_plugin_test, true_te_test),
                "pehe_dml": compute_pehe(tau_dml_test, true_te_test),
            })

    return pd.DataFrame(rows)


def plot_noise_sweep(df: pd.DataFrame, output_path) -> None:
    """Plot PEHE vs noise dimensionality."""
    summary = df.groupby("n_noise")[["pehe_plugin", "pehe_dml"]].agg(["mean", "std"])
    x = summary.index.values

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        x, summary[("pehe_plugin", "mean")], yerr=summary[("pehe_plugin", "std")],
        label="Plug-in", marker="o", capsize=3,
    )
    ax.errorbar(
        x, summary[("pehe_dml", "mean")], yerr=summary[("pehe_dml", "std")],
        label="CTRL-DML", marker="s", capsize=3,
    )
    ax.set_xlabel("Noise dimensions")
    ax.set_ylabel("PEHE")
    ax.set_title("Noise Robustness")
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CTRL-DML Noise Robustness Sweep")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--noise-levels", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    config = RobustnessSweepConfig(
        seeds=args.seeds,
        n_samples=args.n_samples,
        noise_levels=args.noise_levels,
    )
    output = get_output_manager()

    df = run_noise_sweep(config)
    csv_path = output.csv_path("robustness_solid")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    plot_noise_sweep(df, output.figure_path("robustness_solid"))


if __name__ == "__main__":
    main()
