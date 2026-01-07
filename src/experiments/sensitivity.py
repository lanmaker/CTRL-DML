"""
CTRL-DML Sensitivity Sweep.

Sweep sparsity (lambda) and dropout to generate a PEHE heatmap.

Usage:
    python -m src.experiments.sensitivity
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
    resolve_gate_flags,
)
from src.utils.io import get_output_manager
from src.utils.metrics import compute_pehe


FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity sweep."""
    seeds: List[int] = None
    n_samples: int = 2000
    n_noise: int = 50
    lambda_grid: List[float] = None
    dropout_grid: List[float] = None

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42] if FAST_RUN else [42, 7, 1024]
        if self.lambda_grid is None:
            self.lambda_grid = [0.0, 0.05, 0.1] if FAST_RUN else [0.0, 0.03, 0.08, 0.12]
        if self.dropout_grid is None:
            self.dropout_grid = [0.0, 0.2, 0.4] if FAST_RUN else [0.0, 0.1, 0.2, 0.3]


def run_sensitivity(config: SensitivityConfig) -> pd.DataFrame:
    """Run sparsity/dropout sweep and collect PEHE."""
    rows = []
    base_config = CTRLConfig()
    if FAST_RUN:
        base_config.plugin_epochs = 100
        base_config.nuisance_epochs = 80
        base_config.tau_epochs = 150
        base_config.k_folds = min(3, base_config.k_folds)
    else:
        # Keep sweep tractable without changing other experiments.
        base_config.plugin_epochs = min(base_config.plugin_epochs, 160)
        base_config.nuisance_epochs = min(base_config.nuisance_epochs, 120)
        base_config.tau_epochs = min(base_config.tau_epochs, 240)

    plugin_gating, nuisance_gating, tau_gating = resolve_gate_flags(base_config)
    warm_start_backbone = plugin_gating == tau_gating

    for lambda_sparsity in config.lambda_grid:
        for dropout_p in config.dropout_grid:
            for seed in config.seeds:
                print(f"  lambda={lambda_sparsity}, dropout={dropout_p}, seed={seed}...")
                set_seed(seed)

                X_train, T_train, y_train, _ = get_stress_data(
                    n_samples=config.n_samples,
                    n_noise=config.n_noise,
                    seed=seed,
                )
                X_test, _, _, true_te_test = get_stress_data(
                    n_samples=config.n_samples,
                    n_noise=config.n_noise,
                    seed=seed + 10000,
                )

                plugin_model = train_plugin(
                    X_train, y_train, T_train,
                    use_gating=plugin_gating,
                    lambda_sparsity=lambda_sparsity,
                    seed=seed,
                    dropout_p=dropout_p,
                    hidden_dim=base_config.hidden_tau,
                    batch_size=base_config.batch_size,
                    epochs=base_config.plugin_epochs,
                )
                tau_plugin_train = predict_tau_tarnet(plugin_model, X_train)

                m_hat, e_hat = cross_fit_nuisance(
                    X_train, y_train, T_train,
                    use_gating=nuisance_gating,
                    lambda_sparsity=lambda_sparsity,
                    seed=seed,
                    k_folds=base_config.k_folds,
                    dropout_p=dropout_p,
                    hidden_dim=base_config.hidden_dim,
                    batch_size=base_config.batch_size,
                    epochs=base_config.nuisance_epochs,
                )
                e_hat = np.clip(e_hat, 0.01, 0.99)
                R = y_train - m_hat
                W = T_train - e_hat
                Z, weights = stabilize_residuals(
                    R, W,
                    w_clip=base_config.w_clip,
                    z_clip=base_config.z_clip,
                    eps=base_config.eps,
                    w_clip_quantile=base_config.w_clip_quantile,
                    z_clip_quantile=base_config.z_clip_quantile,
                )

                tau_model = train_rlearner(
                    X_train, Z, weights,
                    use_gating=tau_gating,
                    lambda_tau=base_config.lambda_tau,
                    seed=seed,
                    dropout_p=dropout_p,
                    hidden_dim=base_config.hidden_tau,
                    batch_size=base_config.batch_size,
                    epochs=base_config.tau_epochs,
                    lr=base_config.lr_tau,
                    grad_clip=base_config.grad_clip,
                    warm_start_from=plugin_model,
                    warm_start_backbone=warm_start_backbone,
                    teacher_tau=tau_plugin_train,
                    aux_beta_start=base_config.aux_beta_start,
                    aux_beta_end=base_config.aux_beta_end,
                    aux_decay_epochs=base_config.aux_decay_epochs,
                )
                tau_dml_test = predict_tau_rlearner(tau_model, X_test)

                rows.append({
                    "lambda_sparsity": lambda_sparsity,
                    "dropout_p": dropout_p,
                    "seed": seed,
                    "pehe_dml": compute_pehe(tau_dml_test, true_te_test),
                })

    return pd.DataFrame(rows)


def plot_sensitivity_heatmap(df: pd.DataFrame, output_path, lambda_grid: List[float], dropout_grid: List[float]) -> None:
    """Plot PEHE heatmap across sparsity and dropout."""
    summary = df.groupby(["lambda_sparsity", "dropout_p"])["pehe_dml"].mean().reset_index()
    table = summary.pivot(index="lambda_sparsity", columns="dropout_p", values="pehe_dml")
    table = table.reindex(index=lambda_grid, columns=dropout_grid)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(table.values, origin="lower", aspect="auto")
    ax.set_xticks(range(len(dropout_grid)))
    ax.set_xticklabels([f"{v:.2f}" for v in dropout_grid])
    ax.set_yticks(range(len(lambda_grid)))
    ax.set_yticklabels([f"{v:.2f}" for v in lambda_grid])
    ax.set_xlabel("Dropout p")
    ax.set_ylabel("Sparsity lambda")
    ax.set_title("Sensitivity Heatmap (PEHE)")
    fig.colorbar(im, ax=ax, label="PEHE")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CTRL-DML Sensitivity Sweep")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-noise", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=None)
    parser.add_argument("--dropout-grid", type=float, nargs="+", default=None)
    args = parser.parse_args()

    config = SensitivityConfig(
        seeds=args.seeds,
        n_samples=args.n_samples,
        n_noise=args.n_noise,
        lambda_grid=args.lambda_grid,
        dropout_grid=args.dropout_grid,
    )
    output = get_output_manager()

    df = run_sensitivity(config)
    csv_path = output.csv_path("sensitivity_heatmap")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    plot_sensitivity_heatmap(df, output.figure_path("sensitivity_heatmap"), config.lambda_grid, config.dropout_grid)


if __name__ == "__main__":
    main()
