"""
Cross-fitting ablation for CTRL-DML.

Compares orthogonal training with cross-fitted nuisances vs. in-sample nuisances.
Outputs:
  - output/tables/crossfit_ablation.csv
  - output/figures/crossfit_ablation.pdf
"""
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data.synthetic import get_stress_data
from src.models.orthogonal_learner import (
    train_plugin,
    train_nuisance,
    cross_fit_nuisance,
    stabilize_residuals,
    train_rlearner,
    predict_tau_tarnet,
    predict_tau_rlearner,
    set_seed,
    CTRLConfig,
)
from src.models.dragonnet import get_device
from src.utils.io import get_output_manager
from src.utils.metrics import compute_pehe


FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"
DEVICE = get_device()


@dataclass
class CrossfitConfig:
    """Configuration for cross-fitting ablation."""
    seeds: List[int] = None
    n_samples: int = 2000 if not FAST_RUN else 1000
    n_noise: int = 50

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 7] if FAST_RUN else [42, 7, 1024]


def _fit_nuisance_full(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    cfg: CTRLConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit nuisance network on full data and return in-sample m_hat, e_hat."""
    model = train_nuisance(
        X, Y, T,
        use_gating=cfg.use_gating,
        lambda_sparsity=cfg.lambda_sparsity,
        seed=seed,
        dropout_p=cfg.dropout_p,
        hidden_dim=cfg.hidden_dim,
        batch_size=cfg.batch_size,
        epochs=cfg.nuisance_epochs if not FAST_RUN else 80,
    )
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(DEVICE)
        y0, y1, t_prob = model(x_t)
        t_prob_np = np.clip(t_prob.squeeze().cpu().numpy(), 0.01, 0.99)
        y0_np = y0.squeeze().cpu().numpy()
        y1_np = y1.squeeze().cpu().numpy()
        m_hat = t_prob_np * y1_np + (1 - t_prob_np) * y0_np
        e_hat = t_prob_np
    return m_hat, e_hat


def run_crossfit_ablation(config: CrossfitConfig) -> pd.DataFrame:
    """Run cross-fitting ablation and return per-seed results."""
    rows = []
    cfg = CTRLConfig()
    if FAST_RUN:
        cfg.plugin_epochs = 100
        cfg.nuisance_epochs = 80
        cfg.tau_epochs = 150
        cfg.k_folds = min(3, cfg.k_folds)

    for seed in config.seeds:
        print(f"Seed {seed}...")
        set_seed(seed)

        X_train, T_train, Y_train, _ = get_stress_data(
            n_samples=config.n_samples,
            n_noise=config.n_noise,
            seed=seed,
        )
        X_test, _, _, true_te_test = get_stress_data(
            n_samples=config.n_samples,
            n_noise=config.n_noise,
            seed=seed + 10000,
        )

        # Plugin warm start (shared across variants).
        plugin_model = train_plugin(
            X_train, Y_train, T_train,
            use_gating=cfg.use_gating,
            lambda_sparsity=cfg.lambda_sparsity,
            seed=seed,
            dropout_p=cfg.dropout_p,
            hidden_dim=cfg.hidden_tau,
            batch_size=cfg.batch_size,
            epochs=cfg.plugin_epochs if not FAST_RUN else 100,
        )
        tau_plugin_train = predict_tau_tarnet(plugin_model, X_train)
        tau_plugin_test = predict_tau_tarnet(plugin_model, X_test)
        pehe_plugin = compute_pehe(tau_plugin_test, true_te_test)

        # Cross-fitted nuisances.
        m_cf, e_cf = cross_fit_nuisance(
            X_train, Y_train, T_train,
            use_gating=cfg.use_gating,
            lambda_sparsity=cfg.lambda_sparsity,
            seed=seed,
            k_folds=cfg.k_folds,
            dropout_p=cfg.dropout_p,
            hidden_dim=cfg.hidden_dim,
            batch_size=cfg.batch_size,
            epochs=cfg.nuisance_epochs,
        )
        e_cf = np.clip(e_cf, 0.01, 0.99)
        Z_cf, w_cf = stabilize_residuals(Y_train - m_cf, T_train - e_cf,
                                         w_clip=cfg.w_clip, z_clip=cfg.z_clip, eps=cfg.eps)
        tau_cf_model = train_rlearner(
            X_train, Z_cf, w_cf,
            use_gating=cfg.use_gating,
            lambda_tau=cfg.lambda_tau,
            seed=seed,
            dropout_p=cfg.dropout_p,
            hidden_dim=cfg.hidden_tau,
            batch_size=cfg.batch_size,
            epochs=cfg.tau_epochs,
            lr=cfg.lr_tau,
            grad_clip=cfg.grad_clip,
            warm_start_from=plugin_model,
            teacher_tau=tau_plugin_train,
            aux_beta_start=cfg.aux_beta_start,
            aux_beta_end=cfg.aux_beta_end,
            aux_decay_epochs=cfg.aux_decay_epochs,
        )
        tau_cf = predict_tau_rlearner(tau_cf_model, X_test)
        pehe_cf = compute_pehe(tau_cf, true_te_test)

        # In-sample nuisances (no cross-fitting).
        m_nf, e_nf = _fit_nuisance_full(X_train, Y_train, T_train, cfg, seed)
        Z_nf, w_nf = stabilize_residuals(Y_train - m_nf, T_train - e_nf,
                                         w_clip=cfg.w_clip, z_clip=cfg.z_clip, eps=cfg.eps)
        tau_nf_model = train_rlearner(
            X_train, Z_nf, w_nf,
            use_gating=cfg.use_gating,
            lambda_tau=cfg.lambda_tau,
            seed=seed,
            dropout_p=cfg.dropout_p,
            hidden_dim=cfg.hidden_tau,
            batch_size=cfg.batch_size,
            epochs=cfg.tau_epochs,
            lr=cfg.lr_tau,
            grad_clip=cfg.grad_clip,
            warm_start_from=plugin_model,
            teacher_tau=tau_plugin_train,
            aux_beta_start=cfg.aux_beta_start,
            aux_beta_end=cfg.aux_beta_end,
            aux_decay_epochs=cfg.aux_decay_epochs,
        )
        tau_nf = predict_tau_rlearner(tau_nf_model, X_test)
        pehe_nf = compute_pehe(tau_nf, true_te_test)

        rows.append({
            "seed": seed,
            "pehe_crossfit": pehe_cf,
            "pehe_nocf": pehe_nf,
            "pehe_plugin": pehe_plugin,
        })
        print(f"  Plugin={pehe_plugin:.3f}, Cross-fit={pehe_cf:.3f}, No-CF={pehe_nf:.3f}")

    return pd.DataFrame(rows)


def plot_crossfit_ablation(df: pd.DataFrame, output_path) -> None:
    """Plot cross-fitting ablation results."""
    summary = df[["pehe_crossfit", "pehe_nocf"]].agg(["mean", "std"])
    labels = ["Cross-fit", "Shared folds"]
    means = [summary.loc["mean", "pehe_crossfit"], summary.loc["mean", "pehe_nocf"]]
    stds = [summary.loc["std", "pehe_crossfit"], summary.loc["std", "pehe_nocf"]]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.bar(x, means, yerr=stds, capsize=3, color="#7a0f0f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PEHE")
    ax.set_title("Cross-fitting Ablation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-fitting ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=2000 if not FAST_RUN else 1000)
    parser.add_argument("--n-noise", type=int, default=50)
    args = parser.parse_args()

    config = CrossfitConfig(
        seeds=args.seeds,
        n_samples=args.n_samples,
        n_noise=args.n_noise,
    )
    output = get_output_manager()

    df = run_crossfit_ablation(config)
    csv_path = output.csv_path("crossfit_ablation")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    plot_crossfit_ablation(df, output.figure_path("crossfit_ablation"))

    print("\nTo regenerate LaTeX macros, run: python -m src.utils.latex")


if __name__ == "__main__":
    main()
