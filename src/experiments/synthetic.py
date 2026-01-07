"""
CTRL-DML Synthetic Data Experiments.

Runs scaling analysis, uncertainty quantification, and training dynamics.

Usage:
    python -m src.experiments.synthetic --analysis scaling
    python -m src.experiments.synthetic --analysis uq
    python -m src.experiments.synthetic --analysis dynamics
    python -m src.experiments.synthetic --analysis all
"""
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import StratifiedKFold

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
from src.models.dragonnet import MyDragonNet, get_device
from src.utils.io import get_output_manager
from src.utils.metrics import compute_pehe


FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"
DEVICE = get_device()


@dataclass
class SyntheticConfig:
    """Configuration for synthetic experiments."""
    seeds: List[int] = None
    n_noise: int = 50

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 7] if FAST_RUN else [42, 7, 1024]


def run_scaling(config: SyntheticConfig) -> pd.DataFrame:
    """
    Scaling experiment: test performance across sample sizes.
    """
    print("=== Scaling Analysis ===")
    rows = []

    sample_sizes = [500, 1000] if FAST_RUN else [500, 1000, 2000, 5000]
    ctrl_config = CTRLConfig()
    plugin_gating, nuisance_gating, tau_gating = resolve_gate_flags(ctrl_config)
    warm_start_backbone = plugin_gating == tau_gating

    if FAST_RUN:
        ctrl_config.plugin_epochs = 100
        ctrl_config.nuisance_epochs = 80
        ctrl_config.tau_epochs = 150
        ctrl_config.k_folds = min(3, ctrl_config.k_folds)

    for n_samples in sample_sizes:
        for seed in config.seeds:
            print(f"  N={n_samples}, seed={seed}...")
            set_seed(seed)

            # Generate train/test data
            X_train, T_train, y_train, _ = get_stress_data(
                n_samples=n_samples,
                n_noise=config.n_noise,
                seed=seed
            )
            X_test, _, _, true_te_test = get_stress_data(
                n_samples=n_samples,
                n_noise=config.n_noise,
                seed=seed + 10000
            )

            # Train plug-in (warm-start for DML).
            plugin_model = train_plugin(
                X_train, y_train, T_train,
                use_gating=plugin_gating,
                lambda_sparsity=ctrl_config.lambda_sparsity,
                seed=seed,
                dropout_p=ctrl_config.dropout_p,
                hidden_dim=ctrl_config.hidden_tau,
                batch_size=ctrl_config.batch_size,
                epochs=ctrl_config.plugin_epochs,
            )
            tau_plugin = predict_tau_tarnet(plugin_model, X_test)
            tau_plugin_train = predict_tau_tarnet(plugin_model, X_train)

            # Train DML (CTRL)
            m_hat, e_hat = cross_fit_nuisance(
                X_train, y_train, T_train,
                use_gating=nuisance_gating,
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
                w_clip_quantile=ctrl_config.w_clip_quantile,
                z_clip_quantile=ctrl_config.z_clip_quantile,
            )

            tau_model = train_rlearner(
                X_train, Z, weights,
                use_gating=tau_gating,
                lambda_tau=ctrl_config.lambda_tau,
                seed=seed,
                dropout_p=ctrl_config.dropout_p,
                hidden_dim=ctrl_config.hidden_tau,
                batch_size=ctrl_config.batch_size,
                epochs=ctrl_config.tau_epochs,
                lr=ctrl_config.lr_tau,
                grad_clip=ctrl_config.grad_clip,
                warm_start_from=plugin_model,
                warm_start_backbone=warm_start_backbone,
                teacher_tau=tau_plugin_train,
                aux_beta_start=ctrl_config.aux_beta_start,
                aux_beta_end=ctrl_config.aux_beta_end,
                aux_decay_epochs=ctrl_config.aux_decay_epochs,
            )
            tau_dml = predict_tau_rlearner(tau_model, X_test)

            pehe_ctrl = compute_pehe(tau_dml, true_te_test)

            # Causal Forest baseline.
            try:
                from econml.dml import CausalForestDML
            except ImportError as exc:
                raise ImportError("econml is required for the Causal Forest scaling baseline.") from exc
            cf = CausalForestDML(n_estimators=200, random_state=seed)
            cf.fit(y_train, T_train, X=X_train)
            tau_cf = cf.effect(X_test)
            pehe_cf = compute_pehe(tau_cf, true_te_test)

            rows.append({
                "n_samples": n_samples,
                "seed": seed,
                "pehe_cf": pehe_cf,
                "pehe_ctrl": pehe_ctrl,
            })
            print(f"    CF={pehe_cf:.3f}, CTRL={pehe_ctrl:.3f}")

    return pd.DataFrame(rows)


def run_uq(config: SyntheticConfig, n_samples: int = 2000) -> pd.DataFrame:
    """
    Uncertainty Quantification experiment using MC Dropout and cross-conformal calibration.
    """
    print("=== Uncertainty Quantification ===")
    rows = []

    ctrl_config = CTRLConfig()
    if FAST_RUN:
        ctrl_config.k_folds = min(3, ctrl_config.k_folds)
    n_mc_runs = 20 if FAST_RUN else 50
    alpha = 0.05
    epochs = 100 if FAST_RUN else 200

    from src.models.dragonnet import TarNet

    def train_dropout_model(X: np.ndarray, T: np.ndarray, y: np.ndarray, seed: int) -> TarNet:
        set_seed(seed)
        model = TarNet(
            input_dim=X.shape[1],
            hidden=ctrl_config.hidden_dim,
            dropout_p=0.5,
            use_gating=True,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        X_t = torch.from_numpy(X).float().to(DEVICE)
        T_t = torch.from_numpy(T).float().to(DEVICE)
        y_t = torch.from_numpy(y).float().to(DEVICE)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            y0, y1 = model(X_t)
            y_pred = y0 * (1 - T_t.view(-1, 1)) + y1 * T_t.view(-1, 1)
            loss = ((y_pred.squeeze() - y_t) ** 2).mean()
            loss.backward()
            optimizer.step()
        return model

    def mc_dropout_samples(model: TarNet, X_np: np.ndarray, n_runs: int) -> np.ndarray:
        X_t = torch.from_numpy(X_np).float().to(DEVICE)
        samples = []
        model.train()
        with torch.no_grad():
            for _ in range(n_runs):
                y0, y1 = model(X_t)
                tau = (y1 - y0).squeeze().cpu().numpy()
                samples.append(tau)
        return np.array(samples)

    for seed in config.seeds:
        print(f"  Seed {seed}...")
        set_seed(seed)

        # Generate data
        X_train, T_train, y_train, true_te_train = get_stress_data(
            n_samples=n_samples,
            n_noise=config.n_noise,
            seed=seed
        )
        X_test, _, _, true_te_test = get_stress_data(
            n_samples=500,
            n_noise=config.n_noise,
            seed=seed + 10000
        )

        k_folds = min(ctrl_config.k_folds, 5)
        counts = np.bincount(T_train.astype(int), minlength=2)
        min_count = int(np.min(counts))
        if min_count < 2:
            raise ValueError("Need at least two samples per treatment group for conformal UQ.")
        k_folds = min(k_folds, min_count)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        oof_residuals = np.zeros(len(X_train), dtype=np.float32)

        for fold, (tr, val) in enumerate(skf.split(X_train, T_train)):
            model = train_dropout_model(X_train[tr], T_train[tr], y_train[tr], seed + 100 * (fold + 1))
            tau_samples = mc_dropout_samples(model, X_train[val], n_mc_runs)
            tau_mean = tau_samples.mean(axis=0)
            oof_residuals[val] = np.abs(tau_mean - true_te_train[val])

        q = float(np.quantile(oof_residuals, 1 - alpha))

        final_model = train_dropout_model(X_train, T_train, y_train, seed + 999)
        tau_samples = mc_dropout_samples(final_model, X_test, n_mc_runs)
        tau_mean = tau_samples.mean(axis=0)

        # MC Dropout intervals (percentile-based)
        tau_lower = np.percentile(tau_samples, 100 * alpha / 2, axis=0)
        tau_upper = np.percentile(tau_samples, 100 * (1 - alpha / 2), axis=0)

        mc_coverage = np.mean((true_te_test >= tau_lower) & (true_te_test <= tau_upper))
        mc_width = np.mean(tau_upper - tau_lower)

        conf_lower = tau_mean - q
        conf_upper = tau_mean + q
        conf_coverage = np.mean((true_te_test >= conf_lower) & (true_te_test <= conf_upper))
        conf_width = 2 * q

        rows.append({
            "seed": seed,
            "n_samples": n_samples,
            "n_mc_runs": n_mc_runs,
            "alpha": alpha,
            "mc_coverage": mc_coverage,
            "mc_width": mc_width,
            "conf_coverage": conf_coverage,
            "conf_width": conf_width,
        })
        print(f"    MC: cov={mc_coverage:.3f}, width={mc_width:.3f}")
        print(f"    Conformal: cov={conf_coverage:.3f}, width={conf_width:.3f}")

    return pd.DataFrame(rows)


def run_training_dynamics(config: SyntheticConfig, n_samples: int = 2000) -> pd.DataFrame:
    """
    Training dynamics: track mask weights over training.
    """
    print("=== Training Dynamics ===")
    rows = []

    seed = config.seeds[0]
    set_seed(seed)

    # Generate data with known structure
    X, T, y, _ = get_stress_data(
        n_samples=n_samples,
        n_noise=config.n_noise,
        n_confounders=5,
        n_instruments=5,
        n_prognostic=5,
        seed=seed
    )

    # Train model and track mask evolution
    from src.models.dragonnet import TarNet
    model = TarNet(
        input_dim=X.shape[1],
        hidden=120,
        use_gating=True,
        dropout_p=0.4,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.from_numpy(X).float().to(DEVICE)
    T_t = torch.from_numpy(T).float().to(DEVICE)
    y_t = torch.from_numpy(y).float().to(DEVICE)

    epochs = 100 if FAST_RUN else 200
    track_epochs = list(range(0, epochs, 10)) + [epochs - 1]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y0, y1 = model(X_t)
        y_pred = y0 * (1 - T_t.view(-1, 1)) + y1 * T_t.view(-1, 1)
        loss = ((y_pred.squeeze() - y_t) ** 2).mean()
        loss.backward()
        optimizer.step()

        if epoch in track_epochs:
            model.eval()
            with torch.no_grad():
                if hasattr(model.backbone, 'attn') and hasattr(model.backbone.attn, 'mask_net'):
                    mask = model.backbone.attn.mask_net(X_t)
                    mask_weights = mask.mean(dim=0).cpu().numpy()

                    # Average weights for each feature group
                    conf_weight = mask_weights[:5].mean()
                    inst_weight = mask_weights[5:10].mean()
                    prog_weight = mask_weights[10:15].mean()
                    noise_weight = mask_weights[15:].mean()

                    rows.append({
                        "epoch": epoch,
                        "conf_weight": float(conf_weight),
                        "inst_weight": float(inst_weight),
                        "prog_weight": float(prog_weight),
                        "noise_weight": float(noise_weight),
                    })

    return pd.DataFrame(rows)


def plot_scaling(
    df: pd.DataFrame,
    output_path: Path,
    allowed_sizes: Optional[List[int]] = None,
    title: str = "Scaling Analysis",
    log_x: bool = True,
):
    """Plot scaling results."""
    plot_df = df.copy()
    if allowed_sizes is not None:
        plot_df = plot_df[plot_df["n_samples"].isin(allowed_sizes)]

    summary = plot_df.groupby("n_samples").agg({
        "pehe_cf": ["mean", "std"],
        "pehe_ctrl": ["mean", "std"],
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    x = summary.index.values

    ax.errorbar(x, summary[("pehe_cf", "mean")], yerr=summary[("pehe_cf", "std")],
                label="Causal Forest", marker="o", capsize=3)
    ax.errorbar(x, summary[("pehe_ctrl", "mean")], yerr=summary[("pehe_ctrl", "std")],
                label="CTRL-DML", marker="s", capsize=3)

    ax.set_xlabel("Sample Size")
    ax.set_ylabel("PEHE")
    ax.set_title(title)
    ax.legend(frameon=False)
    if log_x:
        ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_training_dynamics(df: pd.DataFrame, output_path: Path) -> None:
    """Plot mask-weight dynamics over training."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["epoch"], df["conf_weight"], label="confounder")
    ax.plot(df["epoch"], df["inst_weight"], label="instrument")
    ax.plot(df["epoch"], df["prog_weight"], label="prognostic")
    ax.plot(df["epoch"], df["noise_weight"], label="noise")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mask weight")
    ax.set_title("Training Dynamics")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_uq_metrics(df: pd.DataFrame, output_path: Path) -> None:
    """Plot MC Dropout vs conformal coverage and width."""
    mc_cov = df["mc_coverage"].mean()
    conf_cov = df["conf_coverage"].mean()
    mc_w = df["mc_width"].mean()
    conf_w = df["conf_width"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].bar(["MC", "Conformal"], [mc_cov, conf_cov], color=["#4C78A8", "#F58518"])
    axes[0].axhline(0.95, linestyle="--", color="gray", linewidth=1)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Coverage")
    axes[0].set_title("Coverage")

    axes[1].bar(["MC", "Conformal"], [mc_w, conf_w], color=["#4C78A8", "#F58518"])
    axes[1].set_ylabel("Avg. width")
    axes[1].set_title("Interval Width")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_latex_reminder():
    """Print reminder to regenerate LaTeX macros."""
    print("\nTo regenerate LaTeX macros, run: python -m src.utils.latex")


def main():
    parser = argparse.ArgumentParser(description="CTRL-DML Synthetic Experiments")
    parser.add_argument("--analysis", choices=["scaling", "uq", "dynamics", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-noise", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=2000)
    args = parser.parse_args()

    config = SyntheticConfig(seeds=args.seeds, n_noise=args.n_noise)
    output = get_output_manager()

    if args.analysis in ["scaling", "all"]:
        scaling_df = run_scaling(config)
        csv_path = output.csv_path("scaling_dml")
        scaling_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        plot_scaling(
            scaling_df,
            output.figure_path("scaling_dml"),
            allowed_sizes=[500, 1000],
            title="Scaling (noise=50)",
            log_x=False,
        )
        plot_scaling(
            scaling_df,
            output.figure_path("scaling_results"),
            title="Scaling Law",
            log_x=True,
        )

    if args.analysis in ["uq", "all"]:
        uq_df = run_uq(config, n_samples=args.n_samples)
        csv_path = output.csv_path("uq_metrics")
        uq_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        plot_uq_metrics(uq_df, output.figure_path("uq_conformal"))

    if args.analysis in ["dynamics", "all"]:
        dynamics_df = run_training_dynamics(config, n_samples=args.n_samples)
        csv_path = output.csv_path("training_dynamics")
        dynamics_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        plot_training_dynamics(dynamics_df, output.figure_path("training_dynamics"))

    print_latex_reminder()


if __name__ == "__main__":
    main()
