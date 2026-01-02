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
from src.models.dragonnet import MyDragonNet, get_device
from src.utils.io import get_output_manager
from src.utils.metrics import compute_pehe
from src.utils.latex import MacroGenerator


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

    if FAST_RUN:
        ctrl_config.plugin_epochs = 100
        ctrl_config.nuisance_epochs = 80
        ctrl_config.tau_epochs = 150

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

            # Train plugin (CF-style)
            plugin_model = train_plugin(
                X_train, y_train, T_train,
                use_gating=True,
                seed=seed,
                epochs=ctrl_config.plugin_epochs,
            )
            tau_plugin = predict_tau_tarnet(plugin_model, X_test)
            tau_plugin_train = predict_tau_tarnet(plugin_model, X_train)

            # Train DML (CTRL)
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

            pehe_cf = compute_pehe(tau_plugin, true_te_test)
            pehe_ctrl = compute_pehe(tau_dml, true_te_test)

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
    Uncertainty Quantification experiment using MC Dropout and conformal.
    """
    print("=== Uncertainty Quantification ===")
    rows = []

    ctrl_config = CTRLConfig()
    n_mc_runs = 20 if FAST_RUN else 50
    alpha = 0.05
    cal_size = 100

    for seed in config.seeds:
        print(f"  Seed {seed}...")
        set_seed(seed)

        # Generate data
        X_train, T_train, y_train, _ = get_stress_data(
            n_samples=n_samples,
            n_noise=config.n_noise,
            seed=seed
        )
        X_test, _, _, true_te_test = get_stress_data(
            n_samples=500,
            n_noise=config.n_noise,
            seed=seed + 10000
        )

        # Train model with dropout for MC inference
        model = MyDragonNet(
            input_dim=X_train.shape[1],
            hidden_dim=ctrl_config.hidden_dim,
            use_gating=True,
            dropout_p=0.5,  # Higher dropout for UQ
        ).to(DEVICE)

        # Simple training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        X_t = torch.from_numpy(X_train).float().to(DEVICE)
        T_t = torch.from_numpy(T_train).float().to(DEVICE)
        y_t = torch.from_numpy(y_train).float().to(DEVICE)

        epochs = 100 if FAST_RUN else 200
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            y0, y1, t_logit = model(X_t)
            y_pred = y0 * (1 - T_t.view(-1, 1)) + y1 * T_t.view(-1, 1)
            loss_y = ((y_pred.squeeze() - y_t) ** 2).mean()
            loss_t = torch.nn.functional.binary_cross_entropy_with_logits(t_logit.squeeze(), T_t)
            loss = loss_y + loss_t
            loss.backward()
            optimizer.step()

        # MC Dropout predictions
        X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
        tau_samples = []

        model.train()  # Keep dropout active
        with torch.no_grad():
            for _ in range(n_mc_runs):
                y0, y1, _ = model(X_test_t)
                tau = (y1 - y0).squeeze().cpu().numpy()
                tau_samples.append(tau)

        tau_samples = np.array(tau_samples)  # (n_runs, n_test)
        tau_mean = tau_samples.mean(axis=0)
        tau_std = tau_samples.std(axis=0)

        # MC Dropout intervals (percentile-based)
        tau_lower = np.percentile(tau_samples, 100 * alpha / 2, axis=0)
        tau_upper = np.percentile(tau_samples, 100 * (1 - alpha / 2), axis=0)

        mc_coverage = np.mean((true_te_test >= tau_lower) & (true_te_test <= tau_upper))
        mc_width = np.mean(tau_upper - tau_lower)

        # Simple conformal calibration
        cal_idx = np.random.choice(len(X_test), min(cal_size, len(X_test)), replace=False)
        cal_residuals = np.abs(tau_mean[cal_idx] - true_te_test[cal_idx])
        q = np.quantile(cal_residuals, 1 - alpha)

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
    model = MyDragonNet(
        input_dim=X.shape[1],
        hidden_dim=120,
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
        y0, y1, t_logit = model(X_t)
        y_pred = y0 * (1 - T_t.view(-1, 1)) + y1 * T_t.view(-1, 1)
        loss_y = ((y_pred.squeeze() - y_t) ** 2).mean()
        loss_t = torch.nn.functional.binary_cross_entropy_with_logits(t_logit.squeeze(), T_t)
        loss = loss_y + loss_t
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


def plot_scaling(df: pd.DataFrame, output_path: Path):
    """Plot scaling results."""
    summary = df.groupby("n_samples").agg({
        "pehe_cf": ["mean", "std"],
        "pehe_ctrl": ["mean", "std"],
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    x = summary.index.values

    ax.errorbar(x, summary[("pehe_cf", "mean")], yerr=summary[("pehe_cf", "std")],
                label="CF (plug-in)", marker="o", capsize=3)
    ax.errorbar(x, summary[("pehe_ctrl", "mean")], yerr=summary[("pehe_ctrl", "std")],
                label="CTRL-DML", marker="s", capsize=3)

    ax.set_xlabel("Sample Size")
    ax.set_ylabel("PEHE")
    ax.set_title("Scaling Analysis")
    ax.legend()
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_latex_macros(scaling_df: pd.DataFrame, uq_df: pd.DataFrame,
                          dynamics_df: pd.DataFrame, gen: MacroGenerator):
    """Generate LaTeX macros."""
    if scaling_df is not None:
        for n in scaling_df["n_samples"].unique():
            mask = scaling_df["n_samples"] == n
            suffix = {500: "FiveHundred", 1000: "OneK", 2000: "TwoK", 5000: "FiveK"}.get(n, str(n))
            gen.add(f"ScaleCfN{suffix}", scaling_df.loc[mask, "pehe_cf"].mean(), "Scaling Results")
            gen.add(f"ScaleCtrlN{suffix}", scaling_df.loc[mask, "pehe_ctrl"].mean(), "Scaling Results")

    if uq_df is not None:
        gen.add("UqMcCov", uq_df["mc_coverage"].mean(), "Uncertainty Quantification")
        gen.add("UqMcWidth", uq_df["mc_width"].mean(), "Uncertainty Quantification")
        gen.add("UqConfCov", uq_df["conf_coverage"].mean(), "Uncertainty Quantification")
        gen.add("UqConfWidth", uq_df["conf_width"].mean(), "Uncertainty Quantification")

    if dynamics_df is not None and len(dynamics_df) > 0:
        last = dynamics_df.iloc[-1]
        gen.add("DynConfWeight", last["conf_weight"], "Training Dynamics")
        gen.add("DynInstWeight", last["inst_weight"], "Training Dynamics")
        gen.add("DynNoiseWeight", last["noise_weight"], "Training Dynamics")
        gen.add("DynDecayEpoch", int(last["epoch"]), "Training Dynamics")


def main():
    parser = argparse.ArgumentParser(description="CTRL-DML Synthetic Experiments")
    parser.add_argument("--analysis", choices=["scaling", "uq", "dynamics", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-noise", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=2000)
    args = parser.parse_args()

    config = SyntheticConfig(seeds=args.seeds, n_noise=args.n_noise)
    output = get_output_manager()
    gen = MacroGenerator()

    scaling_df = None
    uq_df = None
    dynamics_df = None

    if args.analysis in ["scaling", "all"]:
        scaling_df = run_scaling(config)
        csv_path = output.csv_path("scaling_dml")
        scaling_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        plot_scaling(scaling_df, output.figure_path("scaling_plot"))

    if args.analysis in ["uq", "all"]:
        uq_df = run_uq(config, n_samples=args.n_samples)
        csv_path = output.csv_path("uq_metrics")
        uq_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    if args.analysis in ["dynamics", "all"]:
        dynamics_df = run_training_dynamics(config, n_samples=args.n_samples)
        csv_path = output.csv_path("training_dynamics")
        dynamics_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    generate_latex_macros(scaling_df, uq_df, dynamics_df, gen)


if __name__ == "__main__":
    main()
