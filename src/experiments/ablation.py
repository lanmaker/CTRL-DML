"""
CTRL-DML Ablation Experiments.

This module consolidates:
- run_ablation.py (core training)
- run_plot_ablation.py (visualization)

Usage:
    python -m src.experiments.ablation --mode train
    python -m src.experiments.ablation --mode plot
    python -m src.experiments.ablation --mode all
"""
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

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
)
from src.utils.io import get_output_manager
from src.utils.metrics import compute_pehe
from src.utils.latex import df_to_latex_table, MacroGenerator


# Default hyperparameters
FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"
DEFAULT_N_SAMPLES = 1200 if FAST_RUN else 2000
DEFAULT_EPOCHS_PLUGIN = 100 if FAST_RUN else 220
DEFAULT_EPOCHS_NUISANCE = 90 if FAST_RUN else 150
DEFAULT_EPOCHS_TAU = 180 if FAST_RUN else 300
DEFAULT_BATCH_SIZE = 160 if FAST_RUN else 192
DEFAULT_HIDDEN_DIM = 96 if FAST_RUN else 120
DEFAULT_HIDDEN_TAU = 64 if FAST_RUN else 96
DEFAULT_LAMBDA_TAU = 5e-4


@dataclass
class AblationVariant:
    """Configuration for an ablation variant."""
    name: str
    use_gating: bool
    lambda_sparsity: float


# Standard ablation variants
ABLATION_VARIANTS = [
    AblationVariant("no_gating", False, 0.0),
    AblationVariant("gating_no_L1", True, 0.0),
    AblationVariant("gating_L1", True, 0.05),
]


def run_single_experiment(
    X: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    true_te: np.ndarray,
    variant: AblationVariant,
    seed: int,
    config: dict,
) -> dict:
    """
    Run a single ablation experiment.

    Returns:
        Dictionary with results
    """
    # Stage 0: plug-in pretrain
    plugin_model = train_plugin(
        X, y, T,
        use_gating=variant.use_gating,
        lambda_sparsity=variant.lambda_sparsity,
        seed=seed,
        dropout_p=config["dropout_p"],
        hidden_dim=config["hidden_tau"],
        batch_size=config["batch_size"],
        epochs=config["plugin_epochs"],
        lr=config["plugin_lr"],
    )
    tau_plugin = predict_tau_tarnet(plugin_model, X)

    # Stage 1: Cross-fit nuisances
    m_hat, e_hat = cross_fit_nuisance(
        X, y, T,
        use_gating=variant.use_gating,
        lambda_sparsity=variant.lambda_sparsity,
        seed=seed,
        k_folds=config["k_folds"],
        dropout_p=config["dropout_p"],
        hidden_dim=config["hidden_dim"],
        batch_size=config["batch_size"],
        epochs=config["epochs_nuisance"],
    )
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R = y - m_hat
    W = T - e_hat

    if config["standardize_w"]:
        Z, weights = stabilize_residuals(R, W, clip_w=config["clip_w"], z_clip=config["z_clip"])
    else:
        Z, weights = R, np.clip(W, -config["clip_w"], config["clip_w"]) ** 2

    # Stage 2: Orthogonal R-learner
    tau_model = train_rlearner(
        X, Z, weights,
        use_gating=variant.use_gating,
        lambda_tau=config["lambda_tau"],
        seed=seed,
        dropout_p=config["dropout_p"],
        hidden_dim=config["hidden_tau"],
        batch_size=config["batch_size"],
        epochs=config["epochs_tau"],
        lr=config["lr"],
        grad_clip=config["grad_clip"],
        warm_start_from=plugin_model,
        teacher_tau=tau_plugin,
        aux_beta_start=config["aux_beta_start"],
        aux_beta_end=config["aux_beta_end"],
        aux_decay_epochs=config["aux_decay_epochs"],
        freeze_backbone=config["freeze_backbone"],
    )
    tau_pred = predict_tau_rlearner(tau_model, X)

    # Evaluate
    pehe_orth = compute_pehe(tau_pred, true_te)
    pehe_plugin = compute_pehe(tau_plugin, true_te)

    return {
        "variant": variant.name,
        "use_gating": int(variant.use_gating),
        "lambda_sparsity": variant.lambda_sparsity,
        "seed": seed,
        "pehe_dml": pehe_orth,
        "pehe_plugin": pehe_plugin,
    }


def run_ablation(config: dict) -> pd.DataFrame:
    """
    Run full ablation study.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame with results
    """
    rows = []

    for variant in ABLATION_VARIANTS:
        for seed in config["seeds"]:
            print(f"\n>>> Variant={variant.name}, seed={seed}")

            # Generate data
            X, T, y, true_te = get_stress_data(
                n_samples=config["n_samples"],
                n_noise=config["n_noise"],
                seed=seed
            )

            # Run experiment
            result = run_single_experiment(X, y, T, true_te, variant, seed, config)
            print(f"PEHE | DML: {result['pehe_dml']:.3f} | Plug-in: {result['pehe_plugin']:.3f}")
            rows.append(result)

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize results by variant."""
    summary = df.groupby("variant").agg({
        "pehe_dml": ["mean", "std"],
        "pehe_plugin": ["mean", "std"],
    }).round(4)
    summary.columns = ["_".join(col) for col in summary.columns]
    return summary.reset_index()


def plot_ablation(df: pd.DataFrame, output_path: Path):
    """Create ablation bar plot."""
    summary = df.groupby("variant").agg({
        "pehe_dml": ["mean", "std"],
        "pehe_plugin": ["mean", "std"],
    })

    variants = summary.index.tolist()
    dml_means = summary[("pehe_dml", "mean")].values
    dml_stds = summary[("pehe_dml", "std")].values
    plugin_means = summary[("pehe_plugin", "mean")].values
    plugin_stds = summary[("pehe_plugin", "std")].values

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, dml_means, width, yerr=dml_stds, label="DML (orthogonal)", capsize=3)
    ax.bar(x + width/2, plugin_means, width, yerr=plugin_stds, label="Plug-in (no DML)", capsize=3)

    ax.set_ylabel("PEHE")
    ax.set_xlabel("Variant")
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in variants])
    ax.legend()
    ax.set_title("Ablation Study: DML vs Plug-in")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def generate_latex_assets(df: pd.DataFrame, output_manager):
    """Generate LaTeX table and macros from results."""
    # Summary table
    summary = summarize_results(df)

    # Create LaTeX table
    table_df = pd.DataFrame({
        "Variant": summary["variant"].str.replace("_", " ").str.title(),
        "DML PEHE": summary["pehe_dml_mean"].round(2),
        "Plug-in PEHE": summary["pehe_plugin_mean"].round(2),
    })

    df_to_latex_table(
        table_df,
        output_manager.table_path("tab_ablation"),
        caption="Ablation results (mean PEHE across seeds).",
        label="tab:ablation",
    )

    # Generate macros
    gen = MacroGenerator()
    for _, row in summary.iterrows():
        variant = row["variant"].replace("_", "").title().replace(" ", "")
        gen.add(f"Abl{variant}Dml", row["pehe_dml_mean"], "Ablation Results")
        gen.add(f"Abl{variant}Plugin", row["pehe_plugin_mean"], "Ablation Results")

    # Add to global macros (will be merged later)
    return gen


def main():
    parser = argparse.ArgumentParser(description="CTRL-DML Ablation Experiments")
    parser.add_argument("--mode", choices=["train", "plot", "all"], default="all",
                        help="Run mode: train, plot, or all")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 7] if FAST_RUN else [42, 1024, 2023])
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--n-noise", type=int, default=50)
    parser.add_argument("--k-folds", type=int, default=3)
    parser.add_argument("--plugin-epochs", type=int, default=DEFAULT_EPOCHS_PLUGIN)
    parser.add_argument("--epochs-nuisance", type=int, default=DEFAULT_EPOCHS_NUISANCE)
    parser.add_argument("--epochs-tau", type=int, default=DEFAULT_EPOCHS_TAU)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--hidden-tau", type=int, default=DEFAULT_HIDDEN_TAU)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lambda-tau", type=float, default=DEFAULT_LAMBDA_TAU)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--plugin-lr", type=float, default=0.003)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--clip-w", type=float, default=0.05)
    parser.add_argument("--z-clip", type=float, default=10.0)
    parser.add_argument("--aux-beta-start", type=float, default=0.1)
    parser.add_argument("--aux-beta-end", type=float, default=0.0)
    parser.add_argument("--aux-decay-epochs", type=int, default=50)
    parser.add_argument("--standardize-w", action="store_true", default=True)
    parser.add_argument("--freeze-backbone", action="store_true", default=False)

    args = parser.parse_args()

    config = {
        "seeds": args.seeds,
        "n_samples": args.n_samples,
        "n_noise": args.n_noise,
        "k_folds": args.k_folds,
        "plugin_epochs": args.plugin_epochs,
        "epochs_nuisance": args.epochs_nuisance,
        "epochs_tau": args.epochs_tau,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden,
        "hidden_tau": args.hidden_tau,
        "dropout_p": args.dropout,
        "lambda_tau": args.lambda_tau,
        "lr": args.lr,
        "plugin_lr": args.plugin_lr,
        "grad_clip": args.grad_clip,
        "clip_w": args.clip_w,
        "z_clip": args.z_clip,
        "aux_beta_start": args.aux_beta_start,
        "aux_beta_end": args.aux_beta_end,
        "aux_decay_epochs": args.aux_decay_epochs,
        "standardize_w": args.standardize_w,
        "freeze_backbone": args.freeze_backbone,
    }

    output = get_output_manager()

    if args.mode in ["train", "all"]:
        print("=== Running Ablation Study ===")
        df = run_ablation(config)

        # Save CSV
        csv_path = output.csv_path("ablation_results")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved results: {csv_path}")

        # Print summary
        print("\n=== Summary ===")
        summary = summarize_results(df)
        print(summary.to_string(index=False))

        # Generate LaTeX assets
        generate_latex_assets(df, output)

    if args.mode in ["plot", "all"]:
        csv_path = output.csv_path("ablation_results")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            plot_ablation(df, output.figure_path("ablation_plot"))
        else:
            print(f"No results found at {csv_path}. Run with --mode train first.")


if __name__ == "__main__":
    main()
