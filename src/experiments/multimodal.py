"""
CTRL-DML Multimodal Experiments.

Tests CTRL-DML on synthetic multimodal data (text + tabular confounding).

Usage:
    python -m src.experiments.multimodal --experiment benchmark
    python -m src.experiments.multimodal --experiment noise_sweep
    python -m src.experiments.multimodal --experiment all
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
import torch.nn as nn

from src.data.multimodal import get_multimodal_data, get_combined_features
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
from src.models.dragonnet import get_device
from src.utils.io import get_output_manager
from src.utils.metrics import compute_pehe
from src.utils.latex import MacroGenerator


FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"
DEVICE = get_device()


@dataclass
class MultimodalConfig:
    """Configuration for multimodal experiments."""
    seeds: List[int] = None
    n_samples: int = 2000 if not FAST_RUN else 1000
    vocab_size: int = 1000

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 7] if FAST_RUN else [42, 7, 1024]


class SimpleTextEncoder(nn.Module):
    """Simple text encoder using embedding + pooling."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) of token indices
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        pooled = emb.mean(dim=1)  # (batch, embed_dim)
        return torch.relu(self.fc(pooled))


class MultimodalTarNet(nn.Module):
    """TarNet for multimodal data (tabular + text)."""

    def __init__(
        self,
        tab_dim: int,
        vocab_size: int,
        hidden_dim: int = 96,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.text_encoder = SimpleTextEncoder(vocab_size, embed_dim, hidden_dim // 2)
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        combined_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.head_y0 = nn.Linear(hidden_dim, 1)
        self.head_y1 = nn.Linear(hidden_dim, 1)
        self.head_t = nn.Linear(hidden_dim, 1)

    def forward(self, x_tab: torch.Tensor, x_text: torch.Tensor):
        h_text = self.text_encoder(x_text)
        h_tab = self.tab_encoder(x_tab)
        h = torch.cat([h_tab, h_text], dim=-1)
        h = self.backbone(h)
        return self.head_y0(h), self.head_y1(h), self.head_t(h)


def train_multimodal_tarnet(
    X_tab: np.ndarray,
    X_text: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    vocab_size: int,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
) -> MultimodalTarNet:
    """Train multimodal TarNet."""
    set_seed(seed)

    model = MultimodalTarNet(X_tab.shape[1], vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tab_t = torch.from_numpy(X_tab).float().to(DEVICE)
    X_text_t = torch.from_numpy(X_text).long().to(DEVICE)
    Y_t = torch.from_numpy(Y).float().to(DEVICE)
    T_t = torch.from_numpy(T).float().to(DEVICE)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y0, y1, t_logit = model(X_tab_t, X_text_t)
        y_pred = y0.squeeze() * (1 - T_t) + y1.squeeze() * T_t

        loss_y = ((y_pred - Y_t) ** 2).mean()
        loss_t = nn.functional.binary_cross_entropy_with_logits(t_logit.squeeze(), T_t)
        loss = loss_y + loss_t

        loss.backward()
        optimizer.step()

    return model


def predict_multimodal_tau(model: MultimodalTarNet, X_tab: np.ndarray, X_text: np.ndarray) -> np.ndarray:
    """Predict CATE from multimodal model."""
    model.eval()
    with torch.no_grad():
        X_tab_t = torch.from_numpy(X_tab).float().to(DEVICE)
        X_text_t = torch.from_numpy(X_text).long().to(DEVICE)
        y0, y1, _ = model(X_tab_t, X_text_t)
        tau = (y1 - y0).squeeze().cpu().numpy()
    return tau


def run_benchmark(config: MultimodalConfig) -> pd.DataFrame:
    """
    Run benchmark comparing methods on multimodal data.
    """
    print("=== Multimodal Benchmark ===")
    rows = []

    ctrl_config = CTRLConfig()
    epochs = 100 if FAST_RUN else 200

    for seed in config.seeds:
        print(f"  Seed {seed}...")
        set_seed(seed)

        # Generate data
        X_tab, X_text, Y, T, true_te = get_multimodal_data(
            n=config.n_samples,
            vocab_size=config.vocab_size,
            seed=seed
        )

        # Split train/test
        n = len(Y)
        idx = np.random.permutation(n)
        n_train = int(0.8 * n)
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        X_tab_train, X_tab_test = X_tab[train_idx], X_tab[test_idx]
        X_text_train, X_text_test = X_text[train_idx], X_text[test_idx]
        Y_train, T_train = Y[train_idx], T[train_idx]
        true_te_test = true_te[test_idx]

        # Method 1: Multimodal TarNet (proper fusion)
        mm_model = train_multimodal_tarnet(
            X_tab_train, X_text_train, Y_train, T_train,
            vocab_size=config.vocab_size,
            epochs=epochs,
            seed=seed
        )
        tau_mm = predict_multimodal_tau(mm_model, X_tab_test, X_text_test)
        pehe_mm = compute_pehe(tau_mm, true_te_test)

        # Method 2: Tabular-only CTRL-DML (ignores text)
        plugin_model = train_plugin(
            X_tab_train, Y_train, T_train,
            use_gating=True,
            seed=seed,
            epochs=ctrl_config.plugin_epochs if not FAST_RUN else 100,
        )
        tau_tab = predict_tau_tarnet(plugin_model, X_tab_test)
        pehe_tab = compute_pehe(tau_tab, true_te_test)

        # Method 3: Concatenated features (bag-of-words + tabular)
        X_combined_train = get_combined_features(X_tab_train, X_text_train, config.vocab_size)
        X_combined_test = get_combined_features(X_tab_test, X_text_test, config.vocab_size)

        plugin_concat = train_plugin(
            X_combined_train, Y_train, T_train,
            use_gating=True,
            seed=seed,
            epochs=ctrl_config.plugin_epochs if not FAST_RUN else 100,
        )
        tau_concat = predict_tau_tarnet(plugin_concat, X_combined_test)
        pehe_concat = compute_pehe(tau_concat, true_te_test)

        rows.append({
            "seed": seed,
            "pehe_multimodal": pehe_mm,
            "pehe_tabular_only": pehe_tab,
            "pehe_concat_bow": pehe_concat,
        })
        print(f"    MM={pehe_mm:.3f}, Tab={pehe_tab:.3f}, Concat={pehe_concat:.3f}")

    return pd.DataFrame(rows)


def run_noise_sweep(config: MultimodalConfig) -> pd.DataFrame:
    """
    Sweep over text noise levels to test robustness.
    """
    print("=== Text Noise Sweep ===")
    rows = []

    noise_levels = [0.0, 0.3, 0.6] if not FAST_RUN else [0.0, 0.5]
    epochs = 100 if FAST_RUN else 200

    for p_noise in noise_levels:
        for seed in config.seeds:
            print(f"  p_noise={p_noise}, seed={seed}...")
            set_seed(seed)

            # Generate data with noise
            X_tab, X_text, Y, T, true_te = get_multimodal_data(
                n=config.n_samples,
                vocab_size=config.vocab_size,
                p_noise=p_noise,
                seed=seed
            )

            # Split
            n = len(Y)
            idx = np.random.permutation(n)
            n_train = int(0.8 * n)
            train_idx, test_idx = idx[:n_train], idx[n_train:]

            X_tab_train, X_tab_test = X_tab[train_idx], X_tab[test_idx]
            X_text_train, X_text_test = X_text[train_idx], X_text[test_idx]
            Y_train, T_train = Y[train_idx], T[train_idx]
            true_te_test = true_te[test_idx]

            # Multimodal model
            mm_model = train_multimodal_tarnet(
                X_tab_train, X_text_train, Y_train, T_train,
                vocab_size=config.vocab_size,
                epochs=epochs,
                seed=seed
            )
            tau_mm = predict_multimodal_tau(mm_model, X_tab_test, X_text_test)
            pehe_mm = compute_pehe(tau_mm, true_te_test)

            # Tabular-only baseline
            from src.models.orthogonal_learner import train_plugin, predict_tau_tarnet
            plugin = train_plugin(X_tab_train, Y_train, T_train, use_gating=True, seed=seed, epochs=100)
            tau_tab = predict_tau_tarnet(plugin, X_tab_test)
            pehe_tab = compute_pehe(tau_tab, true_te_test)

            rows.append({
                "p_noise": p_noise,
                "seed": seed,
                "pehe_multimodal": pehe_mm,
                "pehe_tabular": pehe_tab,
            })

    return pd.DataFrame(rows)


def plot_noise_sweep(df: pd.DataFrame, output_path: Path):
    """Plot noise sweep results."""
    summary = df.groupby("p_noise").agg({
        "pehe_multimodal": ["mean", "std"],
        "pehe_tabular": ["mean", "std"],
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    x = summary.index.values

    ax.errorbar(x, summary[("pehe_multimodal", "mean")],
                yerr=summary[("pehe_multimodal", "std")],
                label="Multimodal", marker="o", capsize=3)
    ax.errorbar(x, summary[("pehe_tabular", "mean")],
                yerr=summary[("pehe_tabular", "std")],
                label="Tabular-only", marker="s", capsize=3)

    ax.set_xlabel("Text Noise Probability")
    ax.set_ylabel("PEHE")
    ax.set_title("Robustness to Text Noise")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_latex_macros(benchmark_df: pd.DataFrame, sweep_df: pd.DataFrame, gen: MacroGenerator):
    """Generate LaTeX macros."""
    if benchmark_df is not None:
        gen.add("MmFusionPehe", benchmark_df["pehe_multimodal"].mean(), "Multimodal Results")
        gen.add("MmTabOnlyPehe", benchmark_df["pehe_tabular_only"].mean(), "Multimodal Results")
        gen.add("MmConcatPehe", benchmark_df["pehe_concat_bow"].mean(), "Multimodal Results")

    if sweep_df is not None:
        for p in sweep_df["p_noise"].unique():
            mask = sweep_df["p_noise"] == p
            suffix = str(int(p * 10))
            gen.add(f"MmNoise{suffix}Mm", sweep_df.loc[mask, "pehe_multimodal"].mean(), "Multimodal Results")
            gen.add(f"MmNoise{suffix}Tab", sweep_df.loc[mask, "pehe_tabular"].mean(), "Multimodal Results")


def main():
    parser = argparse.ArgumentParser(description="CTRL-DML Multimodal Experiments")
    parser.add_argument("--experiment", choices=["benchmark", "noise_sweep", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=2000 if not FAST_RUN else 1000)
    args = parser.parse_args()

    config = MultimodalConfig(seeds=args.seeds, n_samples=args.n_samples)
    output = get_output_manager()
    gen = MacroGenerator()

    benchmark_df = None
    sweep_df = None

    if args.experiment in ["benchmark", "all"]:
        benchmark_df = run_benchmark(config)
        csv_path = output.csv_path("multimodal_dml")
        benchmark_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        print("\nBenchmark Summary:")
        print(benchmark_df.describe().round(3))

    if args.experiment in ["noise_sweep", "all"]:
        sweep_df = run_noise_sweep(config)
        csv_path = output.csv_path("multimodal_sweep_summary")
        sweep_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        plot_noise_sweep(sweep_df, output.figure_path("multimodal_noise"))

    generate_latex_macros(benchmark_df, sweep_df, gen)


if __name__ == "__main__":
    main()
