"""
CTRL-DML Multimodal Experiments.

Tests CTRL-DML on synthetic multimodal data (text + tabular confounding).

Usage:
    python -m src.experiments.multimodal --experiment benchmark
    python -m src.experiments.multimodal --experiment noise_sweep
    python -m src.experiments.multimodal --experiment bert
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
from src.models.multimodal import MultimodalCTRL
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


def train_multimodal_ctrl(
    X_tab: np.ndarray,
    X_text: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    vocab_size: int,
    fusion: str = "bag",
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
    lambda_sparsity: float = 0.05,
) -> MultimodalCTRL:
    """Train MultimodalCTRL with bag or cross-attention fusion."""
    set_seed(seed)
    model = MultimodalCTRL(X_tab.shape[1], vocab_size, fusion=fusion).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tab_t = torch.from_numpy(X_tab).float().to(DEVICE)
    X_text_t = torch.from_numpy(X_text).long().to(DEVICE)
    Y_t = torch.from_numpy(Y).float().to(DEVICE)
    T_t = torch.from_numpy(T).float().to(DEVICE)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        y0, y1, t_prob = model(X_tab_t, X_text_t)
        y_pred = y0.squeeze() * (1 - T_t) + y1.squeeze() * T_t
        loss_y = ((y_pred - Y_t) ** 2).mean()
        loss_t = nn.functional.binary_cross_entropy(t_prob.squeeze(), T_t)
        loss = loss_y + loss_t + lambda_sparsity * model.mask_penalty
        loss.backward()
        optimizer.step()

    return model


def predict_multimodal_ctrl_tau(model: MultimodalCTRL, X_tab: np.ndarray, X_text: np.ndarray) -> np.ndarray:
    """Predict CATE from MultimodalCTRL."""
    model.eval()
    with torch.no_grad():
        X_tab_t = torch.from_numpy(X_tab).float().to(DEVICE)
        X_text_t = torch.from_numpy(X_text).long().to(DEVICE)
        y0, y1, _ = model(X_tab_t, X_text_t)
        tau = (y1 - y0).squeeze().cpu().numpy()
    return tau


def run_ctrl_dml_combined(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    T_train: np.ndarray,
    true_te_test: np.ndarray,
    ctrl_config: CTRLConfig,
    seed: int,
) -> Tuple[float, float]:
    """Run CTRL-DML on combined tabular + BoW features."""
    plugin_model = train_plugin(
        X_train, Y_train, T_train,
        use_gating=ctrl_config.use_gating,
        lambda_sparsity=ctrl_config.lambda_sparsity,
        seed=seed,
        dropout_p=ctrl_config.dropout_p,
        hidden_dim=ctrl_config.hidden_tau,
        batch_size=ctrl_config.batch_size,
        epochs=ctrl_config.plugin_epochs if not FAST_RUN else 100,
    )
    tau_plugin_train = predict_tau_tarnet(plugin_model, X_train)
    tau_plugin_test = predict_tau_tarnet(plugin_model, X_test)

    m_hat, e_hat = cross_fit_nuisance(
        X_train, Y_train, T_train,
        use_gating=ctrl_config.use_gating,
        lambda_sparsity=ctrl_config.lambda_sparsity,
        seed=seed,
        k_folds=ctrl_config.k_folds,
        dropout_p=ctrl_config.dropout_p,
        hidden_dim=ctrl_config.hidden_dim,
        batch_size=ctrl_config.batch_size,
        epochs=ctrl_config.nuisance_epochs if not FAST_RUN else 80,
    )
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R = Y_train - m_hat
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
        epochs=ctrl_config.tau_epochs if not FAST_RUN else 150,
        lr=ctrl_config.lr_tau,
        grad_clip=ctrl_config.grad_clip,
        warm_start_from=plugin_model,
        teacher_tau=tau_plugin_train,
        aux_beta_start=ctrl_config.aux_beta_start,
        aux_beta_end=ctrl_config.aux_beta_end,
        aux_decay_epochs=ctrl_config.aux_decay_epochs,
    )
    tau_dml = predict_tau_rlearner(tau_model, X_test)

    pehe_plugin = compute_pehe(tau_plugin_test, true_te_test)
    pehe_dml = compute_pehe(tau_dml, true_te_test)
    return pehe_plugin, pehe_dml


def _tokens_to_text(X_text: np.ndarray) -> List[str]:
    """Convert token indices to whitespace-separated strings."""
    vocab = {0: "neutral", 1: "severe", 2: "mild"}
    texts = []
    for row in X_text:
        tokens = []
        for idx in row.tolist():
            tokens.append(vocab.get(idx, f"tok{idx}"))
        texts.append(" ".join(tokens))
    return texts


def _bert_embeddings(
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 32,
    model_name: str = "distilbert-base-uncased",
) -> np.ndarray:
    """Compute frozen DistilBERT embeddings (CLS token)."""
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError as exc:
        raise ImportError("transformers is required for BERT baselines. Install with pip install transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            outputs = model(**encoded)
            cls = outputs.last_hidden_state[:, 0, :]
            all_embs.append(cls.cpu().numpy())
    return np.vstack(all_embs)


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

        # Method 1: Multimodal TarNet (dense fusion, plug-in)
        mm_model = train_multimodal_tarnet(
            X_tab_train, X_text_train, Y_train, T_train,
            vocab_size=config.vocab_size,
            epochs=epochs,
            seed=seed
        )
        tau_mm = predict_multimodal_tau(mm_model, X_tab_test, X_text_test)
        pehe_plugin = compute_pehe(tau_mm, true_te_test)

        # Method 1b: CTRL bag-of-embeddings
        ctrl_bag = train_multimodal_ctrl(
            X_tab_train, X_text_train, Y_train, T_train,
            vocab_size=config.vocab_size,
            fusion="bag",
            epochs=epochs,
            seed=seed,
            lambda_sparsity=ctrl_config.lambda_sparsity,
        )
        tau_ctrl_bag = predict_multimodal_ctrl_tau(ctrl_bag, X_tab_test, X_text_test)
        pehe_ctrl_bag = compute_pehe(tau_ctrl_bag, true_te_test)

        # Method 1c: Cross-attention CTRL
        ctrl_cross = train_multimodal_ctrl(
            X_tab_train, X_text_train, Y_train, T_train,
            vocab_size=config.vocab_size,
            fusion="cross_attn",
            epochs=epochs,
            seed=seed,
            lambda_sparsity=ctrl_config.lambda_sparsity,
        )
        tau_cross = predict_multimodal_ctrl_tau(ctrl_cross, X_tab_test, X_text_test)
        pehe_cross = compute_pehe(tau_cross, true_te_test)

        # Method 2: Tabular-only CTRL (ignores text)
        plugin_model = train_plugin(
            X_tab_train, Y_train, T_train,
            use_gating=ctrl_config.use_gating,
            lambda_sparsity=ctrl_config.lambda_sparsity,
            seed=seed,
            dropout_p=ctrl_config.dropout_p,
            hidden_dim=ctrl_config.hidden_tau,
            batch_size=ctrl_config.batch_size,
            epochs=ctrl_config.plugin_epochs if not FAST_RUN else 100,
        )
        tau_tab = predict_tau_tarnet(plugin_model, X_tab_test)
        pehe_tab = compute_pehe(tau_tab, true_te_test)

        # Method 3: BoW + tabular baselines (CausalForest + CTRL-DML)
        X_combined_train = get_combined_features(X_tab_train, X_text_train, config.vocab_size)
        X_combined_test = get_combined_features(X_tab_test, X_text_test, config.vocab_size)

        pehe_plugin_bow, pehe_dml = run_ctrl_dml_combined(
            X_combined_train, X_combined_test,
            Y_train, T_train,
            true_te_test,
            ctrl_config,
            seed,
        )

        pehe_cf = np.nan
        try:
            from econml.dml import CausalForestDML
            cf = CausalForestDML(n_estimators=100, random_state=seed)
            cf.fit(Y_train, T_train, X=X_combined_train)
            tau_cf = cf.effect(X_combined_test)
            pehe_cf = compute_pehe(tau_cf, true_te_test)
        except ImportError:
            pass

        rows.append({
            "seed": seed,
            "pehe_plugin": pehe_plugin,
            "pehe_ctrl_bag": pehe_ctrl_bag,
            "pehe_cross_attn": pehe_cross,
            "pehe_dml": pehe_dml,
            "pehe_cf": pehe_cf,
            "pehe_tabular_only": pehe_tab,
            "pehe_multimodal": pehe_plugin,
            "pehe_concat_bow": pehe_plugin_bow,
        })
        print(
            f"    Plugin={pehe_plugin:.3f}, CtrlBag={pehe_ctrl_bag:.3f}, "
            f"CrossAttn={pehe_cross:.3f}, DML={pehe_dml:.3f}, CF={pehe_cf:.3f}"
        )

    return pd.DataFrame(rows)


def run_noise_sweep(config: MultimodalConfig) -> pd.DataFrame:
    """
    Sweep over text noise levels to test robustness.
    """
    print("=== Text Noise Sweep ===")
    rows = []

    noise_levels = [0.0, 0.3, 0.6] if not FAST_RUN else [0.0, 0.5]
    epochs = 100 if FAST_RUN else 200
    ctrl_config = CTRLConfig()

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

            # Multimodal CTRL bag-of-embeddings
            mm_model = train_multimodal_ctrl(
                X_tab_train, X_text_train, Y_train, T_train,
                vocab_size=config.vocab_size,
                fusion="bag",
                epochs=epochs,
                seed=seed,
                lambda_sparsity=ctrl_config.lambda_sparsity,
            )
            tau_mm = predict_multimodal_ctrl_tau(mm_model, X_tab_test, X_text_test)
            pehe_mm = compute_pehe(tau_mm, true_te_test)

            # Tabular-only baseline
            cfg = CTRLConfig()
            plugin = train_plugin(
                X_tab_train, Y_train, T_train,
                use_gating=cfg.use_gating,
                lambda_sparsity=cfg.lambda_sparsity,
                seed=seed,
                dropout_p=cfg.dropout_p,
                hidden_dim=cfg.hidden_tau,
                batch_size=cfg.batch_size,
                epochs=100
            )
            tau_tab = predict_tau_tarnet(plugin, X_tab_test)
            pehe_tab = compute_pehe(tau_tab, true_te_test)

            rows.append({
                "p_noise": p_noise,
                "seed": seed,
                "pehe_multimodal": pehe_mm,
                "pehe_tabular": pehe_tab,
            })

    return pd.DataFrame(rows)


def run_bert_baselines(
    config: MultimodalConfig,
    batch_size: int = 32,
    max_length: int = 32,
    model_name: str = "distilbert-base-uncased",
) -> pd.DataFrame:
    """Run DistilBERT baselines (frozen embeddings + TARNet/CF)."""
    print("=== Multimodal BERT Baselines ===")
    rows = []

    ctrl_config = CTRLConfig()
    epochs = 100 if FAST_RUN else 200

    for seed in config.seeds:
        print(f"  Seed {seed}...")
        set_seed(seed)

        X_tab, X_text, Y, T, true_te = get_multimodal_data(
            n=config.n_samples,
            vocab_size=config.vocab_size,
            seed=seed,
        )

        texts = _tokens_to_text(X_text)
        embeddings = _bert_embeddings(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            model_name=model_name,
        )

        X_bert = np.concatenate([X_tab, embeddings], axis=1)

        n = len(Y)
        idx = np.random.permutation(n)
        n_train = int(0.8 * n)
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        X_train, X_test = X_bert[train_idx], X_bert[test_idx]
        Y_train, T_train = Y[train_idx], T[train_idx]
        true_te_test = true_te[test_idx]

        plugin_model = train_plugin(
            X_train, Y_train, T_train,
            use_gating=ctrl_config.use_gating,
            lambda_sparsity=ctrl_config.lambda_sparsity,
            seed=seed,
            dropout_p=ctrl_config.dropout_p,
            hidden_dim=ctrl_config.hidden_tau,
            batch_size=ctrl_config.batch_size,
            epochs=epochs,
        )
        tau_tarnet = predict_tau_tarnet(plugin_model, X_test)
        pehe_tarnet = compute_pehe(tau_tarnet, true_te_test)

        pehe_cf = np.nan
        try:
            from econml.dml import CausalForestDML
            cf = CausalForestDML(n_estimators=100, random_state=seed)
            cf.fit(Y_train, T_train, X=X_train)
            tau_cf = cf.effect(X_test)
            pehe_cf = compute_pehe(tau_cf, true_te_test)
        except ImportError:
            pass

        rows.append({
            "seed": seed,
            "pehe_tarnet": pehe_tarnet,
            "pehe_cf": pehe_cf,
            "model": model_name,
        })
        print(f"    BERT TARNet={pehe_tarnet:.3f}, BERT CF={pehe_cf:.3f}")

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


def plot_bar_summary(df: pd.DataFrame, cols: List[str], labels: List[str], output_path: Path, title: str):
    """Plot mean +/- std bar chart."""
    means = df[cols].mean()
    stds = df[cols].std()

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(cols))
    ax.bar(x, means.values, yerr=stds.values, capsize=3, color="#7a0f0f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("PEHE")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_multimodal_dense(df: pd.DataFrame, output: Path):
    """Plot clean-text dense fusion results (TarNet vs CTRL bag vs CTRL cross-attn)."""
    cols = ["pehe_plugin", "pehe_ctrl_bag", "pehe_cross_attn"]
    labels = ["TarNet (dense)", "CTRL bag", "CTRL cross-attn"]
    plot_bar_summary(df, cols, labels, output, "Multimodal Fusion (clean text)")


def plot_multimodal_dml(df: pd.DataFrame, output: Path):
    """Plot orthogonal head results on combined BoW+tab features."""
    cols = ["pehe_cf", "pehe_concat_bow", "pehe_dml"]
    labels = ["BoW CF", "BoW plug-in", "BoW CTRL-DML"]
    plot_bar_summary(df, cols, labels, output, "Orthogonal Head (BoW + tab)")


def plot_cross_attention(df: pd.DataFrame, output: Path):
    """Plot bag vs cross-attention comparison."""
    cols = ["pehe_ctrl_bag", "pehe_cross_attn"]
    labels = ["CTRL Bag", "Cross-Attn"]
    plot_bar_summary(df, cols, labels, output, "Cross-attention vs Bag")


def plot_bert_baselines(df: pd.DataFrame, output: Path):
    """Plot BERT baselines."""
    cols = ["pehe_tarnet", "pehe_cf"]
    labels = ["BERT TARNet", "BERT CF"]
    plot_bar_summary(df, cols, labels, output, "BERT Upper Bound")


def main():
    parser = argparse.ArgumentParser(description="CTRL-DML Multimodal Experiments")
    parser.add_argument("--experiment", choices=["benchmark", "noise_sweep", "bert", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=2000 if not FAST_RUN else 1000)
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--bert-max-length", type=int, default=32)
    parser.add_argument("--bert-model", type=str, default="distilbert-base-uncased")
    args = parser.parse_args()

    config = MultimodalConfig(seeds=args.seeds, n_samples=args.n_samples)
    output = get_output_manager()

    if args.experiment in ["benchmark", "all"]:
        benchmark_df = run_benchmark(config)
        csv_path = output.csv_path("multimodal_benchmark")
        benchmark_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        print("\nBenchmark Summary:")
        print(benchmark_df.describe().round(3))
        plot_multimodal_dense(benchmark_df, output.figure_path("multimodal_result"))
        plot_multimodal_dml(benchmark_df, output.figure_path("multimodal_dml"))
        plot_cross_attention(benchmark_df, output.figure_path("multimodal_cross_attention"))
        benchmark_df[["seed", "pehe_cross_attn"]].rename(columns={"pehe_cross_attn": "pehe"}).to_csv(
            output.csv_path("multimodal_cross_attention"),
            index=False,
        )

    if args.experiment in ["noise_sweep", "all"]:
        sweep_df = run_noise_sweep(config)
        csv_path = output.csv_path("multimodal_noise_sweep")
        sweep_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        plot_noise_sweep(sweep_df, output.figure_path("multimodal_sweep"))
        plot_noise_sweep(sweep_df, output.figure_path("multimodal_noise"))

    if args.experiment in ["bert", "all"]:
        bert_df = run_bert_baselines(
            config,
            batch_size=args.bert_batch_size,
            max_length=args.bert_max_length,
            model_name=args.bert_model,
        )
        csv_path = output.csv_path("multimodal_bert_baselines")
        bert_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        plot_bert_baselines(bert_df, output.figure_path("multimodal_bert_baselines"))

    print("\nTo regenerate LaTeX macros, run: python -m src.utils.latex")


if __name__ == "__main__":
    main()
