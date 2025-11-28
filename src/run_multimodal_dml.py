"""Multimodal text-confounded benchmark with orthogonal head vs plug-in and TF-IDF Causal Forest."""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from data_multimodal import get_multimodal_data
from ctrl_orthogonal_learner import CTRLOrthogonalLearner, CTRLConfig
from run_ablation import evaluate_pehe, predict_tau_tarnet


def build_features(X_tab: np.ndarray, X_text: np.ndarray, vocab_size: int) -> np.ndarray:
    vec = TfidfVectorizer(max_features=vocab_size)
    text_tokens = [" ".join(map(str, row)) for row in X_text]
    tfidf = vec.fit_transform(text_tokens).toarray()
    return np.concatenate([X_tab, tfidf], axis=1)


def run_once(n: int, vocab_size: int, p_noise: float, seed: int) -> dict:
    np.random.seed(seed)
    X_tab, X_text, Y, T, true_te = get_multimodal_data(n=n, vocab_size=vocab_size, p_noise=p_noise)
    X = build_features(X_tab, X_text, vocab_size)

    # Causal Forest baseline
    cf = CausalForestDML(
        model_y=LassoCV(),
        model_t=LogisticRegressionCV(max_iter=1000),
        n_estimators=80,
        discrete_treatment=True,
        random_state=seed,
    )
    cf.fit(Y, T, X=X)
    tau_cf = cf.effect(X).flatten()

    cfg = CTRLConfig(
        use_gating=True,
        lambda_sparsity=0.02,
        hidden_dim=96,
        hidden_tau=96,
        dropout_p=0.4,
        batch_size=256,
        plugin_epochs=90,
        nuisance_epochs=70,
        tau_epochs=140,
        lr_plugin=0.003,
        lr_tau=3e-4,
        k_folds=3,
        lambda_tau=5e-5,
        grad_clip=1.0,
        z_clip=5.0,
        w_clip=0.05,
        aux_beta_start=0.6,
        aux_beta_end=0.2,
        aux_decay_epochs=120,
        freeze_backbone=True,
    )
    learner = CTRLOrthogonalLearner(cfg)
    learner.fit(X, Y, T, seed=seed)
    tau_dml = learner.predict_tau(X)
    tau_plugin = learner.predict_plugin(X)

    return {
        "pehe_cf": evaluate_pehe(tau_cf, true_te),
        "pehe_plugin": evaluate_pehe(tau_plugin, true_te),
        "pehe_dml": evaluate_pehe(tau_dml, true_te),
    }


def plot_bar(results: dict, out_path: str, title: str):
    labels = ["TF-IDF CF", "CTRL plug-in", "CTRL-DML"]
    vals = [results["pehe_cf"], results["pehe_plugin"], results["pehe_dml"]]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v - 0.05, f"{v:.2f}", ha="center", va="top", color="white", fontweight="bold")
    ax.set_ylabel("PEHE (lower is better)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Multimodal text-confounded benchmark with DML head.")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--vocab-size", type=int, default=800)
    parser.add_argument("--p-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default="multimodal_dml.csv")
    parser.add_argument("--plot", type=str, default="multimodal_dml.pdf")
    args = parser.parse_args()

    res = run_once(args.n, args.vocab_size, args.p_noise, args.seed)
    pd.DataFrame([res]).to_csv(args.csv, index=False)
    print(f"Wrote {args.csv}")
    plot_bar(res, args.plot, f"Multimodal (p_noise={args.p_noise})")


if __name__ == "__main__":
    main()
