"""
Yelp semi-synthetic text+tabular experiment.

Uses real Yelp review text and synthetic treatment/outcome with known CATE.
Outputs:
  - output/tables/yelp_semisynth.csv
  - output/figures/yelp_semisynth.pdf
"""
import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

from src.data.yelp import load_yelp_reviews
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


POS_WORDS = {
    "great", "excellent", "amazing", "love", "wonderful", "perfect", "best", "fantastic",
    "delicious", "friendly", "clean", "fast", "happy", "awesome", "nice",
}
NEG_WORDS = {
    "bad", "terrible", "awful", "hate", "worst", "slow", "dirty", "rude",
    "disappointing", "poor", "boring", "cold", "overpriced", "unhappy",
}


@dataclass
class YelpConfig:
    seeds: List[int] = None
    n_samples: int = 2000 if not FAST_RUN else 800
    vocab_size: int = 2000 if not FAST_RUN else 1000
    n_noise: int = 20

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 7] if FAST_RUN else [42, 7, 1024]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def _text_score(texts: List[str]) -> np.ndarray:
    scores = []
    for t in texts:
        tokens = _tokenize(t)
        if not tokens:
            scores.append(0.0)
            continue
        pos = sum(tok in POS_WORDS for tok in tokens)
        neg = sum(tok in NEG_WORDS for tok in tokens)
        scores.append((pos - neg) / len(tokens))
    scores = np.array(scores, dtype=np.float32)
    if scores.std() > 0:
        scores = (scores - scores.mean()) / scores.std()
    return scores


def _tabular_features(texts: List[str], ratings: np.ndarray, rng: np.random.Generator, n_noise: int) -> np.ndarray:
    lengths = np.array([len(t) for t in texts], dtype=np.float32)
    exclam = np.array([t.count("!") for t in texts], dtype=np.float32)
    question = np.array([t.count("?") for t in texts], dtype=np.float32)
    caps_ratio = np.array(
        [(sum(1 for c in t if c.isupper()) / max(len(t), 1)) for t in texts],
        dtype=np.float32,
    )

    feats = np.column_stack([
        lengths / (lengths.max() + 1e-6),
        exclam / (exclam.max() + 1e-6),
        question / (question.max() + 1e-6),
        caps_ratio,
        ratings / 5.0,
    ])

    noise = rng.normal(0, 1, size=(len(texts), n_noise)).astype(np.float32)
    return np.concatenate([feats, noise], axis=1)


def _vectorize_texts(texts: List[str], vocab_size: int) -> np.ndarray:
    vectorizer = CountVectorizer(max_features=vocab_size, stop_words="english")
    X_bow = vectorizer.fit_transform(texts)
    return X_bow.toarray().astype(np.float32)


def _simulate_treatment_outcome(
    X_tab: np.ndarray,
    text_score: np.ndarray,
    ratings: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf = 0.6 * text_score + 0.4 * X_tab[:, 0]
    propensity = 1 / (1 + np.exp(-conf))
    propensity = np.clip(propensity, 0.05, 0.95)
    T = rng.binomial(1, propensity).astype(np.float32)

    true_tau = 1.0 + 0.8 * text_score
    base = 0.5 * text_score + 0.3 * X_tab[:, 1] + 0.2 * (ratings / 5.0)
    noise = rng.normal(0, 0.5, size=len(text_score)).astype(np.float32)
    Y = (base + true_tau * T + noise).astype(np.float32)

    return T, Y, true_tau.astype(np.float32)


def _run_ctrl_dml(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    T_train: np.ndarray,
    true_te_test: np.ndarray,
    ctrl_config: CTRLConfig,
    seed: int,
) -> Tuple[float, float]:
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


def run_yelp_semisynth(config: YelpConfig) -> pd.DataFrame:
    rows = []
    ctrl_config = CTRLConfig()
    if FAST_RUN:
        ctrl_config.plugin_epochs = 100
        ctrl_config.nuisance_epochs = 80
        ctrl_config.tau_epochs = 150
        ctrl_config.k_folds = min(3, ctrl_config.k_folds)

    for seed in config.seeds:
        print(f"Seed {seed}...")
        set_seed(seed)
        rng = np.random.default_rng(seed)

        texts, ratings = load_yelp_reviews(n_samples=config.n_samples, seed=seed)
        text_score = _text_score(texts)
        X_tab = _tabular_features(texts, ratings, rng, config.n_noise)
        T, Y, true_tau = _simulate_treatment_outcome(X_tab, text_score, ratings, rng)
        X_bow = _vectorize_texts(texts, config.vocab_size)
        X = np.concatenate([X_tab, X_bow], axis=1)

        idx = rng.permutation(len(Y))
        n_train = int(0.8 * len(Y))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, T_train = Y[train_idx], T[train_idx]
        true_te_test = true_tau[test_idx]

        pehe_plugin, pehe_dml = _run_ctrl_dml(
            X_train, X_test, Y_train, T_train, true_te_test, ctrl_config, seed
        )

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
            "pehe_cf": pehe_cf,
            "pehe_plugin": pehe_plugin,
            "pehe_dml": pehe_dml,
        })
        print(f"  PEHE CF={pehe_cf:.3f}, Plugin={pehe_plugin:.3f}, DML={pehe_dml:.3f}")

    return pd.DataFrame(rows)


def plot_yelp_semisynth(df: pd.DataFrame, output_path):
    means = df[["pehe_cf", "pehe_plugin", "pehe_dml"]].mean()
    stds = df[["pehe_cf", "pehe_plugin", "pehe_dml"]].std()

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(3)
    ax.bar(x, means.values, yerr=stds.values, capsize=3, color="#7a0f0f")
    ax.set_xticks(x)
    ax.set_xticklabels(["BoW CF", "Plug-in", "CTRL-DML"], rotation=20, ha="right")
    ax.set_ylabel("PEHE")
    ax.set_title("Yelp semi-synthetic")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Yelp semi-synthetic experiment")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=2000 if not FAST_RUN else 800)
    parser.add_argument("--vocab-size", type=int, default=2000 if not FAST_RUN else 1000)
    parser.add_argument("--n-noise", type=int, default=20)
    args = parser.parse_args()

    config = YelpConfig(
        seeds=args.seeds,
        n_samples=args.n_samples,
        vocab_size=args.vocab_size,
        n_noise=args.n_noise,
    )

    output = get_output_manager()
    df = run_yelp_semisynth(config)
    csv_path = output.csv_path("yelp_semisynth")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    plot_yelp_semisynth(df, output.figure_path("yelp_semisynth"))


if __name__ == "__main__":
    main()
