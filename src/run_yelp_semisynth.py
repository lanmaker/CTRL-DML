"""Semi-synthetic text+tabular benchmark using Yelp reviews with known CATE."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV
import matplotlib.pyplot as plt

from ctrl_orthogonal_learner import CTRLOrthogonalLearner, CTRLConfig
from run_ablation import evaluate_pehe, predict_tau_tarnet


def load_yelp(base_dir: Path, n_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    reviews = pd.read_json(base_dir / "yelp_academic_dataset_review.json", lines=True, nrows=n_rows)
    biz = pd.read_json(base_dir / "yelp_academic_dataset_business.json", lines=True, nrows=n_rows)
    df = reviews.merge(biz[["business_id", "stars", "review_count", "is_open"]], on="business_id", how="inner")
    df = df.dropna(subset=["text", "stars_x", "stars_y"])
    # downsample
    if len(df) > n_rows:
        df = df.sample(n_rows, random_state=seed)
    return df


def build_semisynth(df: pd.DataFrame, max_features: int, seed: int):
    rng = np.random.default_rng(seed)
    tfidf = TfidfVectorizer(max_features=max_features, min_df=5)
    X_text = tfidf.fit_transform(df["text"].tolist()).toarray()
    text_score = X_text.mean(axis=1)  # simple proxy
    tab = df[["stars_y", "review_count", "is_open"]].copy()
    tab["is_open"] = tab["is_open"].fillna(0)
    tab["review_count"] = tab["review_count"].fillna(tab["review_count"].median())
    scaler = StandardScaler()
    X_tab = scaler.fit_transform(tab.values)
    X = np.concatenate([X_tab, X_text], axis=1)

    # Semi-synthetic treatment/outcome
    conf = text_score
    mu = 0.5 * tab["stars_y"].values + 0.1 * X_tab[:, 1]
    tau_true = 1.0 + 0.8 * conf
    logits = 0.5 * tab["stars_y"].values + 1.2 * conf
    e = 1 / (1 + np.exp(-logits))
    T = rng.binomial(1, e)
    Y = mu + tau_true * T + rng.normal(0, 0.5, size=len(df))
    return X.astype(np.float32), T.astype(np.float32), Y.astype(np.float32), tau_true.astype(np.float32)


def run_once(base_dir: Path, n_rows: int, max_features: int, seed: int):
    df = load_yelp(base_dir, n_rows, seed)
    X, T, Y, tau_true = build_semisynth(df, max_features=max_features, seed=seed)

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
        plugin_epochs=80,
        nuisance_epochs=60,
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
    tau_plugin = learner.predict_plugin(X)
    tau_dml = learner.predict_tau(X)

    return {
        "pehe_cf": evaluate_pehe(tau_cf, tau_true),
        "pehe_plugin": evaluate_pehe(tau_plugin, tau_true),
        "pehe_dml": evaluate_pehe(tau_dml, tau_true),
    }


def plot_bar(res: dict, out_path: str):
    labels = ["TF-IDF CF", "CTRL plug-in", "CTRL-DML"]
    vals = [res["pehe_cf"], res["pehe_plugin"], res["pehe_dml"]]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v - 0.05, f"{v:.2f}", ha="center", va="top", color="white", fontweight="bold")
    ax.set_ylabel("PEHE (lower is better)")
    ax.set_title("Yelp semi-synthetic (text+tabular)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Semi-synthetic Yelp text+tabular benchmark with ground-truth CATE.")
    parser.add_argument("--yelp-dir", type=str, default="Yelp JSON")
    parser.add_argument("--n-rows", type=int, default=3000)
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default="yelp_semisynth.csv")
    parser.add_argument("--plot", type=str, default="yelp_semisynth.pdf")
    args = parser.parse_args()

    res = run_once(Path(args.yelp_dir), args.n_rows, args.max_features, args.seed)
    pd.DataFrame([res]).to_csv(args.csv, index=False)
    print(f"Wrote {args.csv}")
    plot_bar(res, args.plot)


if __name__ == "__main__":
    main()
