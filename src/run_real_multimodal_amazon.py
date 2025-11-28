"""
Real text+tabular case study using Amazon Reviews 2023 (verified purchase as treatment).
Exports ATE with bootstrap CIs, balance diagnostics, and simple masks for tabular covariates.
Note: Uses McAuley-Lab/Amazon-Reviews-2023 dataset with verified_purchase as treatment.
"""
import argparse
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def bootstrap_mean(values: np.ndarray, n_boot: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(float(np.mean(values[idx])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(values)), (float(lo), float(hi))


class SimpleTarNet(nn.Module):
    """TARNet-like head for concatenated tabular + text embeddings."""

    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.y0 = nn.Linear(hidden, 1)
        self.y1 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.y0(h), self.y1(h)


def load_amazon(category: str, n_samples: int, seed: int):
    # Use McAuley-Lab Amazon-Reviews-2023 dataset
    # Map old category names to new format
    if "Video_DVD" in category or "video" in category.lower():
        new_category = "raw_review_Video_Games"
    elif category.startswith("raw_review_"):
        new_category = category
    else:
        new_category = "raw_review_Video_Games"  # default

    ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", new_category, split="full", trust_remote_code=True)
    df = ds.to_pandas()

    # Map new column names to old expected names
    df = df.rename(columns={
        "text": "review_body",
        "rating": "star_rating",
        "helpful_vote": "helpful_votes"
    })

    # Select relevant columns (verified_purchase is now boolean, not "Y"/"N")
    df = df[["review_body", "star_rating", "verified_purchase", "helpful_votes", "parent_asin"]]
    df = df.sample(frac=1.0, random_state=seed)  # shuffle
    df = df.head(n_samples)
    df = df.dropna(subset=["review_body", "star_rating"])

    # Use verified_purchase as treatment (1=verified, 0=unverified)
    df["treatment"] = df["verified_purchase"].astype(int)
    # Create helpful_ratio (use helpful_votes as proxy since we don't have total_votes)
    df["helpful_ratio"] = np.clip(df["helpful_votes"] / 10.0, 0, 1)  # normalize by typical max
    df["review_len"] = df["review_body"].str.len()

    t = df["treatment"].to_numpy().astype(np.float32)
    y = df["star_rating"].to_numpy().astype(np.float32)

    # Use parent_asin for product categories (take top 10 most common)
    top_asins = df["parent_asin"].value_counts().head(10).index
    df["product_cat"] = df["parent_asin"].apply(lambda x: x if x in top_asins else "other")
    cat_dummies = pd.get_dummies(df["product_cat"], drop_first=True)

    tab = pd.concat(
        [
            df[["helpful_ratio", "review_len"]].reset_index(drop=True),
            cat_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    tab_cols = list(tab.columns)
    X_tab = tab.to_numpy().astype(np.float32)
    texts = df["review_body"].tolist()
    return X_tab, texts, y, t, tab_cols


def fit_cf_tfidf(X_tab: np.ndarray, texts: list[str], y: np.ndarray, t: np.ndarray, max_features: int = 5000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=5)
    X_tfidf = vec.fit_transform(texts).toarray().astype(np.float32)
    X = np.concatenate([X_tab, X_tfidf], axis=1)
    est = CausalForestDML(
        model_y=LassoCV(),
        model_t=LogisticRegressionCV(max_iter=1000),
        n_estimators=200,
        discrete_treatment=True,
        random_state=42,
    )
    est.fit(y, t, X=X)
    tau = est.effect(X).flatten()
    ate, ci = bootstrap_mean(tau, n_boot=300, seed=42)
    lo, hi = est.effect_interval(X, alpha=0.05)
    ate_lo = float(np.mean(lo))
    ate_hi = float(np.mean(hi))
    return ate, ci, (ate_lo, ate_hi), tau


def fit_ctrl_dense(X_tab: np.ndarray, texts: list[str], y: np.ndarray, t: np.ndarray, max_features: int = 2000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=5)
    X_tfidf = vec.fit_transform(texts).toarray().astype(np.float32)
    X = np.concatenate([X_tab, X_tfidf], axis=1)

    x_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float().unsqueeze(1)
    t_t = torch.from_numpy(t).float().unsqueeze(1)
    model = SimpleTarNet(input_dim=X.shape[1], hidden=128, dropout=0.3)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    ds = TensorDataset(x_t, y_t, t_t)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    model.train()
    for _ in range(60):
        for bx, by, bt in loader:
            opt.zero_grad()
            y0, y1 = model(bx)
            y_pred = bt * y1 + (1 - bt) * y0
            loss = F.mse_loss(y_pred, by)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        y0, y1 = model(x_t)
        tau = (y1 - y0).cpu().numpy().flatten()
    ate, ci = bootstrap_mean(tau, n_boot=300, seed=42)
    # proxy mask: abs first-layer weights on tabular slice
    w = model.shared[0].weight.detach().cpu().numpy()
    mask = np.abs(w).mean(axis=0)[: X_tab.shape[1]]  # first columns correspond to tabular
    mask = mask / (mask.max() + 1e-8)
    return ate, ci, tau, mask


def compute_smd(X: np.ndarray, t: np.ndarray, weights: np.ndarray | None = None):
    if weights is None:
        w = np.ones_like(t, dtype=float)
    else:
        w = weights / (np.mean(weights) + 1e-8)
    smds = []
    for j in range(X.shape[1]):
        xj = X[:, j]
        w1 = w * t
        w0 = w * (1 - t)
        mu1 = np.average(xj, weights=w1)
        mu0 = np.average(xj, weights=w0)
        var1 = np.average((xj - mu1) ** 2, weights=w1)
        var0 = np.average((xj - mu0) ** 2, weights=w0)
        denom = np.sqrt((var1 + var0) / 2 + 1e-8)
        smds.append(float(np.abs(mu1 - mu0) / denom))
    return np.array(smds)


def main():
    parser = argparse.ArgumentParser(description="Real text+tabular ATE: Amazon Reviews 2023 (verified purchase).")
    parser.add_argument("--category", type=str, default="Video_DVD_v1_00", help="Category (maps to Video_Games)")
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", type=str, default="realdata_amazon")
    args = parser.parse_args()

    X_tab, texts, y, t, tab_cols = load_amazon(args.category, args.n_samples, args.seed)
    print(f"Loaded {len(y)} samples, treatment rate {t.mean():.3f}")

    ate_cf, ci_cf, ci_int_cf, tau_cf = fit_cf_tfidf(X_tab, texts, y, t)
    ate_ctrl, ci_ctrl, tau_ctrl, mask_tab = fit_ctrl_dense(X_tab, texts, y, t)

    # Balance diagnostics using propensity weights from CTRL-DML (logistic regression on tabular)
    lr = LogisticRegressionCV(max_iter=1000)
    lr.fit(X_tab, t)
    e_hat_lr = np.clip(lr.predict_proba(X_tab)[:, 1], 0.01, 0.99)
    weights_dragon = t / e_hat_lr + (1 - t) / (1 - e_hat_lr)
    weights_ctrl = weights_dragon  # reuse as proxy; CTRL propensities not modeled separately here

    smd_unw = compute_smd(X_tab, t, None)
    smd_dragon = compute_smd(X_tab, t, weights_dragon)
    smd_ctrl = compute_smd(X_tab, t, weights_ctrl)

    ate_rows = [
        {"dataset": args.category, "method": "CF_TFIDF", "ate": ate_cf, "ci_lower": ci_cf[0], "ci_upper": ci_cf[1]},
        {"dataset": args.category, "method": "CTRL_DML", "ate": ate_ctrl, "ci_lower": ci_ctrl[0], "ci_upper": ci_ctrl[1]},
    ]
    pd.DataFrame(ate_rows).to_csv(f"{args.out_prefix}_ate.csv", index=False)
    pd.DataFrame(
        {
            "feature": tab_cols,
            "smd_unweighted": smd_unw,
            "smd_dragon": smd_dragon,
            "smd_ctrl": smd_ctrl,
            "mask_tab": mask_tab,
        }
    ).to_csv(f"{args.out_prefix}_balance.csv", index=False)
    print("ATE CF_TFIDF:", ate_cf, ci_cf)
    print("ATE CTRL_DML:", ate_ctrl, ci_ctrl)
    print("Saved ATE and balance to", args.out_prefix)


if __name__ == "__main__":
    main()
