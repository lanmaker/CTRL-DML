"""
Real multimodal case study on the Yelp Open Dataset.

Treatment: RestaurantsReservations attribute (whether the business accepts reservations).
Outcome: review star rating.
Covariates: business attributes (is_open, review_count, price range, categories), user stats (average_stars,
review_count, fans, tenure), and review text.

This script samples a manageable subset of reviews, merges with business and user tables, and estimates ATEs using
TF-IDF Causal Forest and a lightweight TARNet-style neural model. It also reports covariate balance via SMDs.
"""
import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DATA_ROOT = Path("Yelp JSON") / "yelp_dataset"


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

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.y0(h), self.y1(h)


def bootstrap_mean(values: np.ndarray, n_boot: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(float(np.mean(values[idx])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(values)), (float(lo), float(hi))


def _parse_price(attrs: str | dict | float) -> float:
    if pd.isna(attrs):
        return 0.0
    try:
        d = attrs if isinstance(attrs, dict) else json.loads(attrs)
    except Exception:
        return 0.0
    val = d.get("RestaurantsPriceRange2")
    try:
        return float(val)
    except Exception:
        return 0.0


def _has_reservation(attrs: str | dict | float) -> int:
    if pd.isna(attrs):
        return 0
    try:
        d = attrs if isinstance(attrs, dict) else json.loads(attrs)
    except Exception:
        return 0
    val = d.get("RestaurantsReservations")
    return int(str(val).lower() == "true")


def load_reviews(n_reviews: int, seed: int) -> pd.DataFrame:
    """Read a small prefix of reviews to avoid loading the full file."""
    rev_path = DATA_ROOT / "yelp_academic_dataset_review.json"
    df = pd.read_json(rev_path, lines=True, nrows=n_reviews, dtype={"text": str})
    df = df[["business_id", "user_id", "text", "stars"]].dropna()
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df.head(n_reviews)


def filter_table(path: Path, key: str, ids: Iterable[str], use_cols: list[str], nrows: int = 400_000) -> pd.DataFrame:
    ids = set(ids)
    chunk = pd.read_json(path, lines=True, nrows=nrows)
    sub = chunk[chunk[key].isin(ids)][[key] + use_cols]
    return sub.reset_index(drop=True)


def build_dataframe(n_reviews: int, seed: int) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]:
    rev = load_reviews(n_reviews, seed)
    biz = filter_table(
        DATA_ROOT / "yelp_academic_dataset_business.json",
        key="business_id",
        ids=rev["business_id"].unique(),
        use_cols=["is_open", "review_count", "attributes", "categories", "stars"],
        nrows=max(200_000, n_reviews * 5),
    )
    users = filter_table(
        DATA_ROOT / "yelp_academic_dataset_user.json",
        key="user_id",
        ids=rev["user_id"].unique(),
        use_cols=["average_stars", "review_count", "fans", "yelping_since"],
        nrows=max(200_000, n_reviews * 5),
    )
    df = (
        rev.merge(biz, on="business_id", how="inner")
        .merge(users, on="user_id", how="inner", suffixes=("_biz", "_user"))
    )
    df["price"] = df["attributes"].apply(_parse_price)
    df["T"] = df["attributes"].apply(_has_reservation)
    df = df[df["T"].isin([0, 1])]
    df["tenure_years"] = pd.to_datetime("today").year - pd.to_datetime(df["yelping_since"]).dt.year
    df["Y"] = df["stars_x"].astype(float)
    texts = df["text"].tolist()

    # Tabular covariates
    tab = pd.DataFrame(
        {
            "is_open": df["is_open"].astype(float),
            "biz_review_count": df["review_count_biz"].astype(float),
            "biz_stars": df["stars_y"].astype(float),
            "price": df["price"].astype(float),
            "user_avg": df["average_stars"].astype(float),
            "user_review_count": df["review_count_user"].astype(float),
            "fans": df["fans"].astype(float),
            "tenure": df["tenure_years"].fillna(0).astype(float),
        }
    )
    tab_cols = list(tab.columns)
    X_tab = tab.to_numpy().astype(np.float32)
    y = df["Y"].to_numpy().astype(np.float32)
    t = df["T"].to_numpy().astype(np.float32)
    return X_tab, texts, y, t, tab_cols


def fit_cf_tfidf(X_tab: np.ndarray, texts: list[str], y: np.ndarray, t: np.ndarray, max_features: int = 5000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=10)
    X_tfidf = vec.fit_transform(texts).toarray().astype(np.float32)
    X = np.concatenate([X_tab, X_tfidf], axis=1)
    est = CausalForestDML(
        model_y=LassoCV(),
        model_t=LogisticRegressionCV(max_iter=1000),
        n_estimators=160,
        discrete_treatment=True,
        random_state=42,
    )
    est.fit(y, t, X=X)
    tau = est.effect(X).flatten()
    ate, ci = bootstrap_mean(tau, n_boot=200, seed=42)
    return ate, ci, tau


def fit_ctrl_dense(X_tab: np.ndarray, texts: list[str], y: np.ndarray, t: np.ndarray, max_features: int = 5000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=10)
    X_tfidf = vec.fit_transform(texts).toarray().astype(np.float32)
    X = np.concatenate([X_tab, X_tfidf], axis=1)
    x_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float().unsqueeze(1)
    t_t = torch.from_numpy(t).float().unsqueeze(1)
    model = SimpleTarNet(input_dim=X.shape[1], hidden=96, dropout=0.25)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    ds = TensorDataset(x_t, y_t, t_t)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    model.train()
    for _ in range(40):
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
    ate, ci = bootstrap_mean(tau, n_boot=200, seed=42)
    return ate, ci, tau


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
    parser = argparse.ArgumentParser(description="Yelp multimodal ATE: reservations -> review stars.")
    parser.add_argument("--n-reviews", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=5000)
    args = parser.parse_args()

    X_tab, texts, y, t, tab_cols = build_dataframe(args.n_reviews, args.seed)
    print(f"Loaded {len(y)} samples; treatment rate {t.mean():.3f}")
    ate_cf, ci_cf, tau_cf = fit_cf_tfidf(X_tab, texts, y, t, max_features=args.max_features)
    ate_ctrl, ci_ctrl, tau_ctrl = fit_ctrl_dense(X_tab, texts, y, t, max_features=args.max_features)

    # Balance using logistic propensity on tabular covariates
    lr = LogisticRegressionCV(max_iter=1000)
    lr.fit(X_tab, t)
    e_hat = np.clip(lr.predict_proba(X_tab)[:, 1], 0.01, 0.99)
    weights = t / e_hat + (1 - t) / (1 - e_hat)
    smd_unweighted = compute_smd(X_tab, t)
    smd_weighted = compute_smd(X_tab, t, weights=weights)

    rows = [
        {"method": "TFIDF_CF", "ate": ate_cf, "ci_low": ci_cf[0], "ci_high": ci_cf[1]},
        {"method": "CTRL_DML_dense", "ate": ate_ctrl, "ci_low": ci_ctrl[0], "ci_high": ci_ctrl[1]},
    ]
    out_ate = Path("realdata_yelp_ate.csv")
    pd.DataFrame(rows).to_csv(out_ate, index=False)
    print(f"Wrote ATEs to {out_ate}")

    smd_df = pd.DataFrame({"feature": tab_cols, "smd_unweighted": smd_unweighted, "smd_weighted": smd_weighted})
    out_smd = Path("realdata_yelp_smd.csv")
    smd_df.to_csv(out_smd, index=False)
    print(f"Wrote balance to {out_smd}")


if __name__ == "__main__":
    main()
