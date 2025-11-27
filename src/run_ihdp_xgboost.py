import argparse
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import KFold

from data_ihdp import load_ihdp_replicate, COLUMNS


def set_seed(seed: int):
    np.random.seed(seed)


def cross_fit_ihdp(df: pd.DataFrame, k_folds: int, seed: int):
    X = df[[c for c in COLUMNS if c.startswith("x")]].to_numpy()
    y = df["y_factual"].to_numpy()
    t = df["treatment"].to_numpy()
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    m_hat = np.zeros_like(y, dtype=float)
    e_hat = np.zeros_like(y, dtype=float)

    for fold, (tr, va) in enumerate(kf.split(X)):
        set_seed(seed + fold)
        m = XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed + fold,
        )
        m.fit(X[tr], y[tr])
        m_hat[va] = m.predict(X[va])

        e = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed + fold,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        e.fit(X[tr], t[tr])
        e_hat[va] = e.predict_proba(X[va])[:, 1]
    return X, y, t, m_hat, e_hat


def stabilize(R, W, clip=0.05):
    W = np.clip(W, 0.01, 0.99)
    R_std = (R - R.mean()) / (R.std() + 1e-8)
    W_res = (W - W.mean()) / (W.std() + 1e-8)
    W_clip = np.clip(W_res, -clip, clip)
    W_safe = np.sign(W_clip) * np.maximum(np.abs(W_clip), 1e-3)
    Z = np.clip(R_std / W_safe, -10, 10)
    s = np.minimum(W_safe**2, clip**2)
    return Z, s


def train_tau_xgb(X, Z, s, seed: int):
    model = XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        objective="reg:squarederror",
    )
    model.fit(X, Z, sample_weight=s)
    return model


def evaluate_pehe(tau_hat, mu0, mu1):
    true_te = mu1 - mu0
    return float(np.sqrt(np.mean((tau_hat - true_te) ** 2)))


def run_rep(rep: int, k_folds: int, seed: int):
    df = load_ihdp_replicate(rep)
    X, y, t, m_hat, e_hat = cross_fit_ihdp(df, k_folds=k_folds, seed=seed)
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R = y - m_hat
    W = t - e_hat
    Z, s = stabilize(R, W, clip=0.05)
    tau_model = train_tau_xgb(X, Z, s, seed=seed)
    tau_hat = tau_model.predict(X)
    pehe = evaluate_pehe(tau_hat, df["mu0"].to_numpy(), df["mu1"].to_numpy())
    ate_hat = float(np.mean(tau_hat))
    return pehe, ate_hat


def main():
    parser = argparse.ArgumentParser(description="IHDP XGBoost R-learner baseline")
    parser.add_argument("--rep", type=int, default=1, help="IHDP replicate (1-100)")
    parser.add_argument("--k-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pehe, ate_hat = run_rep(args.rep, args.k_folds, args.seed)
    print(f"IHDP replicate {args.rep}: PEHE={pehe:.3f}, ATE-hat={ate_hat:.3f}")


if __name__ == "__main__":
    main()
