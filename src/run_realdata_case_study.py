import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV
import torch
import os

from run_ablation import cross_fit_nuisance, stabilize_residuals, train_rlearner, predict_tau_rlearner
from catenets.models.torch.base import DEVICE

# Keep threading conservative to avoid OpenMP conflicts
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


def compute_smds(X: np.ndarray, t: np.ndarray, feature_names: list[str], weights: np.ndarray | None = None):
    if weights is None:
        w = np.ones_like(t, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    w = w / (np.mean(w) + 1e-8)
    smds = {}
    for j, name in enumerate(feature_names):
        xj = X[:, j]
        w1 = w * t
        w0 = w * (1 - t)
        mu1 = np.average(xj, weights=w1)
        mu0 = np.average(xj, weights=w0)
        var1 = np.average((xj - mu1) ** 2, weights=w1)
        var0 = np.average((xj - mu0) ** 2, weights=w0)
        denom = np.sqrt((var1 + var0) / 2 + 1e-8)
        smds[name] = float(np.abs(mu1 - mu0) / denom)
    return smds


def load_lalonde():
    data = sm.datasets.get_rdataset("lalonde", "MatchIt").data
    y = data["re78"].to_numpy()
    t = data["treat"].to_numpy()
    covars = data.drop(columns=["re78", "treat"])
    X = pd.get_dummies(covars, drop_first=True)
    feature_names = list(X.columns)
    return X.to_numpy().astype(np.float32), y.astype(np.float32), t.astype(np.float32), feature_names


def load_star():
    data = sm.datasets.get_rdataset("STAR", "AER").data
    covar_cols = [
        "gender",
        "ethnicity",
        "lunch1",
        "school1",
        "experience1",
        "degree1",
        "ladder1",
        "tethnicity1",
        "system1",
    ]
    keep_cols = ["star1", "read1"] + covar_cols
    data = data[keep_cols]
    data = data.dropna(subset=["star1", "read1"])
    data = data[data["star1"].isin(["small", "regular"])]
    data = data.dropna(subset=covar_cols)
    data = data.copy()
    data["treat"] = (data["star1"] == "small").astype(int)

    X = pd.get_dummies(data[covar_cols], drop_first=True)
    feature_names = list(X.columns)
    y = data["read1"].to_numpy().astype(np.float32)
    t = data["treat"].to_numpy().astype(np.float32)
    return X.to_numpy().astype(np.float32), y, t, feature_names


def fit_ols(X: np.ndarray, y: np.ndarray, t: np.ndarray, n_boot: int, seed: int):
    X_df = pd.DataFrame(X)
    X_df["treat"] = t
    X_df = sm.add_constant(X_df)
    model = sm.OLS(y, X_df).fit()
    ate = float(model.params["treat"])

    rng = np.random.default_rng(seed)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        X_b = X_df.iloc[idx]
        b = sm.OLS(y_b, X_b).fit()
        boots.append(float(b.params["treat"]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return ate, (float(lo), float(hi))


def fit_causal_forest(X: np.ndarray, y: np.ndarray, t: np.ndarray, n_estimators: int = 400):
    est = CausalForestDML(
        model_y=LassoCV(),
        model_t=LogisticRegressionCV(max_iter=800),
        n_estimators=n_estimators,
        discrete_treatment=True,
        random_state=42,
    )
    est.fit(y, t, X=X)
    tau = est.effect(X).flatten()
    ate, ci = bootstrap_mean(tau, n_boot=400, seed=42)
    lo, hi = est.effect_interval(X, alpha=0.05)
    ate_lo = float(np.mean(lo))
    ate_hi = float(np.mean(hi))
    return ate, ci, (ate_lo, ate_hi), tau


def fit_ctrl_dml(X: np.ndarray, y: np.ndarray, t: np.ndarray, feature_names: list[str]):
    m_hat, e_hat = cross_fit_nuisance(
        X,
        y,
        t,
        use_gating=True,
        lambda_sparsity=0.05,
        seed=42,
        k_folds=3,
        dropout_p=0.3,
        hidden_dim=96,
        batch_size=128,
        epochs=200,
        clip_prop=0.01,
    )
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R = y - m_hat
    W = t - e_hat
    W_clip = np.clip(W, -0.2, 0.2)
    Z = np.clip(R / W_clip, -50.0, 50.0)
    w = np.minimum(W_clip ** 2, 0.04)
    model = train_rlearner(
        X,
        Z,
        w,
        use_gating=True,
        lambda_tau=5e-4,
        seed=42,
        dropout_p=0.3,
        hidden_dim=96,
        batch_size=128,
        epochs=280,
        lr=5e-4,
        grad_clip=1.0,
    )
    tau = predict_tau_rlearner(model, X)
    ate, ci = bootstrap_mean(tau, n_boot=400, seed=42)

    # Extract average gating mask
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(DEVICE)
        _ = model(x_t)
        mask = model.backbone.attn.last_mask
        if mask is not None:
            mask_np = mask.cpu().numpy()
            mask_mean = mask_np.mean(axis=0)
            gating = {name: float(val) for name, val in zip(feature_names, mask_mean)}
        else:
            gating = {name: 0.0 for name in feature_names}
    return ate, ci, tau, e_hat, gating


def run_dataset(name: str, n_boot: int, out_prefix: str):
    if name == "lalonde":
        X, y, t, feature_names = load_lalonde()
    elif name == "star":
        X, y, t, feature_names = load_star()
    else:
        raise ValueError("Unknown dataset")

    print(f"\n=== {name.upper()} ===")
    ate_ols, ci_ols = fit_ols(X, y, t, n_boot=n_boot, seed=42)
    print(f"OLS ATE: {ate_ols:.3f} (95% CI [{ci_ols[0]:.3f}, {ci_ols[1]:.3f}])")

    ate_cf, ci_cf, cf_if, tau_cf = fit_causal_forest(X, y, t)
    print(f"Causal Forest ATE: {ate_cf:.3f} (boot CI [{ci_cf[0]:.3f}, {ci_cf[1]:.3f}])")

    ate_ctrl, ci_ctrl, tau_ctrl, e_hat_ctrl, gating = fit_ctrl_dml(X, y, t, feature_names)
    print(f"CTRL-DML ATE: {ate_ctrl:.3f} (boot CI [{ci_ctrl[0]:.3f}, {ci_ctrl[1]:.3f}])")

    # Propensity for DragonNet-style weighting (logistic regression proxy)
    lr = LogisticRegressionCV(max_iter=2000)
    lr.fit(X, t)
    e_hat_lr = np.clip(lr.predict_proba(X)[:, 1], 0.01, 0.99)

    weights_unweighted = np.ones_like(t)
    weights_dragon = t / e_hat_lr + (1 - t) / (1 - e_hat_lr)
    weights_ctrl = t / e_hat_ctrl + (1 - t) / (1 - e_hat_ctrl)

    smd_unweighted = compute_smds(X, t, feature_names, weights_unweighted)
    smd_dragon = compute_smds(X, t, feature_names, weights_dragon)
    smd_ctrl = compute_smds(X, t, feature_names, weights_ctrl)

    ate_rows = [
        {"dataset": name, "method": "OLS", "ate": ate_ols, "ci_lower": ci_ols[0], "ci_upper": ci_ols[1]},
        {"dataset": name, "method": "CausalForest", "ate": ate_cf, "ci_lower": ci_cf[0], "ci_upper": ci_cf[1]},
        {"dataset": name, "method": "CTRL-DML", "ate": ate_ctrl, "ci_lower": ci_ctrl[0], "ci_upper": ci_ctrl[1]},
    ]
    pd.DataFrame(ate_rows).to_csv(f"{out_prefix}_{name}_ate.csv", index=False)

    smd_df = pd.DataFrame(
        {
            "feature": feature_names,
            "smd_unweighted": [smd_unweighted[f] for f in feature_names],
            "smd_dragon": [smd_dragon[f] for f in feature_names],
            "smd_ctrl": [smd_ctrl[f] for f in feature_names],
        }
    )
    smd_df.to_csv(f"{out_prefix}_{name}_smd.csv", index=False)
    pd.DataFrame.from_dict(gating, orient="index", columns=["mask"]).to_csv(f"{out_prefix}_{name}_ctrl_mask.csv")

    print(f"Saved ATE table to {out_prefix}_{name}_ate.csv")
    print(f"Saved SMD table to {out_prefix}_{name}_smd.csv")
    print(f"Saved CTRL-DML mask to {out_prefix}_{name}_ctrl_mask.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-data ATE case studies (Lalonde, STAR).")
    parser.add_argument("--dataset", type=str, choices=["lalonde", "star", "both"], default="both")
    parser.add_argument("--n-boot", type=int, default=300, help="Bootstrap samples for OLS / CI on effects.")
    parser.add_argument("--out-prefix", type=str, default="realdata")
    args = parser.parse_args()

    if args.dataset in ("lalonde", "both"):
        run_dataset("lalonde", n_boot=args.n_boot, out_prefix=args.out_prefix)
    if args.dataset in ("star", "both"):
        run_dataset("star", n_boot=args.n_boot, out_prefix=args.out_prefix)
