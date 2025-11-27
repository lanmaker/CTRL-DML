import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class SimpleTarNet(nn.Module):
    """Lightweight TARNet for plug-in baselines on real data."""

    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.25):
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


class TauNet(nn.Module):
    """One-head network for orthogonal pseudo-outcomes."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    print("CTRL-DML: cross-fitting nuisances", flush=True)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    m_hat = np.zeros_like(y, dtype=np.float32)
    e_hat = np.zeros_like(y, dtype=np.float32)
    for fold, (tr, val) in enumerate(kf.split(X)):
        m_model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42 + fold,
            n_jobs=1,
        )
        e_model = LogisticRegressionCV(max_iter=2000)
        m_model.fit(X[tr], y[tr])
        e_model.fit(X[tr], t[tr])
        m_hat[val] = m_model.predict(X[val])
        e_hat[val] = e_model.predict_proba(X[val])[:, 1]
    print("CTRL-DML: nuisances done", flush=True)
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R = y - m_hat
    W = t - e_hat
    W_clip = np.clip(W, -0.2, 0.2)
    Z = np.clip(R / W_clip, -50.0, 50.0)
    weights = np.minimum(W_clip ** 2, 0.04)

    torch.manual_seed(42)
    tau_model = TauNet(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(tau_model.parameters(), lr=5e-4, weight_decay=1e-4)
    ds = TensorDataset(
        torch.from_numpy(X).float().to(DEVICE),
        torch.from_numpy(Z).float().unsqueeze(1).to(DEVICE),
        torch.from_numpy(weights).float().unsqueeze(1).to(DEVICE),
    )
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    tau_model.train()
    print("CTRL-DML: training tau", flush=True)
    for _ in range(320):
        for bx, bz, bw in loader:
            opt.zero_grad()
            pred = tau_model(bx)
            loss = torch.mean(bw * (pred - bz) ** 2)
            loss.backward()
            opt.step()
    print("CTRL-DML: tau done", flush=True)

    tau_model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(DEVICE)
        tau = tau_model(x_t).cpu().numpy().squeeze()
    ate, ci = bootstrap_mean(tau, n_boot=400, seed=42)

    # Extract average gating mask
    with torch.no_grad():
        first_layer = tau_model.net[0]
        mask_np = np.abs(first_layer.weight.detach().cpu().numpy()).mean(axis=0)
        mask_np = mask_np / (mask_np.max() + 1e-8)
        gating = {name: float(val) for name, val in zip(feature_names, mask_np)}
    return ate, ci, tau, e_hat, gating


def fit_dragonnet(X: np.ndarray, y: np.ndarray, t: np.ndarray, seed: int = 42):
    print("DragonNet: initializing", flush=True)
    torch.manual_seed(seed)
    model = SimpleTarNet(input_dim=X.shape[1], hidden=64, dropout=0.25).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    x_t = torch.from_numpy(X).float().to(DEVICE)
    y_t = torch.from_numpy(y).float().unsqueeze(1).to(DEVICE)
    t_t = torch.from_numpy(t).float().unsqueeze(1).to(DEVICE)
    loader = DataLoader(TensorDataset(x_t, y_t, t_t), batch_size=128, shuffle=True)

    model.train()
    for _ in range(240):
        for bx, by, bt in loader:
            opt.zero_grad()
            y0, y1 = model(bx)
            y_pred = bt * y1 + (1 - bt) * y0
            loss = F.mse_loss(y_pred, by)
            loss.backward()
            opt.step()
    print("DragonNet: training complete", flush=True)

    model.eval()
    with torch.no_grad():
        y0, y1 = model(x_t)
        tau = (y1 - y0).cpu().numpy().squeeze()
    print("DragonNet: bootstrap", flush=True)
    ate, ci = bootstrap_mean(tau, n_boot=400, seed=seed)
    return ate, ci, tau


def run_dataset(name: str, n_boot: int, out_prefix: str, lalonde_scale: float, star_scale: float):
    if name == "lalonde":
        X, y, t, feature_names = load_lalonde()
        if lalonde_scale != 1.0:
            y = y / lalonde_scale
            print(f"Scaling Lalonde outcome by 1/{lalonde_scale:.0f} (thousands of dollars).")
    elif name == "star":
        X, y, t, feature_names = load_star()
        if star_scale != 1.0:
            y = y / star_scale
            print(f"Scaling STAR outcome by 1/{star_scale:.0f}.")
    else:
        raise ValueError("Unknown dataset")

    print(f"\n=== {name.upper()} ===")
    ate_ols, ci_ols = fit_ols(X, y, t, n_boot=n_boot, seed=42)
    print(f"OLS ATE: {ate_ols:.3f} (95% CI [{ci_ols[0]:.3f}, {ci_ols[1]:.3f}])")

    ate_cf, ci_cf, cf_if, tau_cf = fit_causal_forest(X, y, t)
    print(f"Causal Forest ATE: {ate_cf:.3f} (boot CI [{ci_cf[0]:.3f}, {ci_cf[1]:.3f}])")

    print("Training DragonNet plug-in...", flush=True)
    torch.manual_seed(42)
    dragon = SimpleTarNet(input_dim=X.shape[1], hidden=64, dropout=0.25).to(DEVICE)
    opt = torch.optim.Adam(dragon.parameters(), lr=0.003, weight_decay=1e-4)
    x_t = torch.from_numpy(X).float().to(DEVICE)
    y_t = torch.from_numpy(y).float().unsqueeze(1).to(DEVICE)
    t_t = torch.from_numpy(t).float().unsqueeze(1).to(DEVICE)
    loader = DataLoader(TensorDataset(x_t, y_t, t_t), batch_size=128, shuffle=True)
    dragon.train()
    for _ in range(240):
        for bx, by, bt in loader:
            opt.zero_grad()
            y0, y1 = dragon(bx)
            y_pred = bt * y1 + (1 - bt) * y0
            loss = F.mse_loss(y_pred, by)
            loss.backward()
            opt.step()
    dragon.eval()
    with torch.no_grad():
        y0_d, y1_d = dragon(x_t)
        tau_dragon = (y1_d - y0_d).cpu().numpy().squeeze()
    ate_dragon, ci_dragon = bootstrap_mean(tau_dragon, n_boot=400, seed=42)
    print(f"DragonNet ATE: {ate_dragon:.3f} (boot CI [{ci_dragon[0]:.3f}, {ci_dragon[1]:.3f}])")

    print("Training CTRL-DML orthogonal head...", flush=True)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    m_hat = np.zeros_like(y, dtype=np.float32)
    e_hat_ctrl = np.zeros_like(y, dtype=np.float32)
    for fold, (tr, val) in enumerate(kf.split(X)):
        m_model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42 + fold,
            n_jobs=1,
        )
        e_model = LogisticRegressionCV(max_iter=2000)
        m_model.fit(X[tr], y[tr])
        e_model.fit(X[tr], t[tr])
        m_hat[val] = m_model.predict(X[val])
        e_hat_ctrl[val] = e_model.predict_proba(X[val])[:, 1]
    e_hat_ctrl = np.clip(e_hat_ctrl, 0.01, 0.99)
    R = y - m_hat
    W = t - e_hat_ctrl
    W_clip = np.clip(W, -0.2, 0.2)
    Z = np.clip(R / W_clip, -50.0, 50.0)
    weights = np.minimum(W_clip ** 2, 0.04)

    torch.manual_seed(42)
    tau_model = TauNet(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(tau_model.parameters(), lr=5e-4, weight_decay=1e-4)
    ds = TensorDataset(
        torch.from_numpy(X).float().to(DEVICE),
        torch.from_numpy(Z).float().unsqueeze(1).to(DEVICE),
        torch.from_numpy(weights).float().unsqueeze(1).to(DEVICE),
    )
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    tau_model.train()
    for _ in range(320):
        for bx, bz, bw in loader:
            opt.zero_grad()
            pred = tau_model(bx)
            loss = torch.mean(bw * (pred - bz) ** 2)
            loss.backward()
            opt.step()
    tau_model.eval()
    with torch.no_grad():
        tau_ctrl = tau_model(torch.from_numpy(X).float().to(DEVICE)).cpu().numpy().squeeze()
        first_layer = tau_model.net[0]
        mask_np = np.abs(first_layer.weight.detach().cpu().numpy()).mean(axis=0)
        mask_np = mask_np / (mask_np.max() + 1e-8)
        gating = {name: float(val) for name, val in zip(feature_names, mask_np)}
    ate_ctrl, ci_ctrl = bootstrap_mean(tau_ctrl, n_boot=400, seed=42)
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
        {"dataset": name, "method": "DragonNet", "ate": ate_dragon, "ci_lower": ci_dragon[0], "ci_upper": ci_dragon[1]},
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
    parser.add_argument("--lalonde-scale", type=float, default=1000.0, help="Divide Lalonde outcome by this factor.")
    parser.add_argument("--star-scale", type=float, default=1.0, help="Divide STAR outcome by this factor.")
    args = parser.parse_args()

    if args.dataset in ("lalonde", "both"):
        run_dataset(
            "lalonde",
            n_boot=args.n_boot,
            out_prefix=args.out_prefix,
            lalonde_scale=args.lalonde_scale,
            star_scale=args.star_scale,
        )
    if args.dataset in ("star", "both"):
        run_dataset(
            "star",
            n_boot=args.n_boot,
            out_prefix=args.out_prefix,
            lalonde_scale=args.lalonde_scale,
            star_scale=args.star_scale,
        )
