import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from data_twins import load_twins
from data_acic2019 import load_acic


class SimpleTarNet(nn.Module):
    """Lightweight TARNet for tabular plug-in baselines."""

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


def bootstrap_mean(values: np.ndarray, n_boot: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(float(np.mean(values[idx])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(values)), (float(lo), float(hi))


def fit_cf(X: np.ndarray, t: np.ndarray, y: np.ndarray, n_trees: int = 200):
    est = CausalForestDML(
        model_y=LassoCV(),
        model_t=LogisticRegressionCV(max_iter=1000),
        n_estimators=n_trees - (n_trees % 4),
        discrete_treatment=True,
        random_state=42,
    )
    est.fit(y, t, X=X)
    tau = est.effect(X).flatten()
    ate, ci = bootstrap_mean(tau, n_boot=200)
    return ate, ci, tau


def fit_tarnet(X: np.ndarray, t: np.ndarray, y: np.ndarray, epochs: int = 80, hidden: int = 64):
    x_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float().unsqueeze(1)
    t_t = torch.from_numpy(t).float().unsqueeze(1)
    model = SimpleTarNet(input_dim=X.shape[1], hidden=hidden, dropout=0.25)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    ds = TensorDataset(x_t, y_t, t_t)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    model.train()
    for _ in range(epochs):
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
    ate, ci = bootstrap_mean(tau, n_boot=200)
    return ate, ci, tau


def run_twins(sample: int = 5000):
    X, T, Y = load_twins()
    if sample and sample < len(Y):
        idx = np.random.default_rng(42).choice(len(Y), size=sample, replace=False)
        X, T, Y = X[idx], T[idx], Y[idx]
    rows = []
    try:
        ate_cf, ci_cf, _ = fit_cf(X, T, Y, n_trees=80)
        rows.append({"dataset": "TWINS", "method": "Causal Forest", "ate": ate_cf, "ci_low": ci_cf[0], "ci_high": ci_cf[1]})
    except Exception as e:
        rows.append({"dataset": "TWINS", "method": "Causal Forest", "ate": np.nan, "ci_low": np.nan, "ci_high": np.nan})
        print("CF failed on TWINS:", e)
    ate_nn, ci_nn, _ = fit_tarnet(X, T, Y, epochs=60, hidden=96)
    rows.append({"dataset": "TWINS", "method": "TarNet (plug-in)", "ate": ate_nn, "ci_low": ci_nn[0], "ci_high": ci_nn[1]})
    return rows


def run_acic(dataset: str = "low1"):
    X, T, Y = load_acic(dataset, kind="low")
    rows = []
    try:
        ate_cf, ci_cf, _ = fit_cf(X, T, Y, n_trees=80)
        rows.append({"dataset": f"ACIC-{dataset}", "method": "Causal Forest", "ate": ate_cf, "ci_low": ci_cf[0], "ci_high": ci_cf[1]})
    except Exception as e:
        rows.append({"dataset": f"ACIC-{dataset}", "method": "Causal Forest", "ate": np.nan, "ci_low": np.nan, "ci_high": np.nan})
        print("CF failed on ACIC:", e)
    ate_nn, ci_nn, _ = fit_tarnet(X, T, Y, epochs=40, hidden=64)
    rows.append({"dataset": f"ACIC-{dataset}", "method": "TarNet (plug-in)", "ate": ate_nn, "ci_low": ci_nn[0], "ci_high": ci_nn[1]})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run public benchmark baselines (TWINS, ACIC 2019 low1).")
    parser.add_argument("--acic-dataset", type=str, default="low1")
    args = parser.parse_args()

    rows = []
    rows.extend(run_twins())
    rows.extend(run_acic(args.acic_dataset))
    df = pd.DataFrame(rows)
    out = Path("public_benchmarks.csv")
    df.to_csv(out, index=False)
    print(df)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
