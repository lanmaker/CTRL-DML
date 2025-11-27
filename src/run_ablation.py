import csv
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.model_selection import KFold

from my_dragonnet import TabularAttention
from catenets.models.torch.base import DEVICE


FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"
N_SAMPLES = 1200 if FAST_RUN else 2000
EPOCHS_NUISANCE = 120 if FAST_RUN else 300
EPOCHS_TAU = 120 if FAST_RUN else 300
BATCH_SIZE = 192 if FAST_RUN else 256
HIDDEN_DIM = 120 if FAST_RUN else 200
HIDDEN_TAU = 80 if FAST_RUN else 128
LAMBDA_TAU = 0.01


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_stress_data(n_samples: int = 2000, n_noise: int = 50, seed: int = 42):
    """Semi-synthetic stress test with confounders, instruments, and noise."""
    rng = np.random.default_rng(seed)
    C = rng.normal(0, 1, size=(n_samples, 5))
    I = rng.normal(0, 1, size=(n_samples, 5))
    N = rng.normal(0, 1, size=(n_samples, n_noise))
    X = np.concatenate([C, I, N], axis=1)

    logit = (np.sum(C, axis=1) + 0.5 * C[:, 0] * C[:, 1] + 2 * np.sum(I, axis=1))
    propensity = expit(logit)
    T = rng.binomial(1, propensity)

    true_te = 2 * np.sin(C[:, 0] * np.pi) + np.maximum(0, C[:, 1])
    y0 = np.sum(C, axis=1) ** 2 + rng.normal(0, 0.5, size=n_samples)
    y = y0 + true_te * T
    return X, T.astype(np.float32), y.astype(np.float32), true_te.astype(np.float32)


class BaseBackbone(nn.Module):
    """Shared backbone that can optionally apply TabularAttention."""

    def __init__(self, input_dim: int, hidden: int, dropout_p: float, use_gating: bool):
        super().__init__()
        self.use_gating = use_gating
        attn = TabularAttention(input_dim, hidden) if use_gating else nn.Linear(input_dim, hidden)
        self.attn = attn
        self.trunk = nn.Sequential(
            nn.ELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(x)
        return self.trunk(h)

    @property
    def mask_penalty(self) -> torch.Tensor:
        if self.use_gating and hasattr(self.attn, "current_mask_penalty"):
            return self.attn.current_mask_penalty
        return torch.tensor(0.0, device=DEVICE)


class NuisanceNet(nn.Module):
    """Joint outcome + propensity network used for cross-fitting."""

    def __init__(self, input_dim: int, hidden: int, dropout_p: float, use_gating: bool):
        super().__init__()
        self.backbone = BaseBackbone(input_dim, hidden, dropout_p, use_gating)
        self.y0 = nn.Linear(hidden, 1)
        self.y1 = nn.Linear(hidden, 1)
        self.t_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        y0 = self.y0(h)
        y1 = self.y1(h)
        t_prob = torch.sigmoid(self.t_head(h))
        return y0, y1, t_prob

    @property
    def mask_penalty(self) -> torch.Tensor:
        return self.backbone.mask_penalty


class TauNet(nn.Module):
    """CATE network trained on orthogonal residuals."""

    def __init__(self, input_dim: int, hidden: int, dropout_p: float, use_gating: bool):
        super().__init__()
        self.backbone = BaseBackbone(input_dim, hidden, dropout_p, use_gating)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h).squeeze(-1)

    @property
    def mask_penalty(self) -> torch.Tensor:
        return self.backbone.mask_penalty


class TarNet(nn.Module):
    """Direct plug-in estimator (non-orthogonal baseline)."""

    def __init__(self, input_dim: int, hidden: int, dropout_p: float, use_gating: bool):
        super().__init__()
        self.backbone = BaseBackbone(input_dim, hidden, dropout_p, use_gating)
        self.y0 = nn.Linear(hidden, 1)
        self.y1 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.y0(h), self.y1(h)

    @property
    def mask_penalty(self) -> torch.Tensor:
        return self.backbone.mask_penalty


def train_nuisance(
    X: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    use_gating: bool,
    lambda_sparsity: float,
    seed: int,
    dropout_p: float = 0.35,
    hidden_dim: int = HIDDEN_DIM,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS_NUISANCE,
) -> NuisanceNet:
    set_seed(seed)
    model = NuisanceNet(X.shape[1], hidden_dim, dropout_p, use_gating).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    x_t = torch.from_numpy(X).float().to(DEVICE)
    y_t = torch.from_numpy(y).float().to(DEVICE)
    t_t = torch.from_numpy(T).float().to(DEVICE)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(x_t.shape[0])
        for i in range(0, x_t.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            bx, by, bt = x_t[idx], y_t[idx], t_t[idx]
            opt.zero_grad()
            y0_pred, y1_pred, t_prob = model(bx)
            y_pred = bt * y1_pred + (1 - bt) * y0_pred
            loss_y = torch.mean((y_pred.squeeze() - by) ** 2)
            loss_t = torch.nn.functional.binary_cross_entropy(t_prob.squeeze(), bt)
            loss = loss_y + loss_t + lambda_sparsity * model.mask_penalty
            loss.backward()
            opt.step()
    return model


def cross_fit_nuisance(
    X: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    use_gating: bool,
    lambda_sparsity: float,
    seed: int = 42,
    k_folds: int = 5,
    dropout_p: float = 0.35,
    hidden_dim: int = HIDDEN_DIM,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS_NUISANCE,
    clip_prop: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """K-fold cross-fitting for orthogonal residuals."""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    m_hat = np.zeros_like(y, dtype=np.float32)
    e_hat = np.zeros_like(y, dtype=np.float32)

    for fold, (tr, val) in enumerate(kf.split(X)):
        model = train_nuisance(
            X[tr],
            y[tr],
            T[tr],
            use_gating,
            lambda_sparsity,
            seed + fold,
            dropout_p=dropout_p,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            epochs=epochs,
        )
        model.eval()
        with torch.no_grad():
            x_val = torch.from_numpy(X[val]).float().to(DEVICE)
            y0, y1, t_prob = model(x_val)
            t_prob_np = t_prob.squeeze().cpu().numpy()
            t_prob_np = np.clip(t_prob_np, clip_prop, 1 - clip_prop)
            y0_np = y0.squeeze().cpu().numpy()
            y1_np = y1.squeeze().cpu().numpy()
            m_hat[val] = t_prob_np * y1_np + (1 - t_prob_np) * y0_np
            e_hat[val] = t_prob_np
    return m_hat, e_hat


def train_rlearner(
    X: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    use_gating: bool,
    lambda_tau: float,
    seed: int,
    dropout_p: float = 0.35,
    hidden_dim: int = HIDDEN_TAU,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS_TAU,
    lr: float = 0.003,
    standardize_w: bool = False,
    grad_clip: float = 0.0,
    normalize_by_abs_w: bool = False,
    weight_by_w2: bool = True,
) -> TauNet:
    set_seed(seed)
    model = TauNet(X.shape[1], hidden_dim, dropout_p, use_gating).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    x_t = torch.from_numpy(X).float().to(DEVICE)
    r_np, w_np = R.copy(), W.copy()
    if standardize_w:
        r_np = (r_np - np.mean(r_np)) / (np.std(r_np) + 1e-6)
        w_np = w_np / (np.std(w_np) + 1e-6)
    r_t = torch.from_numpy(r_np).float().to(DEVICE)
    w_t = torch.from_numpy(w_np).float().to(DEVICE)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(x_t.shape[0])
        for i in range(0, x_t.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            bx, br, bw = x_t[idx], r_t[idx], w_t[idx]
            opt.zero_grad()
            tau_pred = model(bx)
            resid = br - bw * tau_pred
            if normalize_by_abs_w:
                denom = torch.mean(torch.clamp(torch.abs(bw), min=1e-3))
                resid = resid / denom
            if weight_by_w2:
                weight = torch.clamp(bw ** 2, min=1e-3)
                loss_orth = torch.mean(weight * resid ** 2)
            else:
                loss_orth = torch.mean(resid ** 2)
            loss = loss_orth + lambda_tau * model.mask_penalty
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
    return model


def train_tarnet(
    X: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    use_gating: bool,
    lambda_sparsity: float,
    seed: int,
    dropout_p: float = 0.35,
    hidden_dim: int = HIDDEN_DIM,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS_TAU,
    lr: float = 0.003,
) -> TarNet:
    set_seed(seed)
    model = TarNet(X.shape[1], hidden_dim, dropout_p, use_gating).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    x_t = torch.from_numpy(X).float().to(DEVICE)
    y_t = torch.from_numpy(y).float().to(DEVICE)
    t_t = torch.from_numpy(T).float().to(DEVICE)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(x_t.shape[0])
        for i in range(0, x_t.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            bx, by, bt = x_t[idx], y_t[idx], t_t[idx]
            opt.zero_grad()
            y0_pred, y1_pred = model(bx)
            y_pred = bt * y1_pred + (1 - bt) * y0_pred
            loss_y = torch.mean((y_pred.squeeze() - by) ** 2)
            loss = loss_y + lambda_sparsity * model.mask_penalty
            loss.backward()
            opt.step()
    return model


def evaluate_pehe(tau_pred: np.ndarray, true_te: np.ndarray) -> float:
    return float(np.sqrt(np.mean((true_te - tau_pred) ** 2)))


def predict_tau_rlearner(model: TauNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(DEVICE)
        tau = model(x_t).cpu().numpy()
    return tau


def predict_tau_tarnet(model: TarNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(DEVICE)
        y0, y1 = model(x_t)
        tau = (y1 - y0).squeeze().cpu().numpy()
    return tau


def run_ablation(
    seeds: List[int],
    n_samples: int,
    n_noise: int,
    k_folds: int,
    epochs_nuisance: int,
    epochs_tau: int,
    batch_size: int,
    hidden_dim: int,
    hidden_tau: int,
    dropout_p: float,
    lambda_tau: float,
    standardize_w: bool,
    lr: float,
    grad_clip: float,
    normalize_by_abs_w: bool,
    weight_by_w2: bool,
):
    @dataclass
    class Variant:
        name: str
        use_gating: bool
        lambda_sparsity: float

    variants: List[Variant] = [
        Variant("no_gating", False, 0.0),
        Variant("gating_no_L1", True, 0.0),
        Variant("gating_L1", True, 0.05),
    ]
    rows = []

    for variant in variants:
        for seed in seeds:
            print(f"\n>>> Variant={variant.name}, seed={seed}")
            X, T, y, true_te = get_stress_data(n_samples=n_samples, n_noise=n_noise, seed=seed)

            # Stage 1: Cross-fit nuisances
            m_hat, e_hat = cross_fit_nuisance(
                X,
                y,
                T,
                variant.use_gating,
                variant.lambda_sparsity,
                seed=seed,
                k_folds=k_folds,
                dropout_p=dropout_p,
                hidden_dim=hidden_dim,
                batch_size=batch_size,
                epochs=epochs_nuisance,
            )
            R = y - m_hat
            W = T - e_hat

            # Stage 2a: Orthogonal R-learner
            tau_model = train_rlearner(
                X,
                R,
                W,
                variant.use_gating,
                lambda_tau,
                seed=seed,
                dropout_p=dropout_p,
                hidden_dim=hidden_tau,
                batch_size=batch_size,
                epochs=epochs_tau,
                standardize_w=standardize_w,
                lr=lr,
                grad_clip=grad_clip,
                normalize_by_abs_w=normalize_by_abs_w,
                weight_by_w2=weight_by_w2,
            )
            tau_pred = predict_tau_rlearner(tau_model, X)
            pehe_orth = evaluate_pehe(tau_pred, true_te)

            # Stage 2b: Plug-in TARNet baseline
            tar_model = train_tarnet(
                X,
                y,
                T,
                variant.use_gating,
                variant.lambda_sparsity,
                seed=seed,
                dropout_p=dropout_p,
                hidden_dim=hidden_dim,
                batch_size=batch_size,
                epochs=epochs_tau,
                lr=lr,
            )
            tau_plugin = predict_tau_tarnet(tar_model, X)
            pehe_plugin = evaluate_pehe(tau_plugin, true_te)

            print(f"PEHE | DML Orthogonal: {pehe_orth:.3f} | Plug-in (no DML): {pehe_plugin:.3f}")
            rows.append(
                {
                    "variant": variant.name,
                    "use_gating": int(variant.use_gating),
                    "lambda_sparsity": variant.lambda_sparsity,
                    "seed": seed,
                    "pehe_dml": pehe_orth,
                    "pehe_plugin": pehe_plugin,
                }
            )

    # Aggregate averages per variant
    summary = {}
    for r in rows:
        key = r["variant"]
        summary.setdefault(key, {"dml": [], "plugin": []})
        summary[key]["dml"].append(r["pehe_dml"])
        summary[key]["plugin"].append(r["pehe_plugin"])

    print("\n=== Ablation Summary (mean PEHE across seeds) ===")
    for name, vals in summary.items():
        mean_dml = np.mean(vals["dml"])
        mean_plugin = np.mean(vals["plugin"])
        print(f"{name}: DML={mean_dml:.3f} | No DML={mean_plugin:.3f}")

    # Save CSV for table-ready results
    out_path = "ablation_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["variant", "use_gating", "lambda_sparsity", "seed", "pehe_dml", "pehe_plugin"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved detailed rows to {out_path}")


def parse_args():
    default_seeds = [42, 7] if FAST_RUN else [42, 1024, 2023]
    parser = argparse.ArgumentParser(description="Run CTRL-DML ablations with configurable budget.")
    parser.add_argument("--seeds", type=int, nargs="+", default=default_seeds, help="Random seeds.")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help="Number of samples.")
    parser.add_argument("--n-noise", type=int, default=50, help="Number of noise dimensions.")
    parser.add_argument("--k-folds", type=int, default=5, help="Cross-fitting folds.")
    parser.add_argument("--epochs-nuisance", type=int, default=EPOCHS_NUISANCE, help="Epochs for nuisance training.")
    parser.add_argument("--epochs-tau", type=int, default=EPOCHS_TAU, help="Epochs for tau head training.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--hidden", type=int, default=HIDDEN_DIM, help="Hidden width for nuisances.")
    parser.add_argument("--hidden-tau", type=int, default=HIDDEN_TAU, help="Hidden width for tau network.")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout probability.")
    parser.add_argument("--lambda-tau", type=float, default=LAMBDA_TAU, help="L1 weight on tau head mask.")
    parser.add_argument("--standardize-w", action="store_true", help="Standardize W and R residuals before orthogonal loss.")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate for both stages.")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping norm (0 to disable).")
    parser.add_argument(
        "--normalize-by-abs-w",
        action="store_true",
        help="Divide orthogonal residual by mean |W| to stabilize loss scale.",
    )
    parser.add_argument(
        "--no-weight-by-w2",
        action="store_true",
        help="Disable weighting orthogonal loss by W^2 (overlap-aware by default).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Config:", args)
    run_ablation(
        seeds=args.seeds,
        n_samples=args.n_samples,
        n_noise=args.n_noise,
        k_folds=args.k_folds,
        epochs_nuisance=args.epochs_nuisance,
        epochs_tau=args.epochs_tau,
        batch_size=args.batch_size,
        hidden_dim=args.hidden,
        hidden_tau=args.hidden_tau,
        dropout_p=args.dropout,
        lambda_tau=args.lambda_tau,
        standardize_w=args.standardize_w,
        lr=args.lr,
        grad_clip=args.grad_clip,
        normalize_by_abs_w=args.normalize_by_abs_w,
        weight_by_w2=not args.no_weight_by_w2,
    )
