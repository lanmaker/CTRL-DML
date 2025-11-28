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
EPOCHS_PLUGIN = 100 if FAST_RUN else 220
EPOCHS_NUISANCE = 90 if FAST_RUN else 150
EPOCHS_TAU = 180 if FAST_RUN else 300
BATCH_SIZE = 160 if FAST_RUN else 192
HIDDEN_DIM = 96 if FAST_RUN else 120
HIDDEN_TAU = 64 if FAST_RUN else 96
LAMBDA_TAU = 5e-4


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


def train_plugin(
    X: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    use_gating: bool,
    lambda_sparsity: float,
    seed: int,
    dropout_p: float,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float = 0.003,
) -> TarNet:
    """Stage 0: fit plug-in TARNet to warm-start tau."""
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
    t_weight: float = 1.0,
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
            loss = loss_y + t_weight * loss_t + lambda_sparsity * model.mask_penalty
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
    k_folds: int = 3,
    dropout_p: float = 0.35,
    hidden_dim: int = HIDDEN_DIM,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS_NUISANCE,
    clip_prop: float = 0.01,
    t_weight: float = 1.0,
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
            t_weight=t_weight,
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


def stabilize_residuals(
    R: np.ndarray, W: np.ndarray, clip_w: float = 0.05, z_clip: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Return pseudo-outcome Z and weights s for the ratio R-learner."""
    R_mean, R_std = float(np.mean(R)), float(np.std(R))
    W_std = float(np.std(W))
    R_std = R_std + 1e-8
    W_std = W_std + 1e-8

    R_stdzd = (R - R_mean) / R_std
    W_stdzd = W / W_std
    W_clip = np.clip(W_stdzd, -clip_w, clip_w)
    W_safe = np.sign(W_clip) * np.maximum(np.abs(W_clip), 1e-3)
    Z = R_stdzd / W_safe
    Z = np.clip(Z, -z_clip, z_clip)
    s = np.minimum(W_safe ** 2, clip_w ** 2)
    s = s / (np.mean(s) + 1e-8)
    return Z.astype(np.float32), s.astype(np.float32)


def train_rlearner(
    X: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray,
    use_gating: bool,
    lambda_tau: float,
    seed: int,
    dropout_p: float = 0.35,
    hidden_dim: int = HIDDEN_TAU,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS_TAU,
    lr: float = 3e-4,
    grad_clip: float = 0.0,
    warm_start_from: TarNet | None = None,
    teacher_tau: np.ndarray | None = None,
    aux_beta_start: float = 0.1,
    aux_beta_end: float = 0.0,
    aux_decay_epochs: int = 50,
    freeze_backbone: bool = False,
) -> TauNet:
    set_seed(seed)
    model = TauNet(X.shape[1], hidden_dim, dropout_p, use_gating).to(DEVICE)
    if warm_start_from is not None:
        # Copy backbone weights and initialize tau head with y1 - y0 to warm-start CATE.
        model.backbone.load_state_dict(warm_start_from.backbone.state_dict())
        with torch.no_grad():
            model.head.weight.copy_(warm_start_from.y1.weight - warm_start_from.y0.weight)
            model.head.bias.copy_(warm_start_from.y1.bias - warm_start_from.y0.bias)

    if freeze_backbone:
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    x_t = torch.from_numpy(X).float().to(DEVICE)
    z_t = torch.from_numpy(Z).float().to(DEVICE)
    w_t = torch.from_numpy(np.maximum(weights, 1e-6)).float().to(DEVICE)
    teacher_t = torch.from_numpy(teacher_tau).float().to(DEVICE) if teacher_tau is not None else None

    def beta_for_epoch(epoch: int) -> float:
        if aux_decay_epochs <= 0:
            return aux_beta_end
        if epoch >= aux_decay_epochs:
            return aux_beta_end
        frac = 1 - epoch / aux_decay_epochs
        return aux_beta_end + frac * (aux_beta_start - aux_beta_end)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(x_t.shape[0])
        for i in range(0, x_t.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            bx, bz, bw = x_t[idx], z_t[idx], w_t[idx]
            opt.zero_grad()
            tau_pred = model(bx)
            loss_dml = torch.mean(bw * (tau_pred - bz) ** 2)
            loss = loss_dml + lambda_tau * model.mask_penalty
            beta = beta_for_epoch(epoch)
            if teacher_t is not None and beta > 0:
                teacher_pred = teacher_t[idx]
                aux_loss = torch.mean((tau_pred - teacher_pred) ** 2)
                loss = loss + beta * aux_loss
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
    plugin_epochs: int,
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
    clip_w: float,
    z_clip: float,
    aux_beta_start: float,
    aux_beta_end: float,
    aux_decay_epochs: int,
    plugin_lr: float,
    save_plugin: bool,
    freeze_backbone: bool,
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

            # Stage 0: plug-in pretrain (used for baseline + warm-start)
            plugin_model = train_plugin(
                X,
                y,
                T,
                variant.use_gating,
                variant.lambda_sparsity,
                seed=seed,
                dropout_p=dropout_p,
                hidden_dim=hidden_tau,
                batch_size=batch_size,
                epochs=plugin_epochs,
                lr=plugin_lr,
            )
            tau_plugin = predict_tau_tarnet(plugin_model, X)
            if save_plugin:
                ckpt_path = f"ctrl_plugin_{variant.name}_seed{seed}.pt"
                torch.save(plugin_model.state_dict(), ckpt_path)
                print(f"Saved plug-in checkpoint to {ckpt_path}")

            # Stage 1: Cross-fit nuisances (freshly initialized)
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
            e_hat = np.clip(e_hat, 0.01, 0.99)
            R = y - m_hat
            W = T - e_hat
            if standardize_w:
                Z, weights = stabilize_residuals(R, W, clip_w=clip_w, z_clip=z_clip)
            else:
                Z, weights = R, np.clip(W, -clip_w, clip_w) ** 2

            # Stage 2a: Orthogonal R-learner
            tau_model = train_rlearner(
                X,
                Z,
                weights,
                variant.use_gating,
                lambda_tau,
                seed=seed,
                dropout_p=dropout_p,
                hidden_dim=hidden_tau,
                batch_size=batch_size,
                epochs=epochs_tau,
                lr=lr,
                grad_clip=grad_clip,
                warm_start_from=plugin_model,
                teacher_tau=tau_plugin,
                aux_beta_start=aux_beta_start,
                aux_beta_end=aux_beta_end,
                aux_decay_epochs=aux_decay_epochs,
                freeze_backbone=freeze_backbone,
            )
            tau_pred = predict_tau_rlearner(tau_model, X)
            pehe_orth = evaluate_pehe(tau_pred, true_te)

            # Stage 2b: Plug-in TARNet baseline (from Stage 0 checkpoint)
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
    parser.add_argument("--k-folds", type=int, default=3, help="Cross-fitting folds.")
    parser.add_argument("--plugin-epochs", type=int, default=EPOCHS_PLUGIN, help="Epochs for plug-in pretraining.")
    parser.add_argument("--epochs-nuisance", type=int, default=EPOCHS_NUISANCE, help="Epochs for nuisance training.")
    parser.add_argument("--epochs-tau", type=int, default=EPOCHS_TAU, help="Epochs for tau head training.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--hidden", type=int, default=HIDDEN_DIM, help="Hidden width for nuisances.")
    parser.add_argument("--hidden-tau", type=int, default=HIDDEN_TAU, help="Hidden width for tau network.")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout probability.")
    parser.add_argument("--lambda-tau", type=float, default=LAMBDA_TAU, help="L1 weight on tau head mask.")
    parser.add_argument(
        "--standardize-w",
        dest="standardize_w",
        action="store_true",
        default=True,
        help="Standardize W and R residuals before orthogonal loss.",
    )
    parser.add_argument(
        "--no-standardize-w",
        dest="standardize_w",
        action="store_false",
        help="Disable residual standardization.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for tau fine-tuning.")
    parser.add_argument("--plugin-lr", type=float, default=0.003, help="Learning rate for plug-in pretraining.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm (0 to disable).")
    parser.add_argument(
        "--clip-w",
        type=float,
        default=0.05,
        help="Clip orthogonal residual weights to [-clip_w, clip_w] before ratio loss.",
    )
    parser.add_argument(
        "--z-clip",
        type=float,
        default=10.0,
        help="Clip pseudo-outcomes Z to stabilize the ratio target.",
    )
    parser.add_argument(
        "--aux-beta-start",
        type=float,
        default=0.1,
        help="Initial weight on plug-in auxiliary loss.",
    )
    parser.add_argument(
        "--aux-beta-end",
        type=float,
        default=0.0,
        help="Final weight on plug-in auxiliary loss after decay.",
    )
    parser.add_argument(
        "--aux-decay-epochs",
        type=int,
        default=50,
        help="Epochs over which to decay the auxiliary loss weight.",
    )
    parser.add_argument(
        "--no-save-plugin",
        action="store_true",
        help="Skip saving plug-in checkpoints.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the backbone during tau fine-tuning (keeps gating fixed for stability).",
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
        plugin_epochs=args.plugin_epochs,
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
        clip_w=args.clip_w,
        z_clip=args.z_clip,
        aux_beta_start=args.aux_beta_start,
        aux_beta_end=args.aux_beta_end,
        aux_decay_epochs=args.aux_decay_epochs,
        plugin_lr=args.plugin_lr,
        save_plugin=not args.no_save_plugin,
        freeze_backbone=args.freeze_backbone,
    )
