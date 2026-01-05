"""
High-level CTRL-DML orthogonal learner wrapper.

This class packages the three-stage pipeline:
    1) Plug-in warm-start (TARNet-style head with gating)
    2) Cross-fitted nuisances (outcome/propensity heads)
    3) Orthogonal tau head (R-learner ratio loss with clipping + optional distillation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from .dragonnet import TarNet, TauNet, NuisanceNet, get_device

DEVICE = get_device()


def _resolve_k_folds(T: np.ndarray, k_folds: int) -> int:
    """
    Ensure k_folds does not exceed the minimum class count.

    StratifiedKFold requires at least k samples per class. This function
    clips k_folds to a safe value based on treatment group sizes.

    Args:
        T: Binary treatment array
        k_folds: Desired number of folds

    Returns:
        Safe k_folds value (at least 2, at most min_class_count)

    Raises:
        ValueError: If any treatment group has fewer than 2 samples
    """
    counts = [int(np.sum(T == 0)), int(np.sum(T == 1))]
    min_class = min(counts)
    if min_class < 2:
        raise ValueError(
            f"Need at least 2 samples per treatment group for cross-fitting. "
            f"Got counts: control={counts[0]}, treated={counts[1]}"
        )
    resolved = max(2, min(k_folds, min_class))
    if resolved < k_folds:
        import warnings
        warnings.warn(
            f"Reducing k_folds from {k_folds} to {resolved} due to small class size "
            f"(min_class={min_class})",
            UserWarning,
            stacklevel=2,
        )
    return resolved


@dataclass
class CTRLConfig:
    """Configuration for CTRL-DML orthogonal learner."""
    use_gating: bool = True
    lambda_sparsity: float = 0.05
    hidden_dim: int = 120
    hidden_tau: int = 96
    dropout_p: float = 0.4
    batch_size: int = 192
    plugin_epochs: int = 220
    nuisance_epochs: int = 150
    tau_epochs: int = 360
    lr_plugin: float = 3e-3
    lr_tau: float = 3e-4
    k_folds: int = 5
    lambda_tau: float = 5e-5
    grad_clip: float = 1.0
    z_clip: float = 5.0
    w_clip: float = 0.05
    eps: float = 0.05
    aux_beta_start: float = 0.8
    aux_beta_end: float = 0.3
    aux_decay_epochs: int = 180
    freeze_backbone: bool = False


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
            idx = perm[i:i + batch_size]
            bx, by, bt = x_t[idx], y_t[idx], t_t[idx]
            if bx.shape[0] < 2:
                continue  # Avoid BatchNorm issues on tiny batches.
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
    hidden_dim: int = 120,
    batch_size: int = 192,
    epochs: int = 150,
    t_weight: float = 1.0,
) -> NuisanceNet:
    """Train nuisance network for outcome and propensity."""
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
            idx = perm[i:i + batch_size]
            bx, by, bt = x_t[idx], y_t[idx], t_t[idx]
            if bx.shape[0] < 2:
                continue  # Avoid BatchNorm issues on tiny batches.
            opt.zero_grad()
            y0_pred, y1_pred, t_prob = model(bx)
            y_pred = bt * y1_pred + (1 - bt) * y0_pred
            loss_y = torch.mean((y_pred.squeeze() - by) ** 2)
            loss_t = nn.functional.binary_cross_entropy(t_prob.squeeze(), bt)
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
    hidden_dim: int = 120,
    batch_size: int = 192,
    epochs: int = 150,
    clip_prop: float = 0.01,
    t_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """K-fold cross-fitting for orthogonal residuals."""
    # Resolve k_folds to avoid StratifiedKFold errors with small classes
    k_folds = _resolve_k_folds(T, k_folds)
    # Use StratifiedKFold to ensure each fold has both treatment groups
    T_int = T.astype(int)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    m_hat = np.zeros_like(y, dtype=np.float32)
    e_hat = np.zeros_like(y, dtype=np.float32)

    for fold, (tr, val) in enumerate(skf.split(X, T_int)):
        model = train_nuisance(
            X[tr], y[tr], T[tr],
            use_gating, lambda_sparsity, seed + fold,
            dropout_p=dropout_p, hidden_dim=hidden_dim,
            batch_size=batch_size, epochs=epochs, t_weight=t_weight,
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
    R: np.ndarray,
    W: np.ndarray,
    w_clip: float = 0.05,
    z_clip: float = 5.0,
    eps: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return pseudo-outcome Z and weights s for the ratio R-learner.

    Standard R-learner: minimize E[s * (tau(X) - Z)^2] where Z = R/W.

    Args:
        R: Outcome residuals Y - m(X)
        W: Treatment residuals T - e(X)
        w_clip: Minimum |W| for division stability
        z_clip: Clip Z to [-z_clip, z_clip]
        eps: Weight cap, s = min(W^2, eps^2)

    Returns:
        Z: Pseudo-outcomes (clipped R/W)
        s: Sample weights min(W^2, eps^2)
    """
    # Ensure |W| >= w_clip for division stability
    W_safe = np.sign(W) * np.maximum(np.abs(W), w_clip)

    # Compute pseudo-outcome Z = R / W, then clip
    Z = R / W_safe
    Z = np.clip(Z, -z_clip, z_clip)

    # Compute weights s = min(W^2, eps^2)
    s = np.minimum(W ** 2, eps ** 2)

    return Z.astype(np.float32), s.astype(np.float32)


def train_rlearner(
    X: np.ndarray,
    Z: np.ndarray,
    weights: np.ndarray,
    use_gating: bool,
    lambda_tau: float,
    seed: int,
    dropout_p: float = 0.35,
    hidden_dim: int = 96,
    batch_size: int = 192,
    epochs: int = 360,
    lr: float = 3e-4,
    grad_clip: float = 0.0,
    warm_start_from: Optional[TarNet] = None,
    teacher_tau: Optional[np.ndarray] = None,
    aux_beta_start: float = 0.8,
    aux_beta_end: float = 0.3,
    aux_decay_epochs: int = 180,
    freeze_backbone: bool = False,
    val_split: float = 0.1,
    early_stopping_patience: int = 20,
) -> TauNet:
    """Train orthogonal R-learner tau head with early stopping."""
    set_seed(seed)
    model = TauNet(X.shape[1], hidden_dim, dropout_p, use_gating).to(DEVICE)

    if warm_start_from is not None:
        model.backbone.load_state_dict(warm_start_from.backbone.state_dict())
        with torch.no_grad():
            model.head.weight.copy_(warm_start_from.y1.weight - warm_start_from.y0.weight)
            model.head.bias.copy_(warm_start_from.y1.bias - warm_start_from.y0.bias)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-5
    )

    # Train/val split
    n = X.shape[0]
    n_val = int(n * val_split) if val_split > 0 else 0
    perm = np.random.permutation(n)
    val_idx = perm[:n_val] if n_val > 0 else np.array([], dtype=int)
    train_idx = perm[n_val:]

    x_train = torch.from_numpy(X[train_idx]).float().to(DEVICE)
    z_train = torch.from_numpy(Z[train_idx]).float().to(DEVICE)
    w_train = torch.from_numpy(np.maximum(weights[train_idx], 1e-6)).float().to(DEVICE)

    if n_val > 0:
        x_val = torch.from_numpy(X[val_idx]).float().to(DEVICE)
        z_val = torch.from_numpy(Z[val_idx]).float().to(DEVICE)
        w_val = torch.from_numpy(np.maximum(weights[val_idx], 1e-6)).float().to(DEVICE)

    teacher_t = None
    if teacher_tau is not None:
        teacher_t = torch.from_numpy(teacher_tau[train_idx]).float().to(DEVICE)

    def beta_for_epoch(epoch: int) -> float:
        if aux_decay_epochs <= 0:
            return aux_beta_end
        if epoch >= aux_decay_epochs:
            return aux_beta_end
        progress = epoch / aux_decay_epochs
        # Cosine decay
        return aux_beta_end + 0.5 * (aux_beta_start - aux_beta_end) * (1 + np.cos(np.pi * progress))

    # Early stopping state
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        # Training
        perm = torch.randperm(x_train.shape[0])
        for i in range(0, x_train.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            bx, bz, bw = x_train[idx], z_train[idx], w_train[idx]
            if bx.shape[0] < 2:
                continue  # Avoid BatchNorm issues on tiny batches.
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

        # Validation and early stopping
        if n_val > 0 and early_stopping_patience > 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = torch.mean(w_val * (val_pred - z_val) ** 2).item()
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

    # Load best weights if early stopping was used
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model


def predict_tau_tarnet(model: TarNet, X: np.ndarray) -> np.ndarray:
    """Predict CATE from TarNet."""
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(DEVICE)
        y0, y1 = model(x_t)
        tau = (y1 - y0).squeeze().cpu().numpy()
    return tau


def predict_tau_rlearner(model: TauNet, X: np.ndarray) -> np.ndarray:
    """Predict CATE from TauNet."""
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(DEVICE)
        tau = model(x_t).cpu().numpy()
    return tau


class CTRLOrthogonalLearner:
    """
    Orthogonal meta-learner wrapper with gating and warm-start.

    Implements the three-stage CTRL-DML pipeline:
    - Stage 0: Plug-in warm-start
    - Stage 1: Cross-fitted nuisances
    - Stage 2: Orthogonal tau head with distillation
    """

    def __init__(self, config: Optional[CTRLConfig] = None):
        self.config = config or CTRLConfig()
        self.plugin_model: Optional[TarNet] = None
        self.tau_model: Optional[TauNet] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        T: np.ndarray,
        seed: int = 42
    ) -> "CTRLOrthogonalLearner":
        """
        Fit the CTRL-DML model.

        Args:
            X: Covariates (n_samples, n_features)
            y: Outcomes (n_samples,)
            T: Treatment indicators (n_samples,)
            seed: Random seed

        Returns:
            self
        """
        cfg = self.config

        # Stage 0: plug-in warm start (use hidden_tau for compatible architecture)
        self.plugin_model = train_plugin(
            X, y, T,
            use_gating=cfg.use_gating,
            lambda_sparsity=cfg.lambda_sparsity,
            seed=seed,
            dropout_p=cfg.dropout_p,
            hidden_dim=cfg.hidden_tau,
            batch_size=cfg.batch_size,
            epochs=cfg.plugin_epochs,
            lr=cfg.lr_plugin,
        )

        # Stage 1: cross-fitted nuisances
        m_hat, e_hat = cross_fit_nuisance(
            X, y, T,
            use_gating=cfg.use_gating,
            lambda_sparsity=cfg.lambda_sparsity,
            seed=seed,
            k_folds=cfg.k_folds,
            dropout_p=cfg.dropout_p,
            hidden_dim=cfg.hidden_dim,
            batch_size=cfg.batch_size,
            epochs=cfg.nuisance_epochs,
        )
        e_hat = np.clip(e_hat, 0.01, 0.99)
        R, W = y - m_hat, T - e_hat
        Z, weights = stabilize_residuals(R, W, w_clip=cfg.w_clip, z_clip=cfg.z_clip, eps=cfg.eps)

        # Stage 2: orthogonal tau head with optional distillation
        self.tau_model = train_rlearner(
            X, Z, weights,
            use_gating=cfg.use_gating,
            lambda_tau=cfg.lambda_tau,
            seed=seed,
            dropout_p=cfg.dropout_p,
            hidden_dim=cfg.hidden_tau,
            batch_size=cfg.batch_size,
            epochs=cfg.tau_epochs,
            lr=cfg.lr_tau,
            grad_clip=cfg.grad_clip,
            warm_start_from=self.plugin_model,
            teacher_tau=predict_tau_tarnet(self.plugin_model, X),
            aux_beta_start=cfg.aux_beta_start,
            aux_beta_end=cfg.aux_beta_end,
            aux_decay_epochs=cfg.aux_decay_epochs,
            freeze_backbone=cfg.freeze_backbone,
        )
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE using the orthogonal tau model."""
        if self.tau_model is None:
            raise ValueError("Model not fit. Call fit() first.")
        return predict_tau_rlearner(self.tau_model, X)

    def predict_plugin(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE using the plug-in model (non-orthogonal)."""
        if self.plugin_model is None:
            raise ValueError("Model not fit. Call fit() first.")
        return predict_tau_tarnet(self.plugin_model, X)

    def predict_ate(
        self,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """Predict ATE (optionally weighted)."""
        tau = self.predict_tau(X)
        if weights is None:
            return float(np.mean(tau))
        w = weights / np.sum(weights)
        return float(np.sum(w * tau))
