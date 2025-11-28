"""High-level CTRL-DML orthogonal learner wrapper.

This class packages the three-stage pipeline:
    1) Plug-in warm-start (TARNet-style head with gating)
    2) Cross-fitted nuisances (outcome/propensity heads)
    3) Orthogonal tau head (R-learner ratio loss with clipping + optional distillation)

It mirrors the algorithm in the paper and reuses the primitives from run_ablation.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from run_ablation import (
    train_plugin,
    cross_fit_nuisance,
    stabilize_residuals,
    train_rlearner,
    predict_tau_tarnet,
    predict_tau_rlearner,
)


@dataclass
class CTRLConfig:
    use_gating: bool = True
    lambda_sparsity: float = 0.05
    hidden_dim: int = 120
    hidden_tau: int = 96
    dropout_p: float = 0.4
    batch_size: int = 192
    plugin_epochs: int = 220
    nuisance_epochs: int = 150
    tau_epochs: int = 300
    lr_plugin: float = 3e-3
    lr_tau: float = 3e-4
    k_folds: int = 3
    lambda_tau: float = 5e-5
    grad_clip: float = 1.0
    z_clip: float = 5.0
    w_clip: float = 0.05
    aux_beta_start: float = 0.8
    aux_beta_end: float = 0.3
    aux_decay_epochs: int = 200
    freeze_backbone: bool = True


class CTRLOrthogonalLearner:
    """Orthogonal meta-learner wrapper with gating and warm-start."""

    def __init__(self, config: Optional[CTRLConfig] = None):
        self.config = config or CTRLConfig()
        self.plugin_model = None
        self.tau_model = None

    def fit(self, X: np.ndarray, y: np.ndarray, T: np.ndarray, seed: int = 42) -> "CTRLOrthogonalLearner":
        cfg = self.config
        # Stage 0: plug-in warm start
        self.plugin_model = train_plugin(
            X,
            y,
            T,
            use_gating=cfg.use_gating,
            lambda_sparsity=cfg.lambda_sparsity,
            seed=seed,
            dropout_p=cfg.dropout_p,
            hidden_dim=cfg.hidden_dim,
            batch_size=cfg.batch_size,
            epochs=cfg.plugin_epochs,
            lr=cfg.lr_plugin,
        )
        # Stage 1: cross-fitted nuisances
        m_hat, e_hat = cross_fit_nuisance(
            X,
            y,
            T,
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
        Z, weights = stabilize_residuals(R, W, clip_w=cfg.w_clip, z_clip=cfg.z_clip)
        # Stage 2: orthogonal tau head with optional distillation
        self.tau_model = train_rlearner(
            X,
            Z,
            weights,
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
        if self.tau_model is None:
            raise ValueError("Model not fit. Call fit() first.")
        return predict_tau_rlearner(self.tau_model, X)

    def predict_plugin(self, X: np.ndarray) -> np.ndarray:
        if self.plugin_model is None:
            raise ValueError("Model not fit. Call fit() first.")
        return predict_tau_tarnet(self.plugin_model, X)

    def predict_ate(self, X: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        tau = self.predict_tau(X)
        if weights is None:
            return float(np.mean(tau))
        w = weights / np.sum(weights)
        return float(np.sum(w * tau))
