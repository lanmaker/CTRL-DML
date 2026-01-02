"""
CTRL-DML Core Models: TabularAttention and DragonNet variants.

This module contains:
- TabularAttention: Gating mechanism for feature selection
- MyDragonNet: DragonNet with TabularAttention
- BaseBackbone: Shared backbone with optional gating
- NuisanceNet: Joint outcome + propensity network
- TauNet: CATE network for orthogonal R-learner
- TarNet: Direct plug-in estimator
"""
import torch
from torch import nn
from typing import Any, Optional, Tuple
import numpy as np

# Import from catenets for base DragonNet
try:
    from catenets.models.torch.representation_nets import (
        BasicDragonNet,
        DEFAULT_UNITS_OUT,
        DEFAULT_NONLIN,
        DEFAULT_UNITS_R,
    )
    from catenets.models.torch.base import PropensityNet, DEVICE
    CATENETS_AVAILABLE = True
except ImportError:
    CATENETS_AVAILABLE = False
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularAttention(nn.Module):
    """
    CTRL-DML Core Innovation:
    A Gating Mechanism that learns to filter features before processing them.

    The mask generator learns feature weights in [0, 1] which are multiplied
    element-wise with input features. An L1 penalty on the mask encourages
    sparsity, helping the model focus on true confounders.
    """

    def __init__(self, input_dim: int, output_dim: int, relaxation: float = 1.5):
        super().__init__()
        # 1. Mask Generator (Learns feature weights)
        self.mask_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()  # Output 0-1 weights
        )

        # 2. Feature Transformer (Processes masked features)
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ELU()
        )

        # Store the latest mask penalty
        self.current_mask_penalty = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step A: Generate Mask
        mask = self.mask_net(x)
        # Cache the latest mask for interpretability
        self.last_mask = mask.detach()

        # Calculate Sparsity Penalty (L1 Norm of the mask)
        self.current_mask_penalty = torch.mean(mask)

        # Step B: Apply Mask (Feature Selection)
        x_masked = x * mask

        # Step C: Transform
        out = self.feature_net(x_masked)
        return out


class BaseBackbone(nn.Module):
    """Shared backbone that can optionally apply TabularAttention."""

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        dropout_p: float,
        use_gating: bool
    ):
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
        return torch.tensor(0.0, device=next(self.parameters()).device)


class NuisanceNet(nn.Module):
    """Joint outcome + propensity network used for cross-fitting."""

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        dropout_p: float,
        use_gating: bool
    ):
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

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        dropout_p: float,
        use_gating: bool
    ):
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

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        dropout_p: float,
        use_gating: bool
    ):
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


# Only define MyDragonNet if catenets is available
if CATENETS_AVAILABLE:
    class MyDragonNet(BasicDragonNet):
        """
        Modified DragonNet with TabularAttention.

        Extends BasicDragonNet from catenets with:
        - TabularAttention gating in representation layer
        - MC Dropout for uncertainty quantification
        """

        def __init__(
            self,
            n_unit_in: int,
            binary_y: bool = False,
            n_units_out_prop: int = DEFAULT_UNITS_OUT,
            n_layers_out_prop: int = 0,
            nonlin: str = DEFAULT_NONLIN,
            n_units_r: int = DEFAULT_UNITS_R,
            batch_norm: bool = True,
            dropout: bool = False,
            dropout_prob: float = 0.2,
            **kwargs: Any,
        ) -> None:
            propensity_estimator = PropensityNet(
                "dragonnet_propensity_estimator",
                n_units_r,
                2,
                "prop",
                n_layers_out_prop=n_layers_out_prop,
                n_units_out_prop=n_units_out_prop,
                nonlin=nonlin,
                batch_norm=batch_norm,
                dropout=dropout,
                dropout_prob=dropout_prob,
            ).to(DEVICE)

            super().__init__(
                "DragonNet",
                n_unit_in,
                propensity_estimator,
                binary_y=binary_y,
                nonlin=nonlin,
                batch_norm=batch_norm,
                dropout=dropout,
                dropout_prob=dropout_prob,
                **kwargs,
            )

            # Replace the standard RepresentationNet with TabularAttention
            self._repr_estimator = nn.Sequential(
                TabularAttention(n_unit_in, n_units_r),
                nn.Dropout(p=dropout_prob),
                nn.Linear(n_units_r, n_units_r),
                nn.ELU(),
                nn.Dropout(p=dropout_prob)
            ).to(DEVICE)

            # Modify outcome heads to include Dropout
            self._po_estimators = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(n_units_r, 100),
                    nn.ELU(),
                    nn.Dropout(p=dropout_prob),
                    nn.Linear(100, 1)
                ).to(DEVICE),
                nn.Sequential(
                    nn.Linear(n_units_r, 100),
                    nn.ELU(),
                    nn.Dropout(p=dropout_prob),
                    nn.Linear(100, 1)
                ).to(DEVICE)
            ])

        def predict_with_uncertainty(
            self,
            X: np.ndarray,
            n_runs: int = 50,
            alpha: float = 0.05
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """
            MC Dropout for Uncertainty Quantification.

            Uses percentile-based confidence intervals which are more appropriate
            for potentially non-normal MC dropout distributions.

            Args:
                X: Input data (numpy array or tensor)
                n_runs: Number of forward passes
                alpha: Significance level for CI (default 0.05 for 95% CI)

            Returns:
                Tuple of (mean_cate, std_cate, lower_bound, upper_bound)
            """
            original_mode = self.training
            self.eval()
            dropout_layers = [m for m in self.modules() if isinstance(m, nn.Dropout)]
            for layer in dropout_layers:
                layer.train(True)

            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float().to(DEVICE)

            preds_cate = []

            with torch.no_grad():
                for _ in range(n_runs):
                    repr_preds = self._repr_estimator(X).squeeze()
                    y0_preds = self._po_estimators[0](repr_preds).squeeze()
                    y1_preds = self._po_estimators[1](repr_preds).squeeze()
                    cate = y1_preds - y0_preds
                    preds_cate.append(cate.cpu().numpy())

            self.train(original_mode)

            preds_cate = np.array(preds_cate)
            mean_cate = preds_cate.mean(axis=0)
            std_cate = preds_cate.std(axis=0)
            # Use percentiles for CI (more robust than mean +/- 1.96*std)
            lower_bound = np.percentile(preds_cate, 100 * alpha / 2, axis=0)
            upper_bound = np.percentile(preds_cate, 100 * (1 - alpha / 2), axis=0)

            return mean_cate, std_cate, lower_bound, upper_bound

        def _step(
            self, X: torch.Tensor, w: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            repr_preds = self._repr_estimator(X).squeeze()
            y0_preds = self._po_estimators[0](repr_preds).squeeze()
            y1_preds = self._po_estimators[1](repr_preds).squeeze()
            po_preds = torch.vstack((y0_preds, y1_preds)).T
            prop_preds = self._propensity_estimator(repr_preds)
            return po_preds, prop_preds, self._maximum_mean_discrepancy(repr_preds, w)
else:
    # Placeholder if catenets not installed
    class MyDragonNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("catenets package required for MyDragonNet. Install with: pip install catenets")


# Re-export get_device from training utils for backward compatibility
from src.utils.training import get_device
