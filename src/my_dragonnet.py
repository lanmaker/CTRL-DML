import torch
from torch import nn
from typing import Any, Optional, Tuple
import numpy as np

# Import BasicDragonNet from the installed package
from catenets.models.torch.representation_nets import BasicDragonNet, DEFAULT_UNITS_OUT, DEFAULT_NONLIN, DEFAULT_UNITS_R, DEFAULT_PENALTY_L2, DEFAULT_STEP_SIZE, DEFAULT_N_ITER, DEFAULT_BATCH_SIZE, DEFAULT_VAL_SPLIT, DEFAULT_N_ITER_PRINT, DEFAULT_SEED, DEFAULT_N_ITER_MIN, DEFAULT_PATIENCE
from catenets.models.torch.base import PropensityNet, RepresentationNet, DEVICE

class TabularAttention(nn.Module):
    """
    CTRL-DML Core Innovation:
    A Gating Mechanism that learns to filter features before processing them.
    """
    def __init__(self, input_dim, output_dim, relaxation=1.5):
        super().__init__()
        # 1. Mask Generator (Learns feature weights)
        self.mask_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim), 
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid() # Output 0-1 weights
        )
        
        # 2. Feature Transformer (Processes masked features)
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ELU() 
        )
        
        # Store the latest mask penalty
        self.current_mask_penalty = 0.0

    def forward(self, x):
        # Step A: Generate Mask
        mask = self.mask_net(x)
        # Cache the latest mask for interpretability
        self.last_mask = mask.detach()
        
        # Calculate Sparsity Penalty (L1 Norm of the mask)
        # We want this to be small (sparse)
        self.current_mask_penalty = torch.mean(mask)
        
        # Step B: Apply Mask (Feature Selection)
        x_masked = x * mask
        
        # Step C: Transform
        out = self.feature_net(x_masked)
        return out

class MyDragonNet(BasicDragonNet):
    """
    Modified DragonNet with TabularAttention.
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
        
        # === SURGERY START ===
        # Replace the standard RepresentationNet with TabularAttention
        # The original self._repr_estimator is created in BasicDragonNet.__init__
        # We overwrite it here.
        
        # Note: RepresentationNet usually has multiple layers. 
        # Here we replace it with a sequence: TabularAttention -> Linear -> ELU
        # This matches the "200 -> 200" structure but with attention first.
        
        # === SURGERY START ===
        # Replace the standard RepresentationNet with TabularAttention
        # The original self._repr_estimator is created in BasicDragonNet.__init__
        # We overwrite it here.
        
        # Note: RepresentationNet usually has multiple layers. 
        # Here we replace it with a sequence: TabularAttention -> Linear -> ELU
        # This matches the "200 -> 200" structure but with attention first.
        
        self._repr_estimator = nn.Sequential(
            # Layer 1: Attention Mechanism
            TabularAttention(n_unit_in, n_units_r),
            nn.Dropout(p=dropout_prob), # <--- Added Dropout
            # Layer 2: Standard Non-linear transformation (to match depth)
            nn.Linear(n_units_r, n_units_r),
            nn.ELU(),
            nn.Dropout(p=dropout_prob)  # <--- Added Dropout
        ).to(DEVICE)
        
        # Modify outcome heads (y0_net, y1_net) to include Dropout
        # We overwrite self._po_estimators from BasicDragonNet
        self._po_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_units_r, 100),
                nn.ELU(),
                nn.Dropout(p=dropout_prob), # <--- Added Dropout
                nn.Linear(100, 1)
            ).to(DEVICE),
            nn.Sequential(
                nn.Linear(n_units_r, 100),
                nn.ELU(),
                nn.Dropout(p=dropout_prob), # <--- Added Dropout
                nn.Linear(100, 1)
            ).to(DEVICE)
        ])
        
        # === SURGERY END ===

    def predict_with_uncertainty(self, X, n_runs=50):
        """
        MC Dropout for Uncertainty Quantification
        X: Input data (numpy array or tensor)
        n_runs: Number of forward passes
        """
        # 1. Force train mode to enable Dropout
        self.train() 
        
        # Handle input
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(DEVICE)
        
        # 2. Multiple forward passes
        preds_cate = []
        
        with torch.no_grad(): # No gradients needed for inference
            for _ in range(n_runs):
                # Manual forward pass through our components
                repr_preds = self._repr_estimator(X).squeeze()
                y0_preds = self._po_estimators[0](repr_preds).squeeze()
                y1_preds = self._po_estimators[1](repr_preds).squeeze()
                
                # Calculate CATE = y1 - y0
                cate = y1_preds - y0_preds
                preds_cate.append(cate.cpu().numpy())
        
        # 3. Restore eval mode
        self.eval()
        
        # 4. Statistical analysis
        preds_cate = np.array(preds_cate) # shape: (n_runs, n_samples)
        
        mean_cate = preds_cate.mean(axis=0) # Mean CATE
        std_cate = preds_cate.std(axis=0)   # Uncertainty (Std Dev)
        
        # 95% Confidence Interval: Mean ± 1.96 * Std
        lower_bound = mean_cate - 1.96 * std_cate
        upper_bound = mean_cate + 1.96 * std_cate
        
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
