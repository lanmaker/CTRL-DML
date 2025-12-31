"""
Common training utilities for CTRL-DML.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Dict, Any
from dataclasses import dataclass


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 200
    batch_size: int = 192
    lr: float = 0.003
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stopping_patience: int = 20
    val_split: float = 0.2


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = "min"):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None

    def __call__(self, score: float, model: Optional[nn.Module] = None) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current score
            model: Model to save best weights

        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            if model is not None:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def load_best_weights(self, model: nn.Module) -> None:
        """Load the best weights back into model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    loss_fn: Callable,
    grad_clip: float = 0.0,
    device: torch.device = DEVICE,
) -> float:
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        X: Input features
        y: Target
        batch_size: Batch size
        loss_fn: Loss function (takes model, X_batch, y_batch, returns loss)
        grad_clip: Gradient clipping norm (0 to disable)
        device: Device

    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    perm = torch.randperm(X.shape[0])

    for i in range(0, X.shape[0], batch_size):
        idx = perm[i:i + batch_size]
        X_batch = X[idx].to(device)
        y_batch = y[idx].to(device) if y is not None else None

        optimizer.zero_grad()
        loss = loss_fn(model, X_batch, y_batch)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def eval_epoch(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    loss_fn: Callable,
    device: torch.device = DEVICE,
) -> float:
    """
    Evaluate model.

    Args:
        model: PyTorch model
        X: Input features
        y: Target
        batch_size: Batch size
        loss_fn: Loss function
        device: Device

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size].to(device)
            y_batch = y[i:i + batch_size].to(device) if y is not None else None

            loss = loss_fn(model, X_batch, y_batch)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def stabilize_residuals(
    R: np.ndarray,
    W: np.ndarray,
    clip_w: float = 0.05,
    z_clip: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stabilize residuals for orthogonal R-learner.

    Returns pseudo-outcome Z and weights s for the ratio R-learner.

    Args:
        R: Outcome residuals (Y - m_hat)
        W: Treatment residuals (T - e_hat)
        clip_w: Clipping threshold for W
        z_clip: Clipping threshold for Z

    Returns:
        Tuple of (Z, weights)
    """
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


def numpy_to_tensor(arr: np.ndarray, device: torch.device = DEVICE) -> torch.Tensor:
    """Convert numpy array to tensor on device."""
    return torch.from_numpy(arr).float().to(device)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array."""
    return t.detach().cpu().numpy()


class LinearWarmupCosineDecay:
    """Learning rate scheduler with linear warmup and cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        """Update learning rate."""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            scale = self.current_epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1 + np.cos(np.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = max(self.min_lr, base_lr * scale)

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
