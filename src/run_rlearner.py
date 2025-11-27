import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import KFold
from scipy.special import expit

from my_dragonnet import MyDragonNet, TabularAttention
from catenets.models.torch.base import DEVICE


def get_stress_data_dynamic(n_samples: int = 2000, n_noise: int = 50, seed: int = 42):
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
    return X, T, y, true_te


def train_nuisance_model(X, y, T, lambda_sparsity=0.072, dropout_p=0.35, epochs=400, batch_size=256):
    X_t = torch.from_numpy(X).float().to(DEVICE)
    y_t = torch.from_numpy(y).float().to(DEVICE)
    T_t = torch.from_numpy(T).long().to(DEVICE)

    model = MyDragonNet(n_unit_in=X.shape[1], n_iter=1, batch_size=batch_size, dropout_prob=dropout_p).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.003)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(X_t.shape[0])
        for i in range(0, X_t.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            bx, by, bt = X_t[idx], y_t[idx], T_t[idx]
            opt.zero_grad()
            po_preds, prop_preds, discr = model._step(bx, bt)
            base_loss = model.loss(po_preds, prop_preds, by, bt, discr)
            reg_loss = model._repr_estimator[0].current_mask_penalty
            loss = base_loss + lambda_sparsity * reg_loss
            loss.backward()
            opt.step()
    return model


def cross_fit_nuisance(X, T, y, k_folds=5, seed=42):
    N = X.shape[0]
    m_hat = np.zeros(N, dtype=np.float32)
    e_hat = np.zeros(N, dtype=np.float32)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for train_idx, val_idx in kf.split(X):
        model = train_nuisance_model(X[train_idx], y[train_idx], T[train_idx])
        model.eval()
        with torch.no_grad():
            x_val_t = torch.from_numpy(X[val_idx]).float().to(DEVICE)
            po_preds, prop_preds, _ = model._step(x_val_t, torch.from_numpy(T[val_idx]).long().to(DEVICE))
            y0 = po_preds[:, 0].cpu().numpy()
            y1 = po_preds[:, 1].cpu().numpy()
            ps = torch.softmax(prop_preds, dim=1)[:, 1].cpu().numpy()
            m_hat[val_idx] = ps * y1 + (1 - ps) * y0
            e_hat[val_idx] = ps
    return m_hat, e_hat


class CATEModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 200, dropout_p: float = 0.35):
        super().__init__()
        self.attn = TabularAttention(input_dim, hidden)
        self.backbone = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(p=dropout_p),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.attn(x)
        h = self.backbone(h)
        return self.head(h).squeeze(-1)

    @property
    def mask_penalty(self):
        return self.attn.current_mask_penalty


def train_r_learner(X, R, W, lambda_tau=0.01, dropout_p=0.35, epochs=400, batch_size=256):
    X_t = torch.from_numpy(X).float().to(DEVICE)
    R_t = torch.from_numpy(R).float().to(DEVICE)
    W_t = torch.from_numpy(W).float().to(DEVICE)

    model = CATEModel(X.shape[1], hidden=200, dropout_p=dropout_p).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.003)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(X_t.shape[0])
        for i in range(0, X_t.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            bx, bR, bW = X_t[idx], R_t[idx], W_t[idx]
            opt.zero_grad()
            tau_pred = model(bx)
            loss_orth = torch.mean((bR - bW * tau_pred) ** 2)
            loss = loss_orth + lambda_tau * model.mask_penalty
            loss.backward()
            opt.step()
    return model


def evaluate_pehe(model, X, true_te):
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(DEVICE)
        tau_pred = model(x_t).cpu().numpy()
    return float(np.sqrt(np.mean((true_te - tau_pred) ** 2)))


if __name__ == "__main__":
    X, T, y, true_te = get_stress_data_dynamic(n_samples=2000, n_noise=50, seed=123)
    print("Stage 1: Cross-fitting nuisances...")
    m_hat, e_hat = cross_fit_nuisance(X, T, y, k_folds=5, seed=123)

    R = y - m_hat
    W = T - e_hat

    print("Stage 2: Training R-learner CATE network...")
    cate_model = train_r_learner(X, R, W, lambda_tau=0.01, dropout_p=0.35)

    pehe = evaluate_pehe(cate_model, X, true_te)
    print(f"PEHE (R-learner with gating): {pehe:.3f}")
