import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import expit

from catenets.models.torch import DragonNet as StandardDragonNet
from my_dragonnet import MyDragonNet
from catenets.models.torch.base import DEVICE


def set_seed(seed: int = 7):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data(n: int = 1800, n_noise: int = 15, seed: int = 7):
    set_seed(seed)
    C = np.random.normal(0, 1, size=(n, 5))
    I = np.random.normal(0, 1, size=(n, 5))
    N = np.random.normal(0, 1, size=(n, n_noise))
    X = np.concatenate([C, I, N], axis=1)
    logit = (np.sum(C, axis=1) + 0.5 * C[:, 0] * C[:, 1] + 2 * np.sum(I, axis=1))
    prop = expit(logit)
    T = np.random.binomial(1, prop)
    true_te = 2 * np.sin(C[:, 0] * np.pi) + np.maximum(0, C[:, 1])
    y0 = np.sum(C, axis=1) ** 2 + np.random.normal(0, 0.5, size=n)
    y = y0 + true_te * T
    return X, T, y


def get_propensity_dragon(model, X_tensor):
    with torch.no_grad():
        repr_preds = model._repr_estimator(X_tensor).squeeze()
        logits = model._propensity_estimator(repr_preds)
        ps = torch.softmax(logits, dim=1)[:, 1]
    return ps.cpu().numpy()


def get_propensity_ctrl(model, X_tensor):
    with torch.no_grad():
        repr_preds = model._repr_estimator(X_tensor).squeeze()
        logits = model._propensity_estimator(repr_preds)
        ps = torch.softmax(logits, dim=1)[:, 1]
    return ps.cpu().numpy()


def compute_smd(X, T, weights=None):
    if weights is None:
        weights = np.ones_like(T, dtype=float)
    treat_mask = T == 1
    control_mask = T == 0

    w_t = weights[treat_mask]
    w_c = weights[control_mask]
    X_t = X[treat_mask]
    X_c = X[control_mask]

    mean_t = np.average(X_t, axis=0, weights=w_t)
    mean_c = np.average(X_c, axis=0, weights=w_c)
    var_t = np.average((X_t - mean_t) ** 2, axis=0, weights=w_t)
    var_c = np.average((X_c - mean_c) ** 2, axis=0, weights=w_c)
    pooled_sd = np.sqrt(0.5 * (var_t + var_c) + 1e-8)
    smd = np.abs(mean_t - mean_c) / pooled_sd
    return smd


def main():
    X, T, y = make_data()
    feature_names = [f"C{i}" for i in range(5)] + [f"I{i}" for i in range(5)] + [f"N{i}" for i in range(5)]

    X_tensor = torch.from_numpy(X).float().to(DEVICE)
    y_tensor = torch.from_numpy(y).float().to(DEVICE)
    T_tensor = torch.from_numpy(T).long().to(DEVICE)

    # DragonNet baseline
    dragon = StandardDragonNet(n_unit_in=X.shape[1], n_iter=400, batch_size=256)
    dragon.fit(X, y, T)
    ps_dragon = get_propensity_dragon(dragon, X_tensor)

    # CTRL-DML
    ctrl = MyDragonNet(n_unit_in=X.shape[1], n_iter=1, batch_size=256, dropout_prob=0.35).to(DEVICE)
    optim_ctrl = torch.optim.Adam(ctrl.parameters(), lr=0.003)
    lambda_sparsity = 0.072
    ctrl.train()
    for _ in range(320):
        perm = torch.randperm(X_tensor.shape[0])
        for i in range(0, X_tensor.shape[0], 256):
            idx = perm[i : i + 256]
            batch_X, batch_y, batch_T = X_tensor[idx], y_tensor[idx], T_tensor[idx]
            optim_ctrl.zero_grad()
            po_preds, prop_preds, discr = ctrl._step(batch_X, batch_T)
            base_loss = ctrl.loss(po_preds, prop_preds, batch_y, batch_T, discr)
            reg_loss = ctrl._repr_estimator[0].current_mask_penalty
            loss = base_loss + lambda_sparsity * reg_loss
            loss.backward()
            optim_ctrl.step()
    ps_ctrl = get_propensity_ctrl(ctrl, X_tensor)

    # Stabilized IPTW weights
    eps = 1e-3
    ps_dragon = np.clip(ps_dragon, eps, 1 - eps)
    ps_ctrl = np.clip(ps_ctrl, eps, 1 - eps)
    w_dragon = T / ps_dragon + (1 - T) / (1 - ps_dragon)
    w_ctrl = T / ps_ctrl + (1 - T) / (1 - ps_ctrl)

    smd_unweighted = compute_smd(X[:, :15], T)
    smd_dragon = compute_smd(X[:, :15], T, weights=w_dragon)
    smd_ctrl = compute_smd(X[:, :15], T, weights=w_ctrl)

    idx = np.argsort(smd_unweighted)[::-1]
    y_pos = np.arange(len(feature_names))

    plt.figure(figsize=(8, 6))
    plt.axvline(0.1, color="black", linestyle="--", linewidth=1, label="|SMD|=0.1")
    plt.scatter(smd_unweighted[idx], y_pos, label="Unweighted", color="gray", marker="o")
    plt.scatter(smd_dragon[idx], y_pos, label="DragonNet-weighted", color="red", marker="s")
    plt.scatter(smd_ctrl[idx], y_pos, label="CTRL-DML-weighted", color="green", marker="^")
    plt.yticks(y_pos, np.array(feature_names)[idx])
    plt.xlabel("Standardized Mean Difference (|SMD|)")
    plt.title("Covariate Balance (Love Plot)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("balance_plot.pdf", dpi=300)
    print("Saved balance_plot.pdf")


if __name__ == "__main__":
    main()
