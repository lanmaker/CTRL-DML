import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim
from scipy.special import expit

from my_dragonnet import MyDragonNet
from catenets.models.torch.base import DEVICE


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data(n: int = 800, n_noise: int = 20, seed: int = 42):
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
    return X, T, y, true_te


def train_once(X_np, y_np, T_np, lam: float, dropout_p: float, epochs: int = 90):
    X = torch.from_numpy(X_np).float().to(DEVICE)
    y = torch.from_numpy(y_np).float().to(DEVICE)
    T = torch.from_numpy(T_np).long().to(DEVICE)

    model = MyDragonNet(
        n_unit_in=X.shape[1],
        n_iter=1,
        batch_size=256,
        dropout_prob=dropout_p,
    ).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.003)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(X.shape[0])
        for i in range(0, X.shape[0], 256):
            idx = perm[i : i + 256]
            batch_X, batch_y, batch_T = X[idx], y[idx], T[idx]
            opt.zero_grad()
            po_preds, prop_preds, discr = model._step(batch_X, batch_T)
            base_loss = model.loss(po_preds, prop_preds, batch_y, batch_T, discr)
            reg_loss = model._repr_estimator[0].current_mask_penalty
            loss = base_loss + lam * reg_loss
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model.predict(torch.from_numpy(X_np).float().to(DEVICE))
    cate_pred = pred.cpu().numpy().flatten()
    return cate_pred


def compute_pehe(true_te, cate_pred):
    return float(np.sqrt(np.mean((true_te - cate_pred) ** 2)))


def main():
    X, T, y, true_te = make_data()

    lambdas = np.linspace(0.01, 0.1, 5)
    dropouts = np.linspace(0.1, 0.5, 5)
    heatmap = np.zeros((len(dropouts), len(lambdas)))

    print("Running sensitivity sweep for lambda (x) vs dropout (y)...")
    for i, p in enumerate(dropouts):
        for j, lam in enumerate(lambdas):
            cate_pred = train_once(X, y, T, lam=lam, dropout_p=p)
            pehe = compute_pehe(true_te, cate_pred)
            heatmap[i, j] = pehe
            print(f"  p={p:.2f}, lambda={lam:.3f} -> PEHE={pehe:.3f}")

    plt.figure(figsize=(8, 5))
    im = plt.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[lambdas.min(), lambdas.max(), dropouts.min(), dropouts.max()],
    )
    plt.colorbar(im, label="PEHE (lower is better)")
    plt.xlabel("Sparsity lambda")
    plt.ylabel("Dropout rate")
    plt.title("Sensitivity Heatmap: CTRL-DML Robustness")
    plt.tight_layout()
    plt.savefig("sensitivity_heatmap.pdf", dpi=300)
    print("Saved sensitivity_heatmap.pdf")


if __name__ == "__main__":
    main()
