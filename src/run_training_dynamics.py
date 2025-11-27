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


def make_data(n: int = 1200, n_noise: int = 30, seed: int = 123):
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


def main():
    X_np, T_np, y_np = make_data()

    X = torch.from_numpy(X_np).float().to(DEVICE)
    y = torch.from_numpy(y_np).float().to(DEVICE)
    T = torch.from_numpy(T_np).long().to(DEVICE)

    model = MyDragonNet(
        n_unit_in=X.shape[1],
        n_iter=1,
        batch_size=256,
        dropout_prob=0.35,
    ).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.003)
    lambda_sparsity = 0.072

    epochs = 320
    log_every = 5
    history = {"epoch": [], "confounder": [], "instrument": [], "noise": []}

    for epoch in range(epochs):
        perm = torch.randperm(X.shape[0])
        for i in range(0, X.shape[0], 256):
            idx = perm[i : i + 256]
            batch_X, batch_y, batch_T = X[idx], y[idx], T[idx]
            opt.zero_grad()
            po_preds, prop_preds, discr = model._step(batch_X, batch_T)
            base_loss = model.loss(po_preds, prop_preds, batch_y, batch_T, discr)
            reg_loss = model._repr_estimator[0].current_mask_penalty
            loss = base_loss + lambda_sparsity * reg_loss
            loss.backward()
            opt.step()

        if epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                masks = model._repr_estimator[0].mask_net(X).cpu().numpy()
            model.train()
            conf = masks[:, :5].mean()
            inst = masks[:, 5:10].mean()
            noise = masks[:, 10:].mean()
            history["epoch"].append(epoch)
            history["confounder"].append(conf)
            history["instrument"].append(inst)
            history["noise"].append(noise)
            print(f"Epoch {epoch}: conf={conf:.3f}, inst={inst:.3f}, noise={noise:.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["confounder"], label="Confounders", color="green", linewidth=2)
    plt.plot(history["epoch"], history["instrument"], label="Instruments", color="red", linestyle="--", linewidth=2)
    plt.plot(history["epoch"], history["noise"], label="Noise", color="gray", linestyle=":", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean attention weight")
    plt.title("Training Dynamics of CTRL-DML Mask")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_dynamics.pdf", dpi=300)
    print("Saved training_dynamics.pdf")


if __name__ == "__main__":
    main()
