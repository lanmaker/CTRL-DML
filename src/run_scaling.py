import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim
from scipy.special import expit

from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from my_dragonnet import MyDragonNet as CTRLDML
from catenets.models.torch.base import DEVICE

# Configuration (supports a fast mode via CTRL_DML_FAST=1)
FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"
SAMPLE_SIZES = [2000, 5000, 10000, 20000] if not FAST_RUN else [2000, 5000]
FIXED_NOISE = 50
SEEDS = [42, 1024]
CTRL_EPOCHS = 600 if not FAST_RUN else 300
BATCH_SIZE = 512
CF_TREES = 100 if not FAST_RUN else 48
LAMBDA_SPARSITY = 0.072


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_by_size(n_samples: int, n_noise: int = FIXED_NOISE):
    C = np.random.normal(0, 1, size=(n_samples, 5))
    I = np.random.normal(0, 1, size=(n_samples, 5))
    N = np.random.normal(0, 1, size=(n_samples, n_noise))
    X = np.concatenate([C, I, N], axis=1)
    logit = np.sum(C, axis=1) + 0.5 * C[:, 0] * C[:, 1] + 2 * np.sum(I, axis=1)
    propensity = expit(logit)
    T = np.random.binomial(1, propensity)
    true_te = 2 * np.sin(C[:, 0] * np.pi) + np.maximum(0, C[:, 1])
    y0 = np.sum(C, axis=1) ** 2 + np.random.normal(0, 0.5, size=n_samples)
    y = y0 + true_te * T
    return X, T, y, true_te


def train_ctrl_dml(X_np, y_np, T_np):
    X = torch.from_numpy(X_np).float().to(DEVICE)
    y = torch.from_numpy(y_np).float().to(DEVICE)
    T = torch.from_numpy(T_np).long().to(DEVICE)

    model = CTRLDML(n_unit_in=X.shape[1], n_iter=1, batch_size=BATCH_SIZE, dropout_prob=0.35).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    model.train()
    for epoch in range(CTRL_EPOCHS):
        permutation = torch.randperm(X.shape[0])
        for i in range(0, X.shape[0], BATCH_SIZE):
            indices = permutation[i : i + BATCH_SIZE]
            batch_X, batch_y, batch_T = X[indices], y[indices], T[indices]
            optimizer.zero_grad()
            po_preds, prop_preds, discr = model._step(batch_X, batch_T)
            base_loss = model.loss(po_preds, prop_preds, batch_y, batch_T, discr)
            reg_loss = model._repr_estimator[0].current_mask_penalty
            total_loss = base_loss + (LAMBDA_SPARSITY * reg_loss)
            total_loss.backward()
            optimizer.step()
    return model


def main():
    res_cf = np.zeros((len(SAMPLE_SIZES), len(SEEDS)))
    res_ours = np.zeros((len(SAMPLE_SIZES), len(SEEDS)))

    print(f"Starting scaling benchmark (N={SAMPLE_SIZES}) with noise={FIXED_NOISE}.")
    if FAST_RUN:
        print("Fast mode enabled (CTRL_DML_FAST=1): reduced sample sizes, epochs, and tree count.")

    for i, n in enumerate(SAMPLE_SIZES):
        print(f"\n>>> Testing sample size N={n}")
        for j, seed in enumerate(SEEDS):
            set_seed(seed)
            X, T, y, true_te = get_data_by_size(n, n_noise=FIXED_NOISE)

            # Causal Forest baseline (LogisticRegressionCV for treatment to avoid warnings)
            est = CausalForestDML(
                model_y=LassoCV(),
                model_t=LogisticRegressionCV(max_iter=1000),
                n_estimators=CF_TREES,
                discrete_treatment=True,
                random_state=seed,
            )
            est.fit(y, T, X=X)
            p_cf = est.effect(X)
            pehe_cf = np.sqrt(np.mean((true_te - p_cf.flatten()) ** 2))
            res_cf[i, j] = pehe_cf

            # CTRL-DML
            m_ours = train_ctrl_dml(X, y, T)
            m_ours.eval()
            with torch.no_grad():
                pred_ours_tensor = m_ours.predict(torch.from_numpy(X).float().to(DEVICE))
                p_ours = pred_ours_tensor.cpu().detach().numpy().flatten()
            pehe_ours = np.sqrt(np.mean((true_te - p_ours) ** 2))
            res_ours[i, j] = pehe_ours

            print(f"   Seed {seed} | CF: {pehe_cf:.4f} | Ours: {pehe_ours:.4f}")

    # Plotting
    cf_mean = np.mean(res_cf, axis=1)
    cf_std = np.std(res_cf, axis=1)
    ours_mean = np.mean(res_ours, axis=1)
    ours_std = np.std(res_ours, axis=1)
    offsets = np.linspace(-0.08, 0.08, len(SEEDS))

    plt.figure(figsize=(10, 6))
    plt.plot(SAMPLE_SIZES, cf_mean, label="Causal Forest", color="#1f77b4", marker="^", linestyle="--")
    plt.fill_between(SAMPLE_SIZES, cf_mean - cf_std, cf_mean + cf_std, color="#1f77b4", alpha=0.12)
    for idx, _ in enumerate(SEEDS):
        plt.scatter(np.array(SAMPLE_SIZES) * (1 + offsets[idx]), res_cf[:, idx], color="#1f77b4", alpha=0.4, s=40, label="_nolegend_")

    plt.plot(SAMPLE_SIZES, ours_mean, label="CTRL-DML (Ours)", color="#2ca02c", marker="s", linestyle="-", linewidth=2.5)
    plt.fill_between(SAMPLE_SIZES, ours_mean - ours_std, ours_mean + ours_std, color="#2ca02c", alpha=0.18)
    for idx, _ in enumerate(SEEDS):
        plt.scatter(np.array(SAMPLE_SIZES) * (1 + offsets[idx]), res_ours[:, idx], color="#2ca02c", alpha=0.4, s=40, label="_nolegend_")

    plt.xscale("log")
    plt.xlabel("Sample Size (N, log scale)")
    plt.ylabel("PEHE Error (Lower is Better)")
    plt.title(f"Scaling Analysis: Deep Learning vs Trees (Noise={FIXED_NOISE})")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    output_path = "scaling_results.pdf"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nScaling benchmark complete. Results saved to {output_path}")

    delta = res_ours - res_cf  # Negative means CTRL-DML beats Causal Forest
    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)

    plt.figure(figsize=(9, 5.5))
    plt.axhline(0, color="gray", linestyle=":", linewidth=1)
    plt.plot(SAMPLE_SIZES, delta_mean, color="#9467bd", marker="o", linewidth=2, label="CTRL-DML minus Causal Forest")
    plt.fill_between(SAMPLE_SIZES, delta_mean - delta_std, delta_mean + delta_std, color="#9467bd", alpha=0.15)
    for idx, _ in enumerate(SEEDS):
        plt.scatter(np.array(SAMPLE_SIZES) * (1 + offsets[idx]), delta[:, idx], color="#9467bd", alpha=0.5, s=45, label="_nolegend_")
    plt.xscale("log")
    plt.xlabel("Sample Size (N, log scale)")
    plt.ylabel("PEHE Delta (negative favors CTRL-DML)")
    plt.title("Gap vs. Causal Forest across sample sizes")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    delta_path = "scaling_delta.pdf"
    plt.tight_layout()
    plt.savefig(delta_path, dpi=300)
    print(f"Delta view saved to {delta_path}")


if __name__ == "__main__":
    main()
