import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_ablation import (
    get_stress_data,
    train_plugin,
    cross_fit_nuisance,
    stabilize_residuals,
    train_rlearner,
    predict_tau_rlearner,
    predict_tau_tarnet,
    evaluate_pehe,
)


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "CTRL-DML-Paper"


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def run_once(n_samples: int, n_noise: int, seed: int, drop_conf: bool) -> tuple[float, float]:
    """Train plug-in TARNet and orthogonal head with/without a removed confounder."""
    set_seed(seed)
    X, T, y, true_te = get_stress_data(n_samples=n_samples, n_noise=n_noise, seed=seed)
    if drop_conf:
        X = X.copy()
        X[:, :5] = 0.0  # simulate missing all confounders

    plugin = train_plugin(
        X,
        y,
        T,
        use_gating=True,
        lambda_sparsity=0.05,
        seed=seed,
        dropout_p=0.4,
        hidden_dim=96,
        batch_size=192,
        epochs=180,
        lr=0.003,
    )
    tau_plugin = predict_tau_tarnet(plugin, X)

    m_hat, e_hat = cross_fit_nuisance(
        X,
        y,
        T,
        use_gating=True,
        lambda_sparsity=0.05,
        seed=seed,
        k_folds=3,
        dropout_p=0.4,
        hidden_dim=120,
        batch_size=192,
        epochs=140,
    )
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R, W = y - m_hat, T - e_hat
    Z, weights = stabilize_residuals(R, W, clip_w=0.05, z_clip=5.0)
    tau_model = train_rlearner(
        X,
        Z,
        weights,
        use_gating=True,
        lambda_tau=5e-5,
        seed=seed,
        dropout_p=0.4,
        hidden_dim=96,
        batch_size=192,
        epochs=300,
        lr=3e-4,
        grad_clip=1.0,
        warm_start_from=plugin,
        teacher_tau=tau_plugin,
        aux_beta_start=0.6,
        aux_beta_end=0.2,
        aux_decay_epochs=200,
        freeze_backbone=True,
    )
    tau_dml = predict_tau_rlearner(tau_model, X)

    return evaluate_pehe(tau_plugin, true_te), evaluate_pehe(tau_dml, true_te)


def plot_results(full: tuple[float, float], dropped: tuple[float, float]) -> None:
    labels = ["Plug-in", "DML (orthogonal)"]
    vals_full = [full[0], full[1]]
    vals_drop = [dropped[0], dropped[1]]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - width / 2, vals_full, width, label="All confounders", color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, vals_drop, width, label="Drop all confounders", color="#ff7f0e", alpha=0.85)
    ymax = max(vals_full + vals_drop) + 0.25
    ax.set_ylim(0, ymax)
    for i, v in enumerate(vals_full):
        ax.text(x[i] - width / 2, v - 0.08, f"{v:.2f}", ha="center", va="top", color="white", fontweight="bold")
    for i, v in enumerate(vals_drop):
        ax.text(x[i] + width / 2, v - 0.08, f"{v:.2f}", ha="center", va="top", color="white", fontweight="bold")
    ax.set_ylabel("PEHE (lower is better)")
    ax.set_title("Bias/variance under missing confounder", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right", framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    for base in (ROOT, PAPER_DIR):
        out = base / "bias_variance.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Bias/variance when a confounder is dropped.")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-noise", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Training with all confounders...")
    full = run_once(args.n_samples, args.n_noise, args.seed, drop_conf=False)
    print(f"PEHE full | Plug-in={full[0]:.3f} | DML={full[1]:.3f}")
    print("Training with one confounder dropped...")
    dropped = run_once(args.n_samples, args.n_noise, args.seed, drop_conf=True)
    print(f"PEHE drop | Plug-in={dropped[0]:.3f} | DML={dropped[1]:.3f}")
    plot_results(full, dropped)


if __name__ == "__main__":
    main()
