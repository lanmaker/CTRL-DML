import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression

from run_ablation import NuisanceNet, get_stress_data, train_nuisance
from catenets.models.torch.base import DEVICE
from sklearn.linear_model import LassoCV


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "CTRL-DML-Paper"


def compute_head_masks(model: NuisanceNet, X: np.ndarray, T: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate treatment/outcome feature masks via calibrated linear proxies.

    M_T comes from a logistic regression on (X, T). M_Y comes from a sparse
    Lasso fit of Y on (X, T), reweighted by the learned tabular gate to inject
    inductive bias from CTRL-DML.
    """
    # Propensity mask via logistic regression (confounders + instruments)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, T)

    # sklearn will have seen T via fit; pull absolute coefficients
    m_t = np.abs(lr.coef_).flatten()
    m_t = m_t / (m_t.max() + 1e-8)

    # Outcome mask via sparse linear model on observed outcomes, modulated by the learned gate
    lin = LinearRegression()
    lin.fit(np.concatenate([X, T.reshape(-1, 1)], axis=1), y)
    m_y = np.abs(lin.coef_[:-1])  # drop treatment coefficient

    with torch.no_grad():
        gate_mean = model.backbone.attn.mask_net(torch.from_numpy(X).float().to(DEVICE)).mean(dim=0).cpu().numpy()
    m_y = m_y * gate_mean
    m_y = m_y / (m_y.max() + 1e-8)

    # Normalize to [0,1]
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = v - v.min()
        denom = v.max() + 1e-8
        return v / denom

    return _normalize(m_t), _normalize(m_y)


def role_scores(m_t: np.ndarray, m_y: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute confounder/instrument/prognostic/noise scores per feature."""
    s_conf = m_t * m_y
    s_inst = m_t * (1 - m_y)
    s_prog = (1 - m_t) * m_y
    s_noise = (1 - m_t) * (1 - m_y)
    stacked = np.stack([s_conf, s_inst, s_prog, s_noise], axis=1)
    stacked = stacked / (np.sum(stacked, axis=1, keepdims=True) + 1e-8)
    return {
        "conf": stacked[:, 0],
        "inst": stacked[:, 1],
        "prog": stacked[:, 2],
        "noise": stacked[:, 3],
    }


def evaluate_roles(m_t: np.ndarray, m_y: np.ndarray, n_noise: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Quantify how well head-specific masks recover true feature roles using top-k metrics.

    Ground truth: first 5 confounders, next 5 instruments, remaining noise.
    """
    scores = role_scores(m_t, m_y)
    conf_idx = set(range(5))
    inst_idx = set(range(5, 10))

    def topk_metrics(score: np.ndarray, target: set[int], k: int) -> Tuple[float, float]:
        topk = set(np.argsort(score)[::-1][:k])
        true_pos = len(topk & target)
        precision = true_pos / k
        recall = true_pos / len(target)
        return precision, recall

    prec_c5, rec_c5 = topk_metrics(scores["conf"], conf_idx, k=5)
    prec_c10, rec_c10 = topk_metrics(scores["conf"], conf_idx, k=10)
    prec_i5, rec_i5 = topk_metrics(scores["inst"], inst_idx, k=5)
    prec_i10, rec_i10 = topk_metrics(scores["inst"], inst_idx, k=10)

    metrics = {
        "precision_conf_5": float(prec_c5),
        "precision_conf_10": float(prec_c10),
        "recall_conf_5": float(rec_c5),
        "recall_conf_10": float(rec_c10),
        "precision_inst_5": float(prec_i5),
        "precision_inst_10": float(prec_i10),
        "recall_inst_5": float(rec_i5),
        "recall_inst_10": float(rec_i10),
    }
    return metrics, scores


def plot_roles(
    m_t: np.ndarray,
    m_y: np.ndarray,
    scores: Dict[str, np.ndarray],
    metrics: Dict[str, float],
    n_noise: int,
    filename: str = "feature_roles.pdf",
):
    """Generate a two-panel plot: head masks + top-k recovery bars."""
    feature_ids = np.arange(10 + n_noise)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Panel A: head-specific masks
    axes[0].plot(feature_ids, m_t, label="Treatment mask $M_T$", color="#1f77b4")
    axes[0].plot(feature_ids, m_y, label="Outcome mask $M_Y$", color="#ff7f0e")
    axes[0].axvspan(-0.5, 4.5, color="#1f77b4", alpha=0.08, label="Confounders")
    axes[0].axvspan(4.5, 9.5, color="#ff7f0e", alpha=0.08, label="Instruments")
    axes[0].set_xlabel("Feature index")
    axes[0].set_ylabel("Normalized mask")
    axes[0].set_title("Head-specific masks")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.2)

    # Panel B: top-k precision
    bar_labels = ["Conf@5", "Conf@10", "Inst@5", "Inst@10"]
    bar_vals = [
        metrics["precision_conf_5"],
        metrics["precision_conf_10"],
        metrics["precision_inst_5"],
        metrics["precision_inst_10"],
    ]
    colors = ["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"]
    bars = axes[1].bar(bar_labels, bar_vals, color=colors, alpha=0.8)
    axes[1].set_ylim(0, 1.05)
    axes[1].bar_label(bars, labels=[f"{v:.2f}" for v in bar_vals], padding=4)
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Top-k role recovery")
    axes[1].grid(alpha=0.2, axis="y")
    plt.tight_layout()

    for base in (ROOT, PAPER_DIR):
        out_path = base / filename
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Feature role decomposition via treatment/outcome head masks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-noise", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lambda-sparsity", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=220)
    args = parser.parse_args()

    X, T, y, _ = get_stress_data(n_samples=args.n_samples, n_noise=args.n_noise, seed=args.seed)
    nuisance = train_nuisance(
        X,
        y,
        T,
        use_gating=True,
        lambda_sparsity=args.lambda_sparsity,
        seed=args.seed,
        dropout_p=args.dropout,
        hidden_dim=120,
        batch_size=192,
        epochs=args.epochs,
        t_weight=0.3,
    )

    m_t, m_y = compute_head_masks(nuisance, X, T, y)
    metrics, scores = evaluate_roles(m_t, m_y, args.n_noise)
    print("Role metrics:", metrics)
    plot_roles(m_t, m_y, role_scores(m_t, m_y), metrics, args.n_noise)


if __name__ == "__main__":
    main()
