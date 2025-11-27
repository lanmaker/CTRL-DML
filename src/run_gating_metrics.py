import numpy as np
import torch
from sklearn.metrics import roc_auc_score

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


def compute_mask_metrics(seed: int = 42, n_samples: int = 2000, n_noise: int = 50) -> dict:
    """Train a gated plug-in model and report global mask metrics."""
    X, T, y, _ = get_stress_data(n_samples=n_samples, n_noise=n_noise, seed=seed)
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
        epochs=220,
        lr=0.003,
    )
    plugin.eval()
    with torch.no_grad():
        masks = plugin.backbone.attn.mask_net(
            torch.from_numpy(X).float().to(plugin.backbone.attn.mask_net[0].weight.device)
        )
        masks_np = masks.cpu().numpy()
    mean_mask = masks_np.mean(axis=0)

    conf_idx = list(range(5))
    inst_idx = list(range(5, 10))
    noise_idx = list(range(10, 10 + n_noise))

    conf_vals = mean_mask[conf_idx]
    noise_vals = mean_mask[noise_idx]
    auc_conf_noise = roc_auc_score(
        np.concatenate([np.ones_like(conf_vals), np.zeros_like(noise_vals)]),
        np.concatenate([conf_vals, noise_vals]),
    )

    metrics = {"auc_conf_vs_noise": float(auc_conf_noise)}
    for k in (5, 10):
        topk = np.argsort(mean_mask)[::-1][:k]
        conf_in_topk = np.sum(topk < 5)
        precision = conf_in_topk / k
        recall = conf_in_topk / 5.0
        metrics[f"conf_in_top{k}"] = int(conf_in_topk)
        metrics[f"precision@{k}"] = float(precision)
        metrics[f"recall@{k}"] = float(recall)

    metrics.update(
        {
            "mean_conf": float(conf_vals.mean()),
            "mean_inst": float(mean_mask[inst_idx].mean()),
            "mean_noise": float(noise_vals.mean()),
        }
    )
    return metrics


def evaluate_nuisance_variant(
    use_gating: bool,
    lambda_sparsity: float,
    seed: int = 42,
    n_samples: int = 2000,
    n_noise: int = 50,
) -> float:
    """Train nuisances + DML head for a single variant and return PEHE."""
    X, T, y, true_te = get_stress_data(n_samples=n_samples, n_noise=n_noise, seed=seed)
    plugin = train_plugin(
        X,
        y,
        T,
        use_gating=use_gating,
        lambda_sparsity=lambda_sparsity,
        seed=seed,
        dropout_p=0.4,
        hidden_dim=96,
        batch_size=192,
        epochs=220,
        lr=0.003,
    )
    tau_plugin = predict_tau_tarnet(plugin, X)

    m_hat, e_hat = cross_fit_nuisance(
        X,
        y,
        T,
        use_gating=use_gating,
        lambda_sparsity=lambda_sparsity,
        seed=seed,
        k_folds=3,
        dropout_p=0.4,
        hidden_dim=120,
        batch_size=192,
        epochs=150,
    )
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R, W = y - m_hat, T - e_hat
    Z, weights = stabilize_residuals(R, W, clip_w=0.05, z_clip=2.5)

    tau_model = train_rlearner(
        X,
        Z,
        weights,
        use_gating=use_gating,
        lambda_tau=5e-5,
        seed=seed,
        dropout_p=0.4,
        hidden_dim=96,
        batch_size=192,
        epochs=320,
        lr=3e-4,
        grad_clip=1.0,
        warm_start_from=plugin,
        teacher_tau=tau_plugin,
        aux_beta_start=0.8,
        aux_beta_end=0.3,
        aux_decay_epochs=320,
    )
    tau_pred = predict_tau_rlearner(tau_model, X)
    return evaluate_pehe(tau_pred, true_te)


if __name__ == "__main__":
    mask_metrics = compute_mask_metrics()
    print("Mask metrics:", mask_metrics)

    pehe_no_gating = evaluate_nuisance_variant(False, 0.0)
    pehe_gating_l1 = evaluate_nuisance_variant(True, 0.05)
    print(f"Nuisance DML PEHE | no gating: {pehe_no_gating:.3f} | gating+L1: {pehe_gating_l1:.3f}")
