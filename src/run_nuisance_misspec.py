import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple

from run_ablation import (
    get_stress_data,
    train_plugin,
    cross_fit_nuisance,
    stabilize_residuals,
    train_rlearner,
    predict_tau_tarnet,
    predict_tau_rlearner,
    evaluate_pehe,
)


def run_once(n_samples: int, n_noise: int, seed: int, weaken: bool) -> Tuple[float, float]:
    X, T, y, true_te = get_stress_data(n_samples=n_samples, n_noise=n_noise, seed=seed)
    # Stage 0: plug-in warm start
    plugin = train_plugin(
        X,
        y,
        T,
        use_gating=True,
        lambda_sparsity=0.05,
        seed=seed,
        dropout_p=0.4,
        hidden_dim=96 if weaken else 120,
        batch_size=192,
        epochs=80 if weaken else 180,
        lr=0.003,
    )
    tau_plugin = predict_tau_tarnet(plugin, X)

    # Nuisance cross-fit (weakened means fewer epochs and smaller nets)
    m_hat, e_hat = cross_fit_nuisance(
        X,
        y,
        T,
        use_gating=True,
        lambda_sparsity=0.05,
        seed=seed,
        k_folds=3,
        dropout_p=0.4,
        hidden_dim=72 if weaken else 120,
        batch_size=192,
        epochs=60 if weaken else 150,
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
        hidden_dim=96 if weaken else 120,
        batch_size=192,
        epochs=150 if weaken else 300,
        lr=3e-4,
        grad_clip=1.0,
        warm_start_from=plugin,
        teacher_tau=tau_plugin,
        aux_beta_start=0.6,
        aux_beta_end=0.2,
        aux_decay_epochs=120 if weaken else 200,
        freeze_backbone=True,
    )
    tau_dml = predict_tau_rlearner(tau_model, X)

    return evaluate_pehe(tau_plugin, true_te), evaluate_pehe(tau_dml, true_te)


def main():
    parser = argparse.ArgumentParser(description="Nuisance misspecification stress: plug-in vs DML.")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-noise", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    parser.add_argument("--out", type=str, default="nuisance_misspec.csv")
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        for weaken in [False, True]:
            pehe_plugin, pehe_dml = run_once(args.n_samples, args.n_noise, seed, weaken)
            rows.append(
                {
                    "seed": seed,
                    "n_samples": args.n_samples,
                    "n_noise": args.n_noise,
                    "nuisance_strength": "strong" if not weaken else "weak",
                    "pehe_plugin": pehe_plugin,
                    "pehe_dml": pehe_dml,
                }
            )
            print(f"Seed {seed} | nuisance={ 'weak' if weaken else 'strong' } | plugin {pehe_plugin:.3f} | dml {pehe_dml:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
