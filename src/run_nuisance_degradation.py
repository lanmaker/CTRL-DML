import argparse
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, LogisticRegression

from run_ablation import (
    get_stress_data,
    train_tarnet,
    cross_fit_nuisance,
    train_rlearner,
    predict_tau_tarnet,
    predict_tau_rlearner,
    set_seed,
)


def pehe(tau_hat: np.ndarray, true_te: np.ndarray) -> float:
    return float(np.sqrt(np.mean((tau_hat - true_te) ** 2)))


def ate_bias(tau_hat: np.ndarray, true_te: np.ndarray) -> float:
    return float(np.mean(tau_hat) - np.mean(true_te))


@dataclass
class NuisanceLevel:
    name: str
    kind: str  # "nn", "shallow", "linear", "noisy"
    noise_std: float = 0.0
    epochs: int = 120


def train_linear_nuisance(X: np.ndarray, y: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m_model = LinearRegression()
    e_model = LogisticRegression(max_iter=1000)
    m_model.fit(X, y)
    e_model.fit(X, t)
    m_hat = m_model.predict(X)
    e_hat = e_model.predict_proba(X)[:, 1]
    return m_hat.astype(np.float32), e_hat.astype(np.float32)


def get_nuisances(level: NuisanceLevel, X: np.ndarray, y: np.ndarray, t: np.ndarray, seed: int):
    if level.kind == "linear":
        return train_linear_nuisance(X, y, t)
    if level.kind == "noisy":
        # Start from a strong model then inject Gaussian noise
        m_hat, e_hat = cross_fit_nuisance(
            X,
            y,
            t,
            use_gating=False,
            lambda_sparsity=0.0,
            seed=seed,
            k_folds=3,
            dropout_p=0.25,
            hidden_dim=96,
            batch_size=192,
            epochs=160,
            clip_prop=0.01,
        )
        rng = np.random.default_rng(seed)
        m_hat = m_hat + level.noise_std * rng.standard_normal(size=m_hat.shape)
        e_hat = e_hat + level.noise_std * rng.standard_normal(size=e_hat.shape)
        return m_hat.astype(np.float32), e_hat.astype(np.float32)
    # Neural network nuisances with adjustable epochs (quality)
    return cross_fit_nuisance(
        X,
        y,
        t,
        use_gating=False,
        lambda_sparsity=0.0,
        seed=seed,
        k_folds=3,
        dropout_p=0.25,
        hidden_dim=96,
        batch_size=192,
        epochs=level.epochs,
        clip_prop=0.01,
    )


def run_once(level: NuisanceLevel, seed: int, n_samples: int, n_noise: int):
    set_seed(seed)
    X, T, y, true_te = get_stress_data(n_samples=n_samples, n_noise=n_noise, seed=seed)
    m_hat, e_hat = get_nuisances(level, X, y, T, seed=seed)
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R = y - m_hat
    W = T - e_hat
    # Plug-in TARNet
    plugin = train_tarnet(
        X,
        y,
        T,
        use_gating=False,
        lambda_sparsity=0.0,
        seed=seed,
        dropout_p=0.25,
        hidden_dim=96,
        batch_size=192,
        epochs=200,
        lr=0.003,
    )
    tau_plugin = predict_tau_tarnet(plugin, X)
    # DML tau using fixed nuisances
    W_clip = np.clip(W, -0.2, 0.2)
    Z = np.clip(R / W_clip, -10.0, 10.0)
    weights = np.minimum(W_clip**2, 0.04)
    tau_model = train_rlearner(
        X,
        Z,
        weights,
        use_gating=False,
        lambda_tau=5e-4,
        seed=seed,
        dropout_p=0.25,
        hidden_dim=96,
        batch_size=192,
        epochs=260,
        lr=5e-4,
        grad_clip=1.0,
        warm_start_from=None,
        teacher_tau=None,
    )
    tau_dml = predict_tau_rlearner(tau_model, X)
    # Nuisance quality metrics
    mse_m = float(np.mean((y - m_hat) ** 2))
    logloss_t = float(np.mean(-(T * np.log(e_hat + 1e-8) + (1 - T) * np.log(1 - e_hat + 1e-8))))
    return {
        "seed": seed,
        "level": level.name,
        "crossfit": True,
        "m_mse": mse_m,
        "t_logloss": logloss_t,
        "plugin_pehe": pehe(tau_plugin, true_te),
        "plugin_ate_bias": ate_bias(tau_plugin, true_te),
        "dml_pehe": pehe(tau_dml, true_te),
        "dml_ate_bias": ate_bias(tau_dml, true_te),
    }


def run_no_crossfit(level: NuisanceLevel, seed: int, n_samples: int, n_noise: int):
    """Train nuisances and tau on the same sample (no cross-fitting) as a control."""
    set_seed(seed)
    X, T, y, true_te = get_stress_data(n_samples=n_samples, n_noise=n_noise, seed=seed)
    # Single nuisance model on full data
    from run_ablation import train_nuisance

    model = train_nuisance(
        X,
        y,
        T,
        use_gating=False,
        lambda_sparsity=0.0,
        seed=seed,
        dropout_p=0.25,
        hidden_dim=96,
        batch_size=192,
        epochs=level.epochs if level.kind == "nn" else 60,
    )
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        xt = torch.from_numpy(X).float().to(device)
        y0, y1, t_prob = model(xt)
        m_hat = (
            t_prob.squeeze().cpu().numpy() * y1.squeeze().cpu().numpy()
            + (1 - t_prob.squeeze().cpu().numpy()) * y0.squeeze().cpu().numpy()
        )
        e_hat = t_prob.squeeze().cpu().numpy()
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R = y - m_hat
    W = T - e_hat
    W_clip = np.clip(W, -0.2, 0.2)
    Z = np.clip(R / W_clip, -10.0, 10.0)
    weights = np.minimum(W_clip**2, 0.04)
    tau_model = train_rlearner(
        X,
        Z,
        weights,
        use_gating=False,
        lambda_tau=5e-4,
        seed=seed,
        dropout_p=0.25,
        hidden_dim=96,
        batch_size=192,
        epochs=260,
        lr=5e-4,
        grad_clip=1.0,
        warm_start_from=None,
        teacher_tau=None,
    )
    tau_hat = predict_tau_rlearner(tau_model, X)
    mse_m = float(np.mean((y - m_hat) ** 2))
    logloss_t = float(np.mean(-(T * np.log(e_hat + 1e-8) + (1 - T) * np.log(1 - e_hat + 1e-8))))
    return {
        "seed": seed,
        "level": f"{level.name}_nocf",
        "crossfit": False,
        "m_mse": mse_m,
        "t_logloss": logloss_t,
        "plugin_pehe": np.nan,
        "plugin_ate_bias": np.nan,
        "dml_pehe": pehe(tau_hat, true_te),
        "dml_ate_bias": ate_bias(tau_hat, true_te),
    }


def plot_curves(rows: List[dict], out_path: str):
    import pandas as pd

    df = pd.DataFrame(rows)
    grouped = df.groupby("level").agg(
        m_mse=("m_mse", "mean"),
        t_logloss=("t_logloss", "mean"),
        plugin_pehe=("plugin_pehe", "mean"),
        plugin_bias=("plugin_ate_bias", "mean"),
        dml_pehe=("dml_pehe", "mean"),
        dml_bias=("dml_ate_bias", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    levels = grouped.index.tolist()
    x = np.arange(len(levels))
    width = 0.35
    axes[0].bar(x - width / 2, grouped["plugin_pehe"], width, label="Plug-in")
    axes[0].bar(x + width / 2, grouped["dml_pehe"], width, label="DML")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(levels, rotation=15)
    axes[0].set_ylabel("PEHE")
    axes[0].set_title("Error vs nuisance quality")
    axes[0].legend()

    axes[1].bar(x - width / 2, grouped["plugin_bias"], width, label="Plug-in")
    axes[1].bar(x + width / 2, grouped["dml_bias"], width, label="DML")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(levels, rotation=15)
    axes[1].set_ylabel("ATE bias")
    axes[1].axhline(0, color="k", linewidth=0.8)
    axes[1].set_title("Bias vs nuisance quality")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Nuisance degradation study contrasting plug-in vs DML.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1024, 2023])
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-noise", type=int, default=50)
    parser.add_argument("--output-csv", type=str, default="nuisance_degradation.csv")
    parser.add_argument("--output-fig", type=str, default="nuisance_degradation.pdf")
    args = parser.parse_args()

    levels = [
        NuisanceLevel(name="strong", kind="nn", epochs=180),
        NuisanceLevel(name="shallow", kind="nn", epochs=60),
        NuisanceLevel(name="linear", kind="linear"),
        NuisanceLevel(name="noisy", kind="noisy", noise_std=0.15),
    ]
    rows = []
    for level in levels:
        for seed in args.seeds:
            print(f"=== Level {level.name}, seed {seed} ===")
            rows.append(run_once(level, seed, args.n_samples, args.n_noise))
            # Also record a no-cross-fitting control for the strongest nuisances
            if level.name == "strong":
                rows.append(run_no_crossfit(level, seed, args.n_samples, args.n_noise))
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")
    plot_curves(rows, args.output_fig)


if __name__ == "__main__":
    main()
