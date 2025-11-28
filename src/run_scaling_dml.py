import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from ctrl_orthogonal_learner import CTRLOrthogonalLearner, CTRLConfig
from run_ablation import get_stress_data, evaluate_pehe, predict_tau_tarnet


def run_scale(sample_sizes, n_noise, seeds):
    rows = []
    for n in sample_sizes:
        for seed in seeds:
            X, T, y, true_te = get_stress_data(n_samples=n, n_noise=n_noise, seed=seed)
            # Causal Forest baseline
            cf = CausalForestDML(
                model_y=LassoCV(),
                model_t=LogisticRegressionCV(max_iter=1000),
                n_estimators=80,
                discrete_treatment=True,
                random_state=seed,
            )
            cf.fit(y, T, X=X)
            tau_cf = cf.effect(X).flatten()
            rows.append({"dataset": f"N{n}", "method": "Causal Forest", "seed": seed, "pehe": evaluate_pehe(tau_cf, true_te)})

            # CTRL-DML (plugin + orthogonal)
            cfg = CTRLConfig(
                use_gating=True,
                lambda_sparsity=0.05,
                hidden_dim=96,
                hidden_tau=96,
                dropout_p=0.4,
                batch_size=192,
                plugin_epochs=120,
                nuisance_epochs=90,
                tau_epochs=180,
                lr_plugin=0.003,
                lr_tau=3e-4,
                k_folds=3,
                lambda_tau=5e-5,
                grad_clip=1.0,
                z_clip=5.0,
                w_clip=0.05,
                aux_beta_start=0.6,
                aux_beta_end=0.2,
                aux_decay_epochs=140,
                freeze_backbone=True,
            )
            learner = CTRLOrthogonalLearner(cfg)
            learner.fit(X, y, T, seed=seed)
            tau_dml = learner.predict_tau(X)
            tau_plugin = learner.predict_plugin(X)
            rows.append({"dataset": f"N{n}", "method": "CTRL-DML (orthogonal)", "seed": seed, "pehe": evaluate_pehe(tau_dml, true_te)})
            rows.append({"dataset": f"N{n}", "method": "CTRL plug-in", "seed": seed, "pehe": evaluate_pehe(tau_plugin, true_te)})
            print(f"N={n} seed={seed} | CF {rows[-3]['pehe']:.3f} | plugin {rows[-1]['pehe']:.3f} | DML {rows[-2]['pehe']:.3f}")
    return pd.DataFrame(rows)


def plot_scaling(df: pd.DataFrame, out_path: Path):
    stats = df.groupby(["dataset", "method"]).pehe.agg(["mean", "std"]).reset_index()
    order = sorted(df["dataset"].unique(), key=lambda x: int(x[1:]))
    methods = ["Causal Forest", "CTRL plug-in", "CTRL-DML (orthogonal)"]
    x = np.arange(len(order))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Causal Forest": "#1f77b4", "CTRL plug-in": "#2ca02c", "CTRL-DML (orthogonal)": "#ff7f0e"}
    offsets = [-width, 0, width]
    for idx, method in enumerate(methods):
        means, stds = [], []
        for m in order:
            row = stats[(stats.dataset == m) & (stats.method == method)]
            means.append(float(row["mean"]))
            stds.append(float(row["std"]))
        ax.bar(x + offsets[idx], means, width, yerr=stds, label=method, color=colors[method], alpha=0.9, capsize=4)
        for i, v in enumerate(means):
            ax.text(x[i] + offsets[idx], v - 0.03, f"{v:.2f}", ha="center", va="top", color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("PEHE (lower is better)")
    ax.set_title("Scaling with noise=50 (PEHE by N)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Scaling study: CF vs CTRL plug-in vs CTRL-DML.")
    parser.add_argument("--n-noise", type=int, default=50)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    parser.add_argument("--csv", type=str, default="scaling_dml.csv")
    parser.add_argument("--plot", type=str, default="scaling_dml.pdf")
    args = parser.parse_args()

    df = run_scale(args.sample_sizes, args.n_noise, args.seeds)
    df.to_csv(args.csv, index=False)
    print(f"Wrote {args.csv}")
    plot_scaling(df, Path(args.plot))


if __name__ == "__main__":
    main()
