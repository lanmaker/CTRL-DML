import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from my_dragonnet import MyDragonNet as DragonNet


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data(n: int = 2000, n_noise: int = 50, seed: int = 42):
    """Stress-test DGP with confounders, instruments, and noise."""
    rng = np.random.default_rng(seed)
    C = np.random.normal(0, 1, size=(n, 5))
    I = rng.normal(0, 1, size=(n, 5))
    N = rng.normal(0, 1, size=(n, n_noise))
    X = np.concatenate([C, I, N], axis=1)

    true_te = 2 * np.sin(C[:, 0] * np.pi) + np.maximum(0, C[:, 1])
    logit = np.sum(C[:, :2], axis=1) + np.sum(I, axis=1)
    prop = 1 / (1 + np.exp(-logit))
    T = np.random.binomial(1, prop)
    y = np.sum(C, axis=1) + true_te * T + np.random.normal(0, 0.5, n)
    return X, y, T, true_te


def conformalize(pred_mean: np.ndarray, true_te: np.ndarray, alpha: float = 0.05) -> float:
    """Return the (1-alpha) residual quantile for symmetric conformal intervals."""
    residuals = np.abs(pred_mean - true_te)
    q = np.quantile(residuals, 1 - alpha)
    return float(q)


def plot_intervals(
    te_true,
    mc_mean,
    mc_lower,
    mc_upper,
    conf_lower,
    conf_upper,
    mc_cover: float,
    conf_cover: float,
    filename: str,
    ensemble_mean=None,
    ensemble_lower=None,
    ensemble_upper=None,
    ensemble_cover=None,
):
    """Two-panel plot: intervals and coverage bars."""
    sort_idx = np.argsort(te_true)
    te_sorted = te_true[sort_idx]
    mc_mean_sorted = mc_mean[sort_idx]
    mc_lower_sorted = mc_lower[sort_idx]
    mc_upper_sorted = mc_upper[sort_idx]
    conf_lower_sorted = conf_lower[sort_idx]
    conf_upper_sorted = conf_upper[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    limit = min(140, len(te_sorted))
    xs = np.arange(limit)

    axes[0].plot(xs, te_sorted[:limit], color="black", linestyle="--", linewidth=2.0, label="True CATE")
    axes[0].plot(xs, mc_mean_sorted[:limit], color="#2ca02c", linewidth=1.5, label="MC Dropout mean")
    axes[0].fill_between(xs, mc_lower_sorted[:limit], mc_upper_sorted[:limit], color="#2ca02c", alpha=0.25, label="MC Dropout 95%")
    axes[0].fill_between(xs, conf_lower_sorted[:limit], conf_upper_sorted[:limit], color="#ff7f0e", alpha=0.18, label="Conformal 95%")
    if ensemble_mean is not None:
        ens_mean_sorted = ensemble_mean[sort_idx][:limit]
        ens_low_sorted = ensemble_lower[sort_idx][:limit]
        ens_up_sorted = ensemble_upper[sort_idx][:limit]
        axes[0].plot(xs, ens_mean_sorted, color="#9467bd", linewidth=1.5, label="Ensemble mean")
        axes[0].fill_between(xs, ens_low_sorted, ens_up_sorted, color="#9467bd", alpha=0.15, label="Ensemble 95%")
    axes[0].set_xlabel("Sorted samples")
    axes[0].set_ylabel("CATE")
    axes[0].set_title("Intervals (MC Dropout vs Conformal)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(alpha=0.2)

    mc_width = np.mean(mc_upper - mc_lower)
    conf_width = np.mean(conf_upper - conf_lower)
    labels = ["MC Dropout\ncoverage", "Conformal\ncoverage"]
    vals = [mc_cover, conf_cover]
    colors = ["#2ca02c", "#ff7f0e"]
    if ensemble_cover is not None:
        labels.append("Ensemble\ncoverage")
        vals.append(ensemble_cover)
        colors.append("#9467bd")
    axes[1].bar(labels, vals, color=colors)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Coverage")
    axes[1].set_title(f"Avg widths | MC={mc_width:.2f}, Conf={conf_width:.2f}")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved {filename}")
    return mc_width, conf_width


def main():
    parser = argparse.ArgumentParser(description="Uncertainty analysis: MC Dropout vs conformalized intervals.")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--n-noise", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-ensemble", type=int, default=1, help="Number of independently initialized CTRL-DML models.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-csv", type=str, default="uq_metrics.csv")
    args = parser.parse_args()

    X, y, T, true_te = get_data(n=args.n, n_noise=args.n_noise, seed=args.seed)
    X_train, X_test, y_train, y_test, T_train, T_test, te_train, te_test = train_test_split(
        X, y, T, true_te, test_size=0.2, random_state=42
    )

    print("Training CTRL-DML with MC Dropout for UQ...")
    ensemble_means = []
    ensemble_vars = []
    for m_idx in range(args.n_ensemble):
        set_seed(args.seed + m_idx)
        model = DragonNet(
            n_unit_in=X.shape[1],
            n_units_r=800,
            dropout_prob=args.dropout,
            n_iter=220,
            batch_size=64,
        )
        model.fit(X_train, y_train, T_train)
        mc_mean, mc_std, mc_lower, mc_upper = model.predict_with_uncertainty(X_test, n_runs=args.runs)
        ensemble_means.append(mc_mean)
        ensemble_vars.append(mc_std ** 2)

    mc_mean = np.mean(ensemble_means, axis=0)
    within_var = np.mean(ensemble_vars, axis=0)
    between_var = np.var(ensemble_means, axis=0)
    total_std = np.sqrt(within_var + between_var)
    mc_lower = mc_mean - 1.96 * total_std
    mc_upper = mc_mean + 1.96 * total_std

    # Conformalized intervals using a calibration split from the test fold
    cal_size = max(1, min(len(te_test) // 4, len(te_test) - 1))
    cal_idx = slice(0, cal_size)
    eval_idx = slice(cal_size, len(te_test))
    te_cal, te_eval = te_test[cal_idx], te_test[eval_idx]
    mc_mean_cal = mc_mean[cal_idx]
    mc_mean_eval = mc_mean[eval_idx]
    mc_lower_eval = mc_lower[eval_idx]
    mc_upper_eval = mc_upper[eval_idx]
    q = conformalize(mc_mean_cal, te_cal, alpha=args.alpha)
    conf_lower = mc_mean - q
    conf_upper = mc_mean + q
    conf_lower_eval = conf_lower[eval_idx]
    conf_upper_eval = conf_upper[eval_idx]

    # Coverage metrics on the held-out evaluation split
    if len(te_eval) == 0:
        mc_cover = float("nan")
        conf_cover = float("nan")
    else:
        mc_cover = float(np.mean((te_eval >= mc_lower_eval) & (te_eval <= mc_upper_eval)))
        conf_cover = float(np.mean((te_eval >= conf_lower_eval) & (te_eval <= conf_upper_eval)))
    ensemble_cover = mc_cover if args.n_ensemble > 1 else None
    print(
        f"Coverage | MC Dropout (ensemble={args.n_ensemble}): {mc_cover:.3f} | "
        f"Conformal: {conf_cover:.3f} (target {1-args.alpha:.2f})"
    )

    mc_width, conf_width = plot_intervals(
        te_eval,
        mc_mean_eval,
        mc_lower_eval,
        mc_upper_eval,
        conf_lower_eval,
        conf_upper_eval,
        mc_cover,
        conf_cover,
        filename="uq_conformal.pdf",
        ensemble_mean=mc_mean,
        ensemble_lower=mc_lower,
        ensemble_upper=mc_upper,
        ensemble_cover=ensemble_cover,
    )
    # Save summary metrics for the paper.
    import pandas as pd

    metrics = {
        "n": args.n,
        "n_noise": args.n_noise,
        "runs": args.runs,
        "dropout": args.dropout,
        "alpha": args.alpha,
        "n_ensemble": args.n_ensemble,
        "mc_coverage": mc_cover,
        "conf_coverage": conf_cover,
        "mc_width": mc_width,
        "conf_width": conf_width,
        "cal_size": cal_size,
        "eval_size": len(te_eval),
    }
    out = pd.DataFrame([metrics])
    out.to_csv(args.metrics_csv, index=False)
    print(f"Wrote metrics to {args.metrics_csv}")


if __name__ == "__main__":
    main()
