import argparse
import os
import ctypes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV

# Ensure libomp is discoverable on macOS Homebrew installs before importing xgboost
_libomp_path = "/usr/local/opt/libomp/lib"
if os.path.isdir(_libomp_path):
    os.environ["DYLD_LIBRARY_PATH"] = _libomp_path
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    libomp_so = os.path.join(_libomp_path, "libomp.dylib")
    if os.path.exists(libomp_so):
        try:
            ctypes.cdll.LoadLibrary(libomp_so)
        except OSError:
            pass

# Avoid thread over-subscription / OpenMP conflicts
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from xgboost import XGBRegressor, XGBClassifier

from data_ihdp import load_ihdp_replicate, COLUMNS
from run_ablation import (
    cross_fit_nuisance,
    train_rlearner,
    train_tarnet,
    stabilize_residuals,
    predict_tau_rlearner,
    predict_tau_tarnet,
)


def set_seed(seed: int):
    np.random.seed(seed)


def pehe(tau_hat: np.ndarray, mu0: np.ndarray, mu1: np.ndarray) -> float:
    true_te = mu1 - mu0
    return float(np.sqrt(np.mean((tau_hat - true_te) ** 2)))


def ate_bias(tau_hat: np.ndarray, mu0: np.ndarray, mu1: np.ndarray) -> float:
    true_te = mu1 - mu0
    return float(np.mean(tau_hat) - float(np.mean(true_te)))


def run_xgb_rlearner(df: pd.DataFrame, seed: int, k_folds: int = 3):
    """Cross-fitted XGBoost R-learner used as a strong tabular baseline."""
    from run_ihdp_xgboost import cross_fit_ihdp, stabilize, train_tau_xgb

    X = df[[c for c in COLUMNS if c.startswith("x")]].to_numpy()
    y = df["y_factual"].to_numpy()
    t = df["treatment"].to_numpy()
    X, y, t, m_hat, e_hat = cross_fit_ihdp(df, k_folds=k_folds, seed=seed)
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R = y - m_hat
    W = t - e_hat
    Z, s = stabilize(R, W, clip=0.05)
    tau_model = train_tau_xgb(X, Z, s, seed=seed)
    return tau_model


def run_methods_once(rep: int, seed: int, test_size: float = 0.2):
    df = load_ihdp_replicate(rep)
    feature_cols = [c for c in COLUMNS if c.startswith("x")]

    X = df[feature_cols].to_numpy().astype(np.float32)
    y = df["y_factual"].to_numpy().astype(np.float32)
    t = df["treatment"].to_numpy().astype(np.float32)
    mu0 = df["mu0"].to_numpy().astype(np.float32)
    mu1 = df["mu1"].to_numpy().astype(np.float32)

    # Consistent split per replicate
    set_seed(seed)
    (
        X_tr,
        X_te,
        y_tr,
        y_te,
        t_tr,
        t_te,
        _mu0_tr,
        mu0_te,
        _mu1_tr,
        mu1_te,
    ) = train_test_split(X, y, t, mu0, mu1, test_size=test_size, random_state=seed, stratify=t)

    results = {}

    # 1) TARNet / DragonNet-style plug-in baseline (no gating)
    tarnet = train_tarnet(
        X_tr,
        y_tr,
        t_tr,
        use_gating=False,
        lambda_sparsity=0.0,
        seed=seed,
        dropout_p=0.25,
        hidden_dim=96,
        batch_size=128,
        epochs=250,
        lr=0.003,
    )
    tau_tarnet = predict_tau_tarnet(tarnet, X_te)
    results["tarnet_pehe"] = pehe(tau_tarnet, mu0_te, mu1_te)
    results["tarnet_bias"] = ate_bias(tau_tarnet, mu0_te, mu1_te)

    # 2) CTRL-DML (gating + orthogonal R-learner head)
    plugin_ctrl = train_tarnet(
        X_tr,
        y_tr,
        t_tr,
        use_gating=False,
        lambda_sparsity=0.0,
        seed=seed,
        dropout_p=0.25,
        hidden_dim=96,
        batch_size=128,
        epochs=180,
        lr=0.003,
    )
    tau_plugin = predict_tau_tarnet(plugin_ctrl, X_tr)
    m_hat, e_hat = cross_fit_nuisance(
        X_tr,
        y_tr,
        t_tr,
        use_gating=False,
        lambda_sparsity=0.0,
        seed=seed,
        k_folds=3,
        dropout_p=0.25,
        hidden_dim=96,
        batch_size=128,
        epochs=180,
        clip_prop=0.01,
    )
    e_hat = np.clip(e_hat, 0.01, 0.99)
    R = y_tr - m_hat
    W = t_tr - e_hat
    Z, weights = stabilize_residuals(R, W, clip_w=0.05, z_clip=6.0)
    tau_model = train_rlearner(
        X_tr,
        Z,
        weights,
        use_gating=False,
        lambda_tau=1e-4,
        seed=seed,
        dropout_p=0.25,
        hidden_dim=96,
        batch_size=128,
        epochs=260,
        lr=5e-4,
        grad_clip=1.0,
        warm_start_from=plugin_ctrl,
        teacher_tau=tau_plugin,
        aux_beta_start=0.1,
        aux_beta_end=0.0,
        aux_decay_epochs=40,
    )
    tau_ctrl = predict_tau_rlearner(tau_model, X_te)
    results["ctrl_pehe"] = pehe(tau_ctrl, mu0_te, mu1_te)
    results["ctrl_bias"] = ate_bias(tau_ctrl, mu0_te, mu1_te)

    # 3) Causal Forest (econml)
    est = CausalForestDML(
        model_y=LassoCV(),
        model_t=LogisticRegressionCV(max_iter=1000),
        n_estimators=200,
        discrete_treatment=True,
        random_state=seed,
    )
    est.fit(y_tr, t_tr, X=X_tr)
    tau_cf = est.effect(X_te).flatten()
    results["cf_pehe"] = pehe(tau_cf, mu0_te, mu1_te)
    results["cf_bias"] = ate_bias(tau_cf, mu0_te, mu1_te)

    # 4) XGBoost R-learner
    xgb_model = run_xgb_rlearner(df=pd.DataFrame({"treatment": t_tr, "y_factual": y_tr, **{f"x{i+1}": X_tr[:, i] for i in range(X_tr.shape[1])}}), seed=seed)
    tau_xgb = xgb_model.predict(X_te)
    results["xgb_pehe"] = pehe(tau_xgb, mu0_te, mu1_te)
    results["xgb_bias"] = ate_bias(tau_xgb, mu0_te, mu1_te)

    return results


def aggregate_over_reps(reps, seed: int, test_size: float):
    rows = []
    for rep in reps:
        print(f"\n=== IHDP replicate {rep} ===")
        res = run_methods_once(rep, seed=seed, test_size=test_size)
        res["replicate"] = rep
        rows.append(res)
        for k, v in res.items():
            if k == "replicate":
                continue
            print(f"{k}: {v:.3f}")

    df = pd.DataFrame(rows)
    summary = {}
    robust = {}
    for key in [c for c in df.columns if c != "replicate"]:
        summary[key] = {"mean": df[key].mean(), "std": df[key].std()}
        robust[key] = {"median": df[key].median(), "iqr": df[key].quantile(0.75) - df[key].quantile(0.25)}
    return df, summary, robust


def main():
    parser = argparse.ArgumentParser(description="IHDP benchmark across multiple replicates.")
    parser.add_argument("--reps", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction per replicate.")
    args = parser.parse_args()

    df, summary, robust = aggregate_over_reps(args.reps, seed=args.seed, test_size=args.test_size)
    df.to_csv("ihdp_benchmark_rows.csv", index=False)
    print("\nPer-replicate rows saved to ihdp_benchmark_rows.csv")

    print("\n=== IHDP Benchmark Summary (mean ± std) ===")
    for metric, stats in summary.items():
        print(f"{metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    pd.DataFrame(summary).T.to_csv("ihdp_benchmark_summary.csv")
    print("Summary saved to ihdp_benchmark_summary.csv")

    print("\n=== IHDP Benchmark Robust Stats (median / IQR) ===")
    for metric, stats in robust.items():
        print(f"{metric}: median {stats['median']:.3f}, IQR {stats['iqr']:.3f}")
    pd.DataFrame(robust).T.to_csv("ihdp_benchmark_robust.csv")
    print("Robust summary saved to ihdp_benchmark_robust.csv")


if __name__ == "__main__":
    main()
