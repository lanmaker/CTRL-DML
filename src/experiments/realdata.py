"""
CTRL-DML Real Data Experiments.

Runs CTRL-DML on real-world observational datasets with OOF ATE estimates and refit bootstrap CIs:
- LaLonde: NSW job training program
- STAR: Tennessee class size experiment

Usage:
    python -m src.experiments.realdata --dataset lalonde
    python -m src.experiments.realdata --dataset star
    python -m src.experiments.realdata --dataset all

Data Sources:
    - LaLonde: econml.datasets, causalml, or direct CSV download (fallback)
    - STAR: Harvard Dataverse (doi:10.7910/DVN/SIWH9F)

Note: Amazon multimodal experiments are in src/experiments/multimodal.py
"""
import argparse
import os
import hashlib
from typing import List, Optional, Tuple, Dict, Any, Callable
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import StratifiedKFold

from src.models.orthogonal_learner import (
    train_plugin,
    cross_fit_nuisance,
    stabilize_residuals,
    train_rlearner,
    predict_tau_tarnet,
    predict_tau_rlearner,
    set_seed,
    CTRLConfig,
    resolve_gate_flags,
)
from src.utils.io import get_output_manager


FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"
LALONDE_SHA256 = "a49703ca0233ba0a27fa9cda959a38a07f6825def0f7743cf2380ce4e57dd3ef"
STAR_SHA256 = "769be163ed54515858efa60b1a069c49ca0c475f0b0f9f5bdf90413be9d3ba97"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(path: Path, expected: str, label: str) -> None:
    if not expected:
        return
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} checksum mismatch: expected {expected}, got {actual}")


def load_lalonde() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load LaLonde NSW dataset.

    Returns (X, T, Y) where:
    - X: Covariates (age, education, earnings, etc.)
    - T: Treatment (1=job training, 0=control)
    - Y: Outcome (re78 - earnings in 1978)
    """
    try:
        from econml.datasets import get_lalonde
        X, T, Y, _, _, _, _ = get_lalonde()
        Y = (Y.astype(np.float32) / 1000.0)
        return X.astype(np.float32), T.astype(np.float32), Y
    except ImportError:
        print("econml not installed. Using causalml fallback...")
        try:
            from causalml.dataset import load_lalonde
            df = load_lalonde()
            T = df["treat"].values.astype(np.float32)
            Y = df["re78"].values.astype(np.float32) / 1000.0
            X = df.drop(columns=["treat", "re78"]).values.astype(np.float32)
            return X, T, Y
        except ImportError:
            print("causalml not installed. Downloading Lalonde CSV...")
            data_root = Path(__file__).parents[2] / "data" / "lalonde"
            data_path = data_root / "lalonde.csv"
            if not data_path.exists():
                data_root.mkdir(parents=True, exist_ok=True)
                data_path = _download_lalonde(data_root)
            else:
                _verify_checksum(data_path, LALONDE_SHA256, "Lalonde CSV")

            df = pd.read_csv(data_path)
            if "rownames" in df.columns:
                df = df.drop(columns=["rownames"])
            treat_col = next((c for c in ["treat", "treatment", "T"] if c in df.columns), None)
            outcome_col = next((c for c in ["re78", "outcome", "Y"] if c in df.columns), None)
            if treat_col is None or outcome_col is None:
                raise KeyError("Lalonde CSV missing treatment or outcome columns.")

            T = df[treat_col].values.astype(np.float32)
            Y = df[outcome_col].values.astype(np.float32) / 1000.0
            X_df = df.drop(columns=[treat_col, outcome_col])
            X = pd.get_dummies(X_df, drop_first=True).values.astype(np.float32)

            mask = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
            return X[mask], T[mask], Y[mask]


def load_star() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load STAR class size dataset.

    Returns (X, T, Y) where:
    - X: Student/school covariates
    - T: Treatment (1=small class, 0=regular class)
    - Y: Outcome (test scores)

    Note: Requires downloading data from Harvard Dataverse.
    """
    data_root = Path(__file__).parents[2] / "data" / "star"
    data_path = data_root / "STAR_Students.tab"

    if not data_path.exists():
        data_root.mkdir(parents=True, exist_ok=True)
        data_path = _download_star_students(data_root)
    else:
        _verify_checksum(data_path, STAR_SHA256, "STAR_Students.tab")

    df = pd.read_csv(data_path, sep="\t")

    # Small class (treatment) vs regular (control).
    # STAR uses class-type codes (1=small, 2=regular, 3=regular+aide).
    class_candidates = ["gkclasstype", "g1classtype", "g2classtype", "g3classtype"]
    score_candidates = ["gktreadss", "g1treadss", "g2treadss", "g3treadss"]

    class_col = next((c for c in class_candidates if c in df.columns), None)
    score_col = next((c for c in score_candidates if c in df.columns), None)
    if class_col is None or score_col is None:
        raise KeyError("STAR columns missing class type or reading score.")

    df = df[df[class_col].isin([1, 2])].copy()
    T = (df[class_col] == 1).astype(np.float32).values
    Y = df[score_col].astype(np.float32).values

    # Covariates: gender, race, free lunch, school urbanicity (grade-k preferred).
    cov_cols = ["gender", "race", "gkfreelunch", "gksurban"]
    cov_cols = [c for c in cov_cols if c in df.columns]
    cov_df = df[cov_cols].copy()
    for col in cov_df.columns:
        cov_df[col] = cov_df[col].fillna(-1).astype("category")
    X = pd.get_dummies(cov_df, drop_first=True).values.astype(np.float32)

    # Handle missing outcomes.
    mask = ~np.isnan(Y)
    return X[mask], T[mask], Y[mask]


def _download_star_students(data_root: Path) -> Path:
    """
    Download STAR_Students.tab from Harvard Dataverse if missing.

    Returns path to the downloaded file.
    """
    meta_url = (
        "https://dataverse.harvard.edu/api/datasets/:persistentId/"
        "?persistentId=doi:10.7910/DVN/SIWH9F"
    )
    resp = requests.get(meta_url, timeout=30)
    resp.raise_for_status()
    files = resp.json().get("data", {}).get("latestVersion", {}).get("files", [])

    file_id = None
    filename = None
    for item in files:
        data_file = item.get("dataFile", {})
        name = data_file.get("filename", "")
        if name.lower().startswith("star_students"):
            file_id = data_file.get("id")
            filename = name
            break

    if file_id is None:
        raise RuntimeError("STAR_Students file not found in Dataverse metadata.")

    download_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
    resp = requests.get(download_url, timeout=60)
    resp.raise_for_status()

    out_path = data_root / filename
    out_path.write_bytes(resp.content)
    _verify_checksum(out_path, STAR_SHA256, "STAR_Students.tab")
    return out_path


def _download_lalonde(data_root: Path) -> Path:
    """
    Download Lalonde dataset CSV if missing.

    Returns path to the downloaded file.
    """
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/MatchIt/lalonde.csv"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path = data_root / "lalonde.csv"
    out_path.write_bytes(resp.content)
    _verify_checksum(out_path, LALONDE_SHA256, "Lalonde CSV")
    return out_path


def compute_smd(X: np.ndarray, T: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute standardized mean differences for covariate balance.

    Args:
        X: Covariates (n, d)
        T: Treatment indicator (n,)
        weights: Optional sample weights

    Returns:
        Dict with 'mean_smd' and 'n_imbalanced' (SMD > 0.1)
    """
    t1 = T == 1
    t0 = T == 0

    if weights is None:
        weights = np.ones(len(T))

    smds = []
    for j in range(X.shape[1]):
        x = X[:, j]
        if weights is not None:
            w1 = weights[t1]
            w0 = weights[t0]
            mean1 = np.average(x[t1], weights=w1)
            mean0 = np.average(x[t0], weights=w0)
        else:
            mean1 = x[t1].mean()
            mean0 = x[t0].mean()

        pooled_std = np.sqrt((x[t1].var() + x[t0].var()) / 2 + 1e-8)
        smd = abs(mean1 - mean0) / pooled_std
        smds.append(smd)

    smds = np.array(smds)
    return {
        "max_smd": float(smds.max()),
        "n_imbalanced": int((smds > 0.1).sum()),
    }


def _resolve_k_folds(T: np.ndarray, k_folds: int) -> int:
    counts = [int(np.sum(T == 0)), int(np.sum(T == 1))]
    min_class = min(counts)
    if min_class < 2:
        raise ValueError("Need at least two samples per treatment group for cross-fitting.")
    return max(2, min(k_folds, min_class))


def _stratified_bootstrap_indices(T: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx_t = np.where(T == 1)[0]
    idx_c = np.where(T == 0)[0]
    if len(idx_t) == 0 or len(idx_c) == 0:
        return rng.choice(len(T), size=len(T), replace=True)
    boot_t = rng.choice(idx_t, size=len(idx_t), replace=True)
    boot_c = rng.choice(idx_c, size=len(idx_c), replace=True)
    idx = np.concatenate([boot_t, boot_c])
    return rng.permutation(idx)


def bootstrap_refit_ate(
    estimate_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, int], float],
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = estimate_fn(X, T, Y, seed)
    estimates = []
    for b in range(n_bootstrap):
        idx = _stratified_bootstrap_indices(T, rng)
        estimates.append(estimate_fn(X[idx], T[idx], Y[idx], seed + b + 1))
    estimates = np.array(estimates)
    ci_lower = np.percentile(estimates, 100 * alpha / 2)
    ci_upper = np.percentile(estimates, 100 * (1 - alpha / 2))
    return point, float(ci_lower), float(ci_upper)


def _estimate_ols_ate(X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int) -> float:
    from sklearn.linear_model import LinearRegression
    X_with_t = np.column_stack([X, T])
    ols = LinearRegression().fit(X_with_t, Y)
    return float(ols.coef_[-1])


def _estimate_tarnet_oof_ate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    ctrl_config: CTRLConfig,
    seed: int,
    k_folds: int,
) -> float:
    """Estimate ATE using OOF TARNet plug-in predictions (no gating for fair baseline)."""
    k_folds = _resolve_k_folds(T, k_folds)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    tau_oof = np.zeros(len(Y), dtype=np.float32)

    for fold, (tr, val) in enumerate(skf.split(X, T)):
        fold_seed = seed + 100 * (fold + 1)
        plugin_model = train_plugin(
            X[tr], Y[tr], T[tr],
            use_gating=False,  # TARNet baseline: no gating
            lambda_sparsity=0.0,  # No sparsity for vanilla TARNet
            seed=fold_seed,
            dropout_p=ctrl_config.dropout_p,
            hidden_dim=ctrl_config.hidden_tau,
            batch_size=ctrl_config.batch_size,
            epochs=ctrl_config.plugin_epochs,
        )
        tau_oof[val] = predict_tau_tarnet(plugin_model, X[val])

    return float(np.mean(tau_oof))


def _estimate_ctrl_dml_oof_ate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    ctrl_config: CTRLConfig,
    seed: int,
    k_folds: int,
) -> float:
    k_folds = _resolve_k_folds(T, k_folds)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    tau_oof = np.zeros(len(Y), dtype=np.float32)
    plugin_gating, nuisance_gating, tau_gating = resolve_gate_flags(ctrl_config)
    warm_start_backbone = plugin_gating == tau_gating

    for fold, (tr, val) in enumerate(skf.split(X, T)):
        fold_seed = seed + 200 * (fold + 1)
        plugin_model = train_plugin(
            X[tr], Y[tr], T[tr],
            use_gating=plugin_gating,
            lambda_sparsity=ctrl_config.lambda_sparsity,
            seed=fold_seed,
            dropout_p=ctrl_config.dropout_p,
            hidden_dim=ctrl_config.hidden_tau,
            batch_size=ctrl_config.batch_size,
            epochs=ctrl_config.plugin_epochs,
        )
        tau_plugin_train = predict_tau_tarnet(plugin_model, X[tr])

        m_hat, e_hat = cross_fit_nuisance(
            X[tr], Y[tr], T[tr],
            use_gating=nuisance_gating,
            lambda_sparsity=ctrl_config.lambda_sparsity,
            seed=fold_seed,
            k_folds=ctrl_config.k_folds,
            dropout_p=ctrl_config.dropout_p,
            hidden_dim=ctrl_config.hidden_dim,
            batch_size=ctrl_config.batch_size,
            epochs=ctrl_config.nuisance_epochs,
        )
        e_hat = np.clip(e_hat, 0.01, 0.99)
        R = Y[tr] - m_hat
        W = T[tr] - e_hat
        Z, weights = stabilize_residuals(
            R, W,
            w_clip=ctrl_config.w_clip,
            z_clip=ctrl_config.z_clip,
            eps=ctrl_config.eps,
            w_clip_quantile=ctrl_config.w_clip_quantile,
            z_clip_quantile=ctrl_config.z_clip_quantile,
        )

        tau_model = train_rlearner(
            X[tr], Z, weights,
            use_gating=tau_gating,
            lambda_tau=ctrl_config.lambda_tau,
            seed=fold_seed,
            dropout_p=ctrl_config.dropout_p,
            hidden_dim=ctrl_config.hidden_tau,
            batch_size=ctrl_config.batch_size,
            epochs=ctrl_config.tau_epochs,
            lr=ctrl_config.lr_tau,
            grad_clip=ctrl_config.grad_clip,
            warm_start_from=plugin_model,
            warm_start_backbone=warm_start_backbone,
            teacher_tau=tau_plugin_train,
            aux_beta_start=ctrl_config.aux_beta_start,
            aux_beta_end=ctrl_config.aux_beta_end,
            aux_decay_epochs=ctrl_config.aux_decay_epochs,
        )
        tau_oof[val] = predict_tau_rlearner(tau_model, X[val])

    return float(np.mean(tau_oof))


def _estimate_cf_ate_inference(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, Optional[float], Optional[float]]:
    from econml.dml import CausalForestDML
    cf = CausalForestDML(n_estimators=200 if not FAST_RUN else 100, random_state=seed)
    cf.fit(Y, T, X=X)

    if hasattr(cf, "ate_inference"):
        inf = cf.ate_inference(X)
        ate = float(getattr(inf, "mean_point", np.mean(cf.effect(X))))
        if hasattr(inf, "conf_int"):
            ci = inf.conf_int(alpha=alpha)
            return ate, float(ci[0]), float(ci[1])
        if hasattr(inf, "conf_int_mean"):
            ci = inf.conf_int_mean(alpha=alpha)
            return ate, float(ci[0]), float(ci[1])
        if hasattr(inf, "mean_ci_lower") and hasattr(inf, "mean_ci_upper"):
            return ate, float(inf.mean_ci_lower), float(inf.mean_ci_upper)

    if hasattr(cf, "effect_inference"):
        inf = cf.effect_inference(X)
        summary = inf.population_summary()
        ate = float(summary.mean_point)
        if hasattr(summary, "conf_int_mean"):
            ci = summary.conf_int_mean(alpha=alpha)
            return ate, float(ci[0]), float(ci[1])
        if hasattr(summary, "mean_ci_lower") and hasattr(summary, "mean_ci_upper"):
            return ate, float(summary.mean_ci_lower), float(summary.mean_ci_upper)

    return float(np.mean(cf.effect(X))), None, None


def run_ate_estimation(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    dataset_name: str,
    seed: int = 42,
    k_folds: int = 5,
    n_bootstrap: int = 30,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run ATE estimation with multiple methods.
    """
    rows = []
    ctrl_config = CTRLConfig()
    ctrl_config.k_folds = k_folds

    if FAST_RUN:
        ctrl_config.plugin_epochs = 100
        ctrl_config.nuisance_epochs = 80
        ctrl_config.tau_epochs = 150
        ctrl_config.k_folds = min(3, k_folds)
        n_bootstrap = min(20, n_bootstrap)
        k_folds = ctrl_config.k_folds

    set_seed(seed)

    # OLS baseline
    ate_ols, ols_ci_lo, ols_ci_hi = bootstrap_refit_ate(
        _estimate_ols_ate, X, T, Y, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha
    )
    rows.append({
        "dataset": dataset_name,
        "method": "OLS",
        "ate": ate_ols,
        "ci_lower": ols_ci_lo,
        "ci_upper": ols_ci_hi,
    })

    # TARNet baseline (no gating, no sparsity)
    def _tarnet_estimator(Xi, Ti, Yi, est_seed):
        return _estimate_tarnet_oof_ate(
            Xi, Ti, Yi, ctrl_config=ctrl_config, seed=est_seed, k_folds=k_folds
        )

    ate_tarnet, tarnet_ci_lo, tarnet_ci_hi = bootstrap_refit_ate(
        _tarnet_estimator, X, T, Y, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha
    )
    rows.append({
        "dataset": dataset_name,
        "method": "TARNet",
        "ate": ate_tarnet,
        "ci_lower": tarnet_ci_lo,
        "ci_upper": tarnet_ci_hi,
    })

    # CTRL-DML
    def _ctrl_dml_estimator(Xi, Ti, Yi, est_seed):
        return _estimate_ctrl_dml_oof_ate(
            Xi, Ti, Yi, ctrl_config=ctrl_config, seed=est_seed, k_folds=k_folds
        )

    ate_ctrl, ctrl_ci_lo, ctrl_ci_hi = bootstrap_refit_ate(
        _ctrl_dml_estimator, X, T, Y, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha
    )

    rows.append({
        "dataset": dataset_name,
        "method": "CTRL-DML",
        "ate": ate_ctrl,
        "ci_lower": ctrl_ci_lo,
        "ci_upper": ctrl_ci_hi,
    })

    # CausalForest (if available)
    try:
        ate_cf, cf_ci_lo, cf_ci_hi = _estimate_cf_ate_inference(X, T, Y, seed=seed, alpha=alpha)
        if cf_ci_lo is None or cf_ci_hi is None:
            def _cf_estimator(Xi, Ti, Yi, est_seed):
                return _estimate_cf_ate_inference(Xi, Ti, Yi, seed=est_seed, alpha=alpha)[0]
            ate_cf, cf_ci_lo, cf_ci_hi = bootstrap_refit_ate(
                _cf_estimator, X, T, Y, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha
            )
        rows.append({
            "dataset": dataset_name,
            "method": "CausalForest",
            "ate": ate_cf,
            "ci_lower": cf_ci_lo,
            "ci_upper": cf_ci_hi,
        })
    except ImportError:
        pass

    return pd.DataFrame(rows)


def run_balance_check(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    dataset_name: str,
    seed: int = 42,
    k_folds: int = 5,
) -> pd.DataFrame:
    """Check covariate balance before/after weighting."""
    rows = []
    ctrl_config = CTRLConfig()
    ctrl_config.k_folds = k_folds
    if FAST_RUN:
        ctrl_config.k_folds = min(3, k_folds)
    set_seed(seed)
    _, nuisance_gating, _ = resolve_gate_flags(ctrl_config)

    # Unweighted balance
    unweighted = compute_smd(X, T)
    rows.append({
        "dataset": dataset_name,
        "weighting": "unweighted",
        **unweighted,
    })

    # Propensity-weighted balance
    m_hat, e_hat = cross_fit_nuisance(
        X, Y, T,
        use_gating=nuisance_gating,
        lambda_sparsity=ctrl_config.lambda_sparsity,
        seed=seed,
        k_folds=ctrl_config.k_folds,
        dropout_p=ctrl_config.dropout_p,
        hidden_dim=ctrl_config.hidden_dim,
        batch_size=ctrl_config.batch_size,
        epochs=ctrl_config.nuisance_epochs if not FAST_RUN else 80,
    )
    e_hat = np.clip(e_hat, 0.01, 0.99)

    # IPW weights
    ipw = T / e_hat + (1 - T) / (1 - e_hat)
    weighted = compute_smd(X, T, weights=ipw)
    rows.append({
        "dataset": dataset_name,
        "weighting": "CTRL-IPW",
        **weighted,
    })

    return pd.DataFrame(rows)


def print_latex_reminder():
    """Print reminder to regenerate LaTeX macros."""
    print("\nTo regenerate LaTeX macros, run: python -m src.utils.latex")


def main():
    parser = argparse.ArgumentParser(description="CTRL-DML Real Data Experiments")
    parser.add_argument("--dataset", choices=["lalonde", "star", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=0, help="Override bootstrap iterations")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    output = get_output_manager()

    all_ate = []
    all_balance = []

    if args.dataset in ["lalonde", "all"]:
        try:
            print("=== LaLonde Dataset ===")
            X, T, Y = load_lalonde()
            print(f"  Loaded: N={len(Y)}, treated={T.sum():.0f}")

            n_bootstrap = args.bootstrap if args.bootstrap > 0 else 30
            ate_df = run_ate_estimation(
                X, T, Y, "lalonde",
                seed=args.seed,
                k_folds=args.k_folds,
                n_bootstrap=n_bootstrap,
                alpha=args.alpha,
            )
            balance_df = run_balance_check(
                X, T, Y, "lalonde",
                seed=args.seed,
                k_folds=args.k_folds,
            )

            # Save per-dataset files (matching existing naming)
            ate_df.to_csv(output.csv_path("realdata_lalonde_ate"), index=False)
            balance_df.to_csv(output.csv_path("realdata_lalonde_smd"), index=False)

            all_ate.append(ate_df)
            all_balance.append(balance_df)

            print("\nATE Estimates:")
            print(ate_df.to_string(index=False))
            print("\nBalance Check:")
            print(balance_df.to_string(index=False))
        except (ImportError, FileNotFoundError) as e:
            print(f"  Skipping LaLonde: {e}")

    if args.dataset in ["star", "all"]:
        try:
            print("\n=== STAR Dataset ===")
            X, T, Y = load_star()
            print(f"  Loaded: N={len(Y)}, treated={T.sum():.0f}")

            n_bootstrap = args.bootstrap if args.bootstrap > 0 else 30
            ate_df = run_ate_estimation(
                X, T, Y, "star",
                seed=args.seed,
                k_folds=args.k_folds,
                n_bootstrap=n_bootstrap,
                alpha=args.alpha,
            )
            balance_df = run_balance_check(
                X, T, Y, "star",
                seed=args.seed,
                k_folds=args.k_folds,
            )

            # Save per-dataset files (matching existing naming)
            ate_df.to_csv(output.csv_path("realdata_star_ate"), index=False)
            balance_df.to_csv(output.csv_path("realdata_star_smd"), index=False)

            all_ate.append(ate_df)
            all_balance.append(balance_df)

            print("\nATE Estimates:")
            print(ate_df.to_string(index=False))
            print("\nBalance Check:")
            print(balance_df.to_string(index=False))
        except (ImportError, FileNotFoundError) as e:
            print(f"  Skipping STAR: {e}")

    if all_ate:
        ate_combined = pd.concat(all_ate, ignore_index=True)
        csv_path = output.csv_path("realdata_ate")
        ate_combined.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    if all_balance:
        balance_combined = pd.concat(all_balance, ignore_index=True)
        csv_path = output.csv_path("realdata_balance")
        balance_combined.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        print_latex_reminder()


if __name__ == "__main__":
    main()
