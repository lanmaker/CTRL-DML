"""
Common metrics and evaluation functions for CTRL-DML.
"""
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from scipy import stats


def compute_pehe(tau_pred: np.ndarray, tau_true: np.ndarray) -> float:
    """
    Compute Precision in Estimation of Heterogeneous Effects (PEHE).

    PEHE = sqrt(mean((tau_true - tau_pred)^2))

    Args:
        tau_pred: Predicted treatment effects
        tau_true: True treatment effects

    Returns:
        PEHE value (lower is better)
    """
    return float(np.sqrt(np.mean((tau_true - tau_pred) ** 2)))


def compute_ate(tau_pred: np.ndarray) -> float:
    """
    Compute Average Treatment Effect from individual predictions.

    Args:
        tau_pred: Predicted individual treatment effects

    Returns:
        ATE (mean of individual effects)
    """
    return float(np.mean(tau_pred))


def compute_ate_bias(tau_pred: np.ndarray, tau_true: np.ndarray) -> float:
    """
    Compute ATE bias.

    Args:
        tau_pred: Predicted treatment effects
        tau_true: True treatment effects

    Returns:
        ATE bias (predicted - true)
    """
    return compute_ate(tau_pred) - compute_ate(tau_true)


def bootstrap_mean(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean.

    Args:
        values: Array of values
        n_bootstrap: Number of bootstrap iterations
        alpha: Significance level (default 0.05 for 95% CI)
        seed: Random seed

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    means = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        means.append(np.mean(values[idx]))

    means = np.array(means)
    mean_est = np.mean(values)
    ci_lower = np.percentile(means, 100 * alpha / 2)
    ci_upper = np.percentile(means, 100 * (1 - alpha / 2))

    return mean_est, ci_lower, ci_upper


def bootstrap_ci(
    estimate_fn,
    data: Dict[str, np.ndarray],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Generic bootstrap CI for any estimator.

    Args:
        estimate_fn: Function that takes data dict and returns estimate
        data: Dictionary of arrays (all same length)
        n_bootstrap: Number of bootstrap iterations
        alpha: Significance level
        seed: Random seed

    Returns:
        Tuple of (estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(list(data.values())[0])
    estimates = []

    # Point estimate
    point_est = estimate_fn(data)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_data = {k: v[idx] for k, v in data.items()}
        estimates.append(estimate_fn(boot_data))

    estimates = np.array(estimates)
    ci_lower = np.percentile(estimates, 100 * alpha / 2)
    ci_upper = np.percentile(estimates, 100 * (1 - alpha / 2))

    return point_est, ci_lower, ci_upper


def compute_smd(
    x: np.ndarray,
    t: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute Standardized Mean Difference (SMD) for a single covariate.

    SMD = (mean_treated - mean_control) / pooled_std

    Args:
        x: Covariate values
        t: Treatment indicator (0/1)
        weights: Optional sample weights

    Returns:
        SMD value
    """
    t = t.astype(bool)

    if weights is None:
        mean_t = np.mean(x[t])
        mean_c = np.mean(x[~t])
        var_t = np.var(x[t], ddof=1) if np.sum(t) > 1 else 0
        var_c = np.var(x[~t], ddof=1) if np.sum(~t) > 1 else 0
    else:
        # Weighted means and variances
        w_t = weights[t]
        w_c = weights[~t]
        w_t = w_t / w_t.sum() if w_t.sum() > 0 else w_t
        w_c = w_c / w_c.sum() if w_c.sum() > 0 else w_c

        mean_t = np.average(x[t], weights=w_t) if len(w_t) > 0 else 0
        mean_c = np.average(x[~t], weights=w_c) if len(w_c) > 0 else 0

        # Weighted variance
        var_t = np.average((x[t] - mean_t) ** 2, weights=w_t) if len(w_t) > 1 else 0
        var_c = np.average((x[~t] - mean_c) ** 2, weights=w_c) if len(w_c) > 1 else 0

    pooled_std = np.sqrt((var_t + var_c) / 2)

    if pooled_std < 1e-10:
        return 0.0

    return (mean_t - mean_c) / pooled_std


def compute_all_smd(
    X: np.ndarray,
    t: np.ndarray,
    weights: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute SMD for all covariates.

    Args:
        X: Covariate matrix (n_samples, n_features)
        t: Treatment indicator
        weights: Optional sample weights
        feature_names: Optional feature names

    Returns:
        Dictionary mapping feature name to SMD
    """
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]

    smds = {}
    for i, name in enumerate(feature_names):
        smds[name] = compute_smd(X[:, i], t, weights)

    return smds


def coverage_rate(
    tau_true: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray
) -> float:
    """
    Compute coverage rate of confidence intervals.

    Args:
        tau_true: True treatment effects
        ci_lower: Lower bounds of intervals
        ci_upper: Upper bounds of intervals

    Returns:
        Coverage rate (fraction of true values within intervals)
    """
    covered = (tau_true >= ci_lower) & (tau_true <= ci_upper)
    return float(np.mean(covered))


def interval_width(ci_lower: np.ndarray, ci_upper: np.ndarray) -> float:
    """
    Compute average interval width.

    Args:
        ci_lower: Lower bounds
        ci_upper: Upper bounds

    Returns:
        Mean interval width
    """
    return float(np.mean(ci_upper - ci_lower))


def e_value(
    rr: float,
    rr_ci_lower: float = None,
    is_log_rr: bool = False
) -> tuple:
    """
    Compute E-value for sensitivity analysis.

    The E-value is the minimum strength of unmeasured confounding
    (on the risk ratio scale) that would be needed to explain away
    the observed effect.

    For a risk ratio RR >= 1:
        E-value = RR + sqrt(RR * (RR - 1))

    For RR < 1, we use 1/RR to compute the E-value.

    Args:
        rr: Risk ratio (or log risk ratio if is_log_rr=True)
        rr_ci_lower: Lower bound of confidence interval for RR
                     (used to compute E-value for CI)
        is_log_rr: If True, rr is on log scale and will be exponentiated

    Returns:
        Tuple of (e_val_point, e_val_ci) where e_val_ci is None if
        rr_ci_lower not provided
    """
    if is_log_rr:
        rr = np.exp(rr)
        if rr_ci_lower is not None:
            rr_ci_lower = np.exp(rr_ci_lower)

    # E-value formula works for RR >= 1, so flip if needed
    rr_use = rr if rr >= 1 else 1 / rr
    e_val_point = rr_use + np.sqrt(rr_use * (rr_use - 1))

    e_val_ci = None
    if rr_ci_lower is not None:
        # For CI, use the bound closer to 1
        if rr >= 1:
            rr_ci_use = max(rr_ci_lower, 1.0)  # If CI crosses 1, E-value = 1
        else:
            rr_ci_use = min(1 / rr_ci_lower, 1.0) if rr_ci_lower > 0 else 1.0

        if rr_ci_use <= 1:
            e_val_ci = 1.0  # CI crosses null, E-value is 1
        else:
            e_val_ci = rr_ci_use + np.sqrt(rr_ci_use * (rr_ci_use - 1))

    return e_val_point, e_val_ci


def ate_to_approximate_rr(
    ate: float,
    outcome_sd: float = 1.0,
    baseline_risk: float = None
) -> float:
    """
    Convert ATE to approximate risk ratio for E-value computation.

    For continuous outcomes, uses Cohen's d approximation:
        RR ≈ exp(0.91 * d) where d = ATE / outcome_sd

    For binary outcomes with known baseline risk:
        RR = (baseline_risk + ATE) / baseline_risk

    Args:
        ate: Average treatment effect
        outcome_sd: Standard deviation of outcome (for continuous)
        baseline_risk: Baseline risk for control group (for binary outcomes)

    Returns:
        Approximate risk ratio
    """
    if baseline_risk is not None:
        # Binary outcome
        p1 = np.clip(baseline_risk + ate, 0.001, 0.999)
        p0 = np.clip(baseline_risk, 0.001, 0.999)
        return p1 / p0
    else:
        # Continuous outcome - use Cohen's d approximation
        d = ate / outcome_sd
        return np.exp(0.91 * np.abs(d))


def summarize_results(
    results: List[Dict[str, Any]],
    metrics: List[str] = ["pehe", "ate_bias"]
) -> Dict[str, Dict[str, float]]:
    """
    Summarize results across multiple runs.

    Args:
        results: List of result dictionaries
        metrics: Metrics to summarize

    Returns:
        Dictionary with mean, std, median, iqr for each metric
    """
    summary = {}
    for metric in metrics:
        values = [r[metric] for r in results if metric in r]
        if len(values) == 0:
            continue
        values = np.array(values)
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary
