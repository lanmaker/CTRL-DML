"""
Synthetic data generators for CTRL-DML experiments.

This module provides:
- get_stress_data: Semi-synthetic stress test with confounders, instruments, and noise
- generate_complex_data: More complex DGP with nonlinear effects
"""
import numpy as np
from scipy.special import expit
from typing import Tuple, Optional


def get_stress_data(
    n_samples: int = 2000,
    n_noise: int = 50,
    n_confounders: int = 5,
    n_instruments: int = 5,
    n_prognostic: int = 5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate semi-synthetic stress test with all four feature roles.

    This DGP is designed to test robustness to high-dimensional noise:
    - Confounders (C): affect both T and Y
    - Instruments (I): affect only T
    - Prognostic (P): affect only Y (not T)
    - Noise (N): affect neither T nor Y

    Feature ordering in X: [Confounders, Instruments, Prognostic, Noise]

    Args:
        n_samples: Number of samples
        n_noise: Number of noise dimensions
        n_confounders: Number of confounder dimensions
        n_instruments: Number of instrument dimensions
        n_prognostic: Number of prognostic dimensions
        seed: Random seed

    Returns:
        Tuple of (X, T, Y, true_te)
            X: Features (n_samples, n_confounders + n_instruments + n_prognostic + n_noise)
            T: Treatment indicator (n_samples,)
            Y: Outcome (n_samples,)
            true_te: Ground-truth CATE (n_samples,)
    """
    rng = np.random.default_rng(seed)

    # Generate features
    C = rng.normal(0, 1, size=(n_samples, n_confounders))   # Confounders
    I = rng.normal(0, 1, size=(n_samples, n_instruments))   # Instruments
    P = rng.normal(0, 1, size=(n_samples, n_prognostic))    # Prognostic
    N = rng.normal(0, 1, size=(n_samples, n_noise))         # Noise
    X = np.concatenate([C, I, P, N], axis=1).astype(np.float32)

    # Treatment assignment (depends on C and I, not P or N)
    if n_confounders >= 2:
        logit = np.sum(C, axis=1) + 0.5 * C[:, 0] * C[:, 1] + 2 * np.sum(I, axis=1)
    elif n_confounders == 1:
        logit = C[:, 0] + 2 * np.sum(I, axis=1)
    else:
        logit = 2 * np.sum(I, axis=1)
    propensity = expit(logit)
    T = rng.binomial(1, propensity).astype(np.float32)

    # Outcome (depends on C, P, and T, not I or N)
    # IMPORTANT: true_te depends on PROGNOSTIC features (P), not confounders (C)
    # This allows the bias/variance test to work properly:
    # - Dropping C tests debiasing ability (C affects T and baseline Y)
    # - But heterogeneity (true_te) is still learnable from P
    if n_prognostic >= 2:
        true_te = (2.0 + 1.5 * np.sin(P[:, 0] * np.pi) + 0.8 * np.maximum(0, P[:, 1])).astype(np.float32)
    elif n_prognostic == 1:
        true_te = (2.0 + 1.5 * np.sin(P[:, 0] * np.pi)).astype(np.float32)
    else:
        true_te = np.ones(n_samples, dtype=np.float32) * 2.0  # Constant treatment effect

    # y0 (baseline outcome) includes confounders and prognostic features
    # Confounders create bias (affect both T via propensity and Y via y0)
    y0_conf = np.sum(C, axis=1) ** 2 if n_confounders > 0 else np.zeros(n_samples)
    y0_prog = 0.5 * np.sum(P, axis=1) if n_prognostic > 0 else np.zeros(n_samples)
    y0_prog_nl = 0.3 * P[:, 0] ** 2 if n_prognostic > 0 else np.zeros(n_samples)
    y0 = y0_conf + y0_prog + y0_prog_nl + rng.normal(0, 0.5, size=n_samples)
    Y = (y0 + true_te * T).astype(np.float32)

    return X, T, Y, true_te


def generate_complex_data(
    n_samples: int = 2000,
    n_features: int = 25,
    treatment_effect_heterogeneity: str = "moderate",
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate complex synthetic data with various effect patterns.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        treatment_effect_heterogeneity: One of "none", "moderate", "high"
        seed: Random seed

    Returns:
        Tuple of (X, T, Y, true_te)
    """
    rng = np.random.default_rng(seed)

    # Generate features
    X = rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float32)

    # Treatment assignment
    logit = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]
    propensity = expit(logit)
    T = rng.binomial(1, propensity).astype(np.float32)

    # Base outcome
    y0 = X[:, 0] + 0.5 * X[:, 1]**2 - 0.3 * X[:, 2]

    # Treatment effect
    if treatment_effect_heterogeneity == "none":
        true_te = np.ones(n_samples) * 2.0
    elif treatment_effect_heterogeneity == "moderate":
        true_te = 2.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1]
    elif treatment_effect_heterogeneity == "high":
        true_te = 2.0 + np.sin(X[:, 0] * np.pi) + np.maximum(0, X[:, 1])
    else:
        raise ValueError(f"Unknown heterogeneity: {treatment_effect_heterogeneity}")

    true_te = true_te.astype(np.float32)
    Y = (y0 + true_te * T + rng.normal(0, 0.5, n_samples)).astype(np.float32)

    return X, T, Y, true_te


def get_ihdp_style_data(
    n_samples: int = 747,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data similar to IHDP benchmark structure.

    Args:
        n_samples: Number of samples
        seed: Random seed

    Returns:
        Tuple of (X, T, Y, true_te)
    """
    rng = np.random.default_rng(seed)

    # 25 features like IHDP
    X = np.zeros((n_samples, 25), dtype=np.float32)

    # 6 continuous features
    X[:, :6] = rng.normal(0, 1, size=(n_samples, 6))

    # 19 binary features
    X[:, 6:] = rng.binomial(1, 0.5, size=(n_samples, 19))

    # Treatment assignment (biased toward certain features)
    logit = -0.5 + 0.3 * X[:, 0] + 0.2 * X[:, 6] - 0.4 * X[:, 7]
    propensity = expit(logit)
    T = rng.binomial(1, propensity).astype(np.float32)

    # Outcome with heterogeneous effects
    mu0 = X[:, 0] + 0.5 * X[:, 1] + 0.25 * X[:, 6]
    mu1 = mu0 + 4.0 + 0.5 * X[:, 0] - 0.5 * X[:, 1]

    true_te = (mu1 - mu0).astype(np.float32)
    Y = (T * mu1 + (1 - T) * mu0 + rng.normal(0, 1, n_samples)).astype(np.float32)

    return X, T, Y, true_te


def get_feature_roles(
    n_confounders: int = 5,
    n_instruments: int = 5,
    n_prognostic: int = 5,
    n_noise: int = 50
):
    """
    Get indices for different feature roles in stress test data.

    Feature ordering: [Confounders, Instruments, Prognostic, Noise]

    Returns:
        Dictionary with 'confounders', 'instruments', 'prognostic', 'noise' keys
    """
    c_end = n_confounders
    i_end = c_end + n_instruments
    p_end = i_end + n_prognostic
    n_end = p_end + n_noise
    return {
        "confounders": list(range(0, c_end)),
        "instruments": list(range(c_end, i_end)),
        "prognostic": list(range(i_end, p_end)),
        "noise": list(range(p_end, n_end)),
    }


if __name__ == "__main__":
    # Test stress data
    X, T, Y, cate = get_stress_data(n_samples=1000, n_noise=50)
    print(f"Stress data: X={X.shape}, T mean={T.mean():.3f}")
    print(f"CATE: mean={cate.mean():.3f}, std={cate.std():.3f}")

    # Test complex data
    X, T, Y, cate = generate_complex_data(n_samples=1000)
    print(f"\nComplex data: X={X.shape}, T mean={T.mean():.3f}")
    print(f"CATE: mean={cate.mean():.3f}, std={cate.std():.3f}")
