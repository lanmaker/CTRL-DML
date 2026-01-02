"""
Feature Role Decomposition Analysis.

This script computes and visualizes feature role scores (confounder, instrument,
prognostic, noise) using TRAINED CTRL-DML model masks.

Usage:
    python -m src.experiments.feature_roles              # Uses model masks (default)
    python -m src.experiments.feature_roles --no-model   # Use linear proxy instead
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from src.data.synthetic import get_stress_data
from src.models.orthogonal_learner import set_seed, train_plugin, train_nuisance
from src.models.dragonnet import get_device
from src.utils.io import get_output_manager

DEVICE = get_device()

# Colors for this script
COLORS = {
    'primary': '#8B1C1C',
    'secondary': '#2C5F8A',
}


def extract_model_masks(
    X: np.ndarray,
    T: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    hidden_dim: int = 120,
    epochs: int = 100,
) -> tuple:
    """
    Train CTRL-DML models and extract learned mask weights.

    Returns:
        Tuple of (M_T, M_Y) - normalized mask weights for treatment and outcome
    """
    # Train nuisance model (has treatment and outcome heads)
    model = train_nuisance(
        X, y, T,
        use_gating=True,
        lambda_sparsity=0.05,
        seed=seed,
        hidden_dim=hidden_dim,
        epochs=epochs,
    )

    # Extract mask weights by passing data through
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(DEVICE)

        # Get mask from TabularAttention
        # The backbone contains the attention layer
        if hasattr(model.backbone, 'attn') and hasattr(model.backbone.attn, 'mask_net'):
            mask = model.backbone.attn.mask_net(x_t)
            mask_weights = mask.mean(dim=0).cpu().numpy()  # Average across samples
        else:
            # Fallback if structure is different
            mask_weights = np.ones(X.shape[1])

        # Get predictions for T and Y to compute importance
        y0, y1, t_prob = model(x_t)

    # Compute gradient-based importance for T head
    model.train()
    x_t.requires_grad_(True)
    _, _, t_logits = model(x_t)
    t_loss = t_logits.sum()
    t_loss.backward()
    M_T = np.abs(x_t.grad.mean(dim=0).cpu().numpy())
    M_T = M_T / (M_T.max() + 1e-8)

    # Compute gradient-based importance for Y heads
    x_t = torch.from_numpy(X).float().to(DEVICE)
    x_t.requires_grad_(True)
    y0, y1, _ = model(x_t)
    y_loss = (y0.sum() + y1.sum())
    y_loss.backward()
    M_Y = np.abs(x_t.grad.mean(dim=0).cpu().numpy())
    M_Y = M_Y / (M_Y.max() + 1e-8)

    return M_T, M_Y, mask_weights


def compute_feature_roles_linear(X: np.ndarray, T: np.ndarray, y: np.ndarray) -> tuple:
    """
    Compute feature role scores using linear models (baseline method).

    This is a proxy when model masks are not available.
    """
    from sklearn.linear_model import LogisticRegression, Ridge

    # Fit treatment model (logistic regression)
    lr_t = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)
    lr_t.fit(X, T)
    M_T = np.abs(lr_t.coef_[0])
    M_T = M_T / (M_T.max() + 1e-8)

    # Fit outcome model (ridge regression)
    ridge_y = Ridge(alpha=1.0)
    ridge_y.fit(X, y)
    M_Y = np.abs(ridge_y.coef_)
    M_Y = M_Y / (M_Y.max() + 1e-8)

    return M_T, M_Y


def compute_role_scores(M_T: np.ndarray, M_Y: np.ndarray) -> dict:
    """Compute role scores from mask weights."""
    # Normalize to ensure scores sum to 1
    Z = M_T * M_Y + M_T * (1 - M_Y) + (1 - M_T) * M_Y + (1 - M_T) * (1 - M_Y) + 1e-8

    s_conf = (M_T * M_Y) / Z
    s_inst = (M_T * (1 - M_Y)) / Z
    s_prog = ((1 - M_T) * M_Y) / Z
    s_noise = ((1 - M_T) * (1 - M_Y)) / Z

    return {
        's_conf': s_conf,
        's_inst': s_inst,
        's_prog': s_prog,
        's_noise': s_noise,
    }


def compute_feature_roles(
    X: np.ndarray,
    T: np.ndarray,
    y: np.ndarray,
    n_conf: int = 5,
    n_inst: int = 5,
    n_prog: int = 5,
    use_model: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute feature role scores.

    Args:
        X: Covariates (n_samples, n_features)
        T: Treatment (n_samples,)
        y: Outcome (n_samples,)
        n_conf: Number of true confounders
        n_inst: Number of true instruments
        n_prog: Number of true prognostic features
        use_model: If True, train CTRL-DML model and extract masks
        seed: Random seed

    Returns:
        DataFrame with role scores for each feature
    """
    n_features = X.shape[1]

    if use_model:
        print("  Training CTRL-DML model to extract masks...")
        M_T, M_Y, mask_weights = extract_model_masks(X, T, y, seed=seed)
    else:
        print("  Using linear proxy (logistic/ridge regression)...")
        M_T, M_Y = compute_feature_roles_linear(X, T, y)
        mask_weights = np.ones(n_features)

    # Compute role scores
    scores = compute_role_scores(M_T, M_Y)

    # Create DataFrame
    df = pd.DataFrame({
        'feature': range(n_features),
        'M_T': M_T,
        'M_Y': M_Y,
        'mask_weight': mask_weights,
        's_conf': scores['s_conf'],
        's_inst': scores['s_inst'],
        's_prog': scores['s_prog'],
        's_noise': scores['s_noise'],
    })

    # Add true labels based on feature ordering: [Conf, Inst, Prog, Noise]
    df['true_role'] = 'noise'
    c_end = n_conf
    i_end = c_end + n_inst
    p_end = i_end + n_prog
    df.loc[:c_end-1, 'true_role'] = 'confounder'
    df.loc[c_end:i_end-1, 'true_role'] = 'instrument'
    df.loc[i_end:p_end-1, 'true_role'] = 'prognostic'

    return df


def compute_precision_at_k(df: pd.DataFrame, role: str, k: int) -> float:
    """Compute precision@k for a given role."""
    score_col = f's_{role[:4]}'
    sorted_df = df.sort_values(score_col, ascending=False)
    top_k = sorted_df.head(k)

    if role == 'confounder':
        true_positives = (top_k['true_role'] == 'confounder').sum()
    elif role == 'instrument':
        true_positives = (top_k['true_role'] == 'instrument').sum()
    elif role == 'prognostic':
        true_positives = (top_k['true_role'] == 'prognostic').sum()
    else:
        true_positives = (top_k['true_role'] == 'noise').sum()

    return true_positives / k


def plot_feature_roles(df: pd.DataFrame, output_path: Path, use_model: bool = False):
    """Create feature role visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Bar chart of M_T and M_Y
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df['M_T'], width, label='$M_T$ (treatment)', color=COLORS['secondary'])
    ax.bar(x + width/2, df['M_Y'], width, label='$M_Y$ (outcome)', color=COLORS['primary'])

    ax.set_xlabel('Feature index')
    ax.set_ylabel('Importance weight')
    title_suffix = "(CTRL-DML masks)" if use_model else "(linear proxy)"
    ax.set_title(f'Feature importance {title_suffix}')
    ax.legend(loc='upper right', frameon=False)

    # Add vertical lines to separate feature types
    n_conf = (df['true_role'] == 'confounder').sum()
    n_inst = (df['true_role'] == 'instrument').sum()
    n_prog = (df['true_role'] == 'prognostic').sum()
    ax.axvline(n_conf - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(n_conf + n_inst - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(n_conf + n_inst + n_prog - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(n_conf/2, ax.get_ylim()[1]*0.95, 'Conf.', ha='center', fontsize=9)
    ax.text(n_conf + n_inst/2, ax.get_ylim()[1]*0.95, 'Inst.', ha='center', fontsize=9)
    ax.text(n_conf + n_inst + n_prog/2, ax.get_ylim()[1]*0.95, 'Prog.', ha='center', fontsize=9)
    ax.text(n_conf + n_inst + n_prog + 5, ax.get_ylim()[1]*0.95, 'Noise', ha='center', fontsize=9)

    # Right: Precision at k
    ax = axes[1]
    ks = [5, 10]
    roles = ['confounder', 'instrument']
    x = np.arange(len(ks))
    width = 0.35

    for i, role in enumerate(roles):
        precisions = [compute_precision_at_k(df, role, k) for k in ks]
        offset = (i - 0.5) * width
        color = COLORS['primary'] if role == 'confounder' else COLORS['secondary']
        bars = ax.bar(x + offset, precisions, width, label=role.capitalize(), color=color)

        for bar, val in zip(bars, precisions):
            ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in ks])
    ax.set_ylabel('Precision@k')
    ax.set_title('Feature role recovery')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Feature Role Analysis")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-noise", type=int, default=50)
    parser.add_argument("--n-conf", type=int, default=5)
    parser.add_argument("--n-inst", type=int, default=5)
    parser.add_argument("--n-prog", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-model", action="store_true",
                        help="Use linear proxy instead of trained CTRL-DML model masks")
    args = parser.parse_args()

    set_seed(args.seed)
    output = get_output_manager()

    print("=== Feature Role Decomposition ===")
    use_model = not args.no_model
    if use_model:
        print("Mode: Using CTRL-DML model masks")
    else:
        print("Mode: Using linear proxy (default uses model masks)")

    # Generate data
    print(f"Generating data (N={args.n_samples}, noise={args.n_noise})...")
    X, T, y, true_te = get_stress_data(
        n_samples=args.n_samples,
        n_noise=args.n_noise,
        n_confounders=args.n_conf,
        n_instruments=args.n_inst,
        n_prognostic=args.n_prog,
        seed=args.seed
    )

    # Compute feature roles
    print("Computing feature role scores...")
    df = compute_feature_roles(X, T, y,
                               n_conf=args.n_conf,
                               n_inst=args.n_inst,
                               n_prog=args.n_prog,
                               use_model=use_model, seed=args.seed)

    # Compute metrics
    prec_conf_5 = compute_precision_at_k(df, 'confounder', 5)
    prec_conf_10 = compute_precision_at_k(df, 'confounder', 10)
    prec_inst_5 = compute_precision_at_k(df, 'instrument', 5)
    prec_inst_10 = compute_precision_at_k(df, 'instrument', 10)
    prec_prog_5 = compute_precision_at_k(df, 'prognostic', 5)
    prec_prog_10 = compute_precision_at_k(df, 'prognostic', 10)

    # Recall@10 for confounders
    sorted_conf = df.sort_values('s_conf', ascending=False).head(10)
    recall_conf_10 = (sorted_conf['true_role'] == 'confounder').sum() / args.n_conf

    print(f"\nResults:")
    print(f"  Confounder precision@5:  {prec_conf_5:.2f}")
    print(f"  Confounder precision@10: {prec_conf_10:.2f}")
    print(f"  Confounder recall@10:    {recall_conf_10:.2f}")
    print(f"  Instrument precision@5:  {prec_inst_5:.2f}")
    print(f"  Instrument precision@10: {prec_inst_10:.2f}")
    print(f"  Prognostic precision@5:  {prec_prog_5:.2f}")
    print(f"  Prognostic precision@10: {prec_prog_10:.2f}")

    # Save results
    csv_path = output.csv_path("feature_roles")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Plot
    fig_path = output.figure_path("feature_roles")
    plot_feature_roles(df, fig_path, use_model=use_model)
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
