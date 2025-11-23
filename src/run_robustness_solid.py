import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim
from torch.nn import functional as F
from scipy.special import expit
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add external/CATENets to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'external', 'CATENets'))

# === 1. Import Models ===
from catenets.models.torch import DragonNet as StandardDragonNet
from my_dragonnet import MyDragonNet as CTRLDML
from catenets.models.torch.base import DEVICE

# === 2. Helper: Set Seed ===
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# === 3. Dynamic Data Generator ===
def get_stress_data_dynamic(n_samples=2000, n_noise=10):
    """
    Generate data with varying noise dimensions.
    """
    # Fixed parts
    C = np.random.normal(0, 1, size=(n_samples, 5)) # Confounders
    I = np.random.normal(0, 1, size=(n_samples, 5)) # Instruments
    
    # Dynamic part: Noise
    N = np.random.normal(0, 1, size=(n_samples, n_noise))
    
    X = np.concatenate([C, I, N], axis=1)
    
    # Propensity depends on C and I
    logit = (np.sum(C, axis=1) + 0.5 * C[:, 0] * C[:, 1] + 2 * np.sum(I, axis=1))
    propensity = expit(logit)
    T = np.random.binomial(1, propensity)
    
    # Outcome depends on C and T (and C^2), but NOT I or N
    true_te = 2 * np.sin(C[:, 0] * np.pi) + np.maximum(0, C[:, 1])
    y0 = np.sum(C, axis=1) ** 2 + np.random.normal(0, 0.5, size=n_samples)
    y = y0 + true_te * T
    
    return X, T, y, true_te

# === 4. Train CTRL-DML Wrapper (with Sparsity) ===
def train_ctrl_dml(X_np, y_np, T_np):
    # Convert to Tensor
    X = torch.from_numpy(X_np).float().to(DEVICE)
    y = torch.from_numpy(y_np).float().to(DEVICE) # Shape (N,)
    T = torch.from_numpy(T_np).long().to(DEVICE) # Shape (N,)
    
    # Initialize Model
    model = CTRLDML(n_unit_in=X.shape[1], n_iter=1, batch_size=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    epochs = 800 
    batch_size = 256
    lambda_sparsity = 0.05
    n_samples = X.shape[0]
    
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            indices = permutation[i : i + batch_size]
            batch_X, batch_y, batch_T = X[indices], y[indices], T[indices]
            
            optimizer.zero_grad()
            
            # Use _step to get predictions
            po_preds, prop_preds, discr = model._step(batch_X, batch_T)
            
            # Calculate Base Loss
            base_loss = model.loss(po_preds, prop_preds, batch_y, batch_T, discr)
            
            # Sparsity Regularization
            reg_loss = model._repr_estimator[0].current_mask_penalty
            
            total_loss = base_loss + (lambda_sparsity * reg_loss)
            total_loss.backward()
            optimizer.step()
            
    return model

# === 5. Main Loop: Solid Robustness Test ===
def run_robustness_solid():
    noise_levels = [10, 50, 100] 
    seeds = [42, 1024, 2023, 7, 999]

    # Result Matrix: [Noise_Level, Seed]
    results_baseline = np.zeros((len(noise_levels), len(seeds)))
    results_ours = np.zeros((len(noise_levels), len(seeds)))

    print(f"Starting Solid Robustness Test (Noise: {noise_levels}, Seeds: {len(seeds)})...")

    for i, n_noise in enumerate(noise_levels):
        print(f"\n>>> Testing Noise Dim: {n_noise}")
        for j, seed in enumerate(seeds):
            set_seed(seed) # Lock randomness
            
            # 1. Generate Data
            X, T, y, true_te = get_stress_data_dynamic(n_samples=2000, n_noise=n_noise)
            
            # 2. Baseline
            model_base = StandardDragonNet(n_unit_in=X.shape[1], n_iter=1000, batch_size=256, lr=1e-3)
            model_base.fit(X, y, T)
            
            # Predict Baseline
            pred_base_tensor = model_base.predict(torch.from_numpy(X).float().to(DEVICE))
            pred_base = pred_base_tensor.cpu().detach().numpy().flatten()
            
            pehe_base = np.sqrt(np.mean((true_te - pred_base)**2))
            results_baseline[i, j] = pehe_base
            
            # 3. Ours
            model_ours = train_ctrl_dml(X, y, T)
            
            # Predict Ours
            model_ours.eval()
            with torch.no_grad():
                pred_ours_tensor = model_ours.predict(torch.from_numpy(X).float().to(DEVICE))
                pred_ours = pred_ours_tensor.cpu().detach().numpy().flatten()
            
            pehe_ours = np.sqrt(np.mean((true_te - pred_ours)**2))
            results_ours[i, j] = pehe_ours
            
            print(f"    Seed {seed}: Base={pehe_base:.2f} | Ours={pehe_ours:.2f}")

    # === 6. Statistics & Plotting ===
    base_mean = np.mean(results_baseline, axis=1)
    base_std = np.std(results_baseline, axis=1)
    ours_mean = np.mean(results_ours, axis=1)
    ours_std = np.std(results_ours, axis=1)

    plt.figure(figsize=(10, 6))

    # Plot Baseline (Red Line + Red Shadow)
    plt.plot(noise_levels, base_mean, label='Standard DragonNet', marker='o', color='#d62728', linestyle='--', linewidth=2)
    plt.fill_between(noise_levels, base_mean - base_std, base_mean + base_std, color='#d62728', alpha=0.15)

    # Plot Ours (Green Line + Green Shadow)
    plt.plot(noise_levels, ours_mean, label='CTRL-DML (Ours)', marker='s', color='#2ca02c', linestyle='-', linewidth=3)
    plt.fill_between(noise_levels, ours_mean - ours_std, ours_mean + ours_std, color='#2ca02c', alpha=0.2)

    plt.xlabel("Number of Noise Features (Complexity)", fontsize=12)
    plt.ylabel("PEHE Error (Lower is Better)", fontsize=12)
    plt.title("Robustness Analysis with Confidence Intervals (5 Runs)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = "robustness_solid.pdf"
    plt.savefig(output_path, dpi=300)
    print(f"\nTest Complete! Solid chart generated: {output_path}")

if __name__ == "__main__":
    run_robustness_solid()
