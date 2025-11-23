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

# === 2. Dynamic Data Generator ===
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
    
    # Same generation logic as data_gen.py
    # Propensity depends on C and I
    logit = (np.sum(C, axis=1) + 0.5 * C[:, 0] * C[:, 1] + 2 * np.sum(I, axis=1))
    propensity = expit(logit)
    T = np.random.binomial(1, propensity)
    
    # Outcome depends on C and T (and C^2), but NOT I or N
    true_te = 2 * np.sin(C[:, 0] * np.pi) + np.maximum(0, C[:, 1])
    y0 = np.sum(C, axis=1) ** 2 + np.random.normal(0, 0.5, size=n_samples)
    y = y0 + true_te * T
    
    return X, T, y, true_te

# === 3. Train CTRL-DML Wrapper (with Sparsity) ===
def train_ctrl_dml(X_np, y_np, T_np, n_noise):
    # Convert to Tensor
    X = torch.from_numpy(X_np).float().to(DEVICE)
    y = torch.from_numpy(y_np).float().to(DEVICE) # Shape (N,)
    T = torch.from_numpy(T_np).long().to(DEVICE) # Shape (N,) - Must be Long for CrossEntropy
    
    # Initialize Model
    # Note: n_unit_in instead of n_input
    model = CTRLDML(n_unit_in=X.shape[1], n_iter=1, batch_size=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    epochs = 1000 
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

# === 4. Main Loop: Robustness Test ===
def run_robustness():
    noise_levels = [10, 30, 50, 80] 
    scores_baseline = []
    scores_ours = []

    print(f"Starting Robustness Stress Test (Noise Levels: {noise_levels})...")

    for n_noise in noise_levels:
        print(f"\n>>> Testing Noise Dim: {n_noise} ...")
        
        # A. Generate Data
        X, T, y, true_te = get_stress_data_dynamic(n_samples=2000, n_noise=n_noise)
        
        # B. Run Standard DragonNet (Baseline)
        print("   Training Baseline...")
        # Note: n_unit_in, lr
        model_base = StandardDragonNet(n_unit_in=X.shape[1], n_iter=1500, batch_size=256, lr=1e-3)
        model_base.fit(X, y, T)
        
        # Predict
        pred_base_tensor = model_base.predict(torch.from_numpy(X).float().to(DEVICE))
        pred_base = pred_base_tensor.cpu().detach().numpy().flatten()
        
        pehe_base = np.sqrt(np.mean((true_te - pred_base)**2))
        scores_baseline.append(pehe_base)
        print(f"   [Baseline] PEHE: {pehe_base:.4f}")
        
        # C. Run CTRL-DML (Ours)
        print("   Training CTRL-DML...")
        model_ours = train_ctrl_dml(X, y, T, n_noise)
        
        # Predict
        model_ours.eval()
        with torch.no_grad():
            pred_ours_tensor = model_ours.predict(torch.from_numpy(X).float().to(DEVICE))
            pred_ours = pred_ours_tensor.cpu().detach().numpy().flatten()
            
        pehe_ours = np.sqrt(np.mean((true_te - pred_ours)**2))
        scores_ours.append(pehe_ours)
        print(f"   [CTRL-DML] PEHE: {pehe_ours:.4f}")

    # === 5. Plotting ===
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, scores_baseline, label='Standard DragonNet', marker='o', linestyle='--', color='red', linewidth=2)
    plt.plot(noise_levels, scores_ours, label='CTRL-DML (Ours)', marker='s', linestyle='-', color='green', linewidth=3)

    plt.xlabel("Number of Noise Features (Difficulty Level)", fontsize=12)
    plt.ylabel("PEHE Error (Lower is Better)", fontsize=12)
    plt.title("Robustness Analysis: CTRL-DML vs Baseline", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    output_path = "robustness_curve.pdf"
    plt.savefig(output_path, dpi=300)
    print(f"\nTest Complete! Result saved to {output_path}")

if __name__ == "__main__":
    run_robustness()
