import numpy as np
import torch
import sys
import os

# Add external/CATENets to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'external', 'CATENets'))

from catenets.models.torch import DragonNet

def benchmark_baseline():
    print("Loading stress_data.npz...")
    data = np.load("stress_data.npz")
    X = data['X']
    T = data['T']
    y = data['y']
    true_te = data['true_te']
    
    print(f"Data shapes: X={X.shape}, T={T.shape}, y={y.shape}")
    
    # Initialize Standard DragonNet
    print("Initializing Standard DragonNet...")
    model = DragonNet(
        n_unit_in=X.shape[1],
        n_iter=2000, 
        batch_size=256,
        lr=1e-3,
        n_units_out_prop=100,
        n_units_out=100,
        val_split_prop=0.0 # Disable validation split to avoid stratification issues
    )
    
    print("Training Standard DragonNet...")
    model.fit(X, y, T)
    
    print("Predicting CATE...")
    pred_cate = model.predict(X)
    
    if isinstance(pred_cate, torch.Tensor):
        pred_cate = pred_cate.detach().cpu().numpy().flatten()
        
    # Calculate PEHE
    pehe = np.sqrt(np.mean((true_te - pred_cate)**2))
    
    print("-" * 30)
    print(f"Standard DragonNet Baseline on Stress Data")
    print(f"PEHE Score: {pehe:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    benchmark_baseline()
