import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add external/CATENets to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'external', 'CATENets'))

from my_dragonnet import MyDragonNet
from catenets.models.torch.base import DEVICE

def train_sparse():
    # 1. Load Data
    print("Loading stress_data.npz...")
    data = np.load("stress_data.npz")
    X = torch.from_numpy(data['X']).float().to(DEVICE)
    y = torch.from_numpy(data['y']).float().to(DEVICE) # Shape (N,)
    T = torch.from_numpy(data['T']).long().to(DEVICE) # Shape (N,)
    
    # 2. Initialize Model
    print("Initializing CTRL-DML (MyDragonNet)...")
    model = MyDragonNet(
        n_unit_in=X.shape[1], 
        n_iter=1, # Dummy value, we control loop
        batch_size=256,
        lr=1e-3,
        val_split_prop=0.0
    ).to(DEVICE)
    
    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # === Hyperparameters ===
    LAMBDA_SPARSITY = 0.05
    EPOCHS = 200
    BATCH_SIZE = 256
    N_SAMPLES = X.shape[0]
    
    print(f"Starting Sparse Training (Lambda={LAMBDA_SPARSITY}, Epochs={EPOCHS})...")
    
    # 4. Manual Training Loop
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(N_SAMPLES)
        
        epoch_loss = 0
        epoch_reg = 0
        
        for i in range(0, N_SAMPLES, BATCH_SIZE):
            indices = permutation[i : i + BATCH_SIZE]
            batch_X = X[indices]
            batch_y = y[indices]
            batch_T = T[indices]
            
            optimizer.zero_grad()
            
            # Use _step to get predictions and discrepancy
            # _step returns: po_preds, prop_preds, discr
            po_preds, prop_preds, discr = model._step(batch_X, batch_T)
            
            # Calculate Base Loss (Prediction + Propensity + Discrepancy)
            base_loss = model.loss(po_preds, prop_preds, batch_y, batch_T, discr)
            
            # Calculate Sparsity Regularization
            # Access the TabularAttention layer (first layer of _repr_estimator)
            reg_loss = model._repr_estimator[0].current_mask_penalty
            
            # Total Loss
            total_loss = base_loss + (LAMBDA_SPARSITY * reg_loss)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_reg += reg_loss.item()
            
        if (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / (N_SAMPLES / BATCH_SIZE)
            avg_reg = epoch_reg / (N_SAMPLES / BATCH_SIZE)
            print(f"Epoch {epoch+1} | Total Loss: {avg_loss:.4f} | Mask Penalty: {avg_reg:.4f}")
            
    # 5. Visualization
    print("Training complete. Generating visualization...")
    model.eval()
    with torch.no_grad():
        # Extract Mask
        # Need to move X to cpu for numpy conversion later if on gpu
        masks = model._repr_estimator[0].mask_net(X).cpu().numpy()
        mean_importance = np.mean(masks, axis=0)
        
    feature_names = [f"C{i}" for i in range(5)] + \
                    [f"I{i}" for i in range(5)] + \
                    [f"N{i}" for i in range(X.shape[1]-10)]
    
    colors = ['green']*5 + ['red']*5 + ['gray']*(len(feature_names)-10)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_names)), mean_importance, color=colors)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.ylabel("Attention Weight (Importance)")
    plt.title(f"CTRL-DML with Sparsity Penalty (lambda={LAMBDA_SPARSITY})")
    
    plt.axvline(x=4.5, color='black', linestyle='--')
    plt.axvline(x=9.5, color='black', linestyle='--')
    
    output_path = "feature_importance_sparse.pdf"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Result saved to {output_path}")
    
    # Numerical Analysis
    avg_c = np.mean(mean_importance[:5])
    avg_i = np.mean(mean_importance[5:10])
    avg_n = np.mean(mean_importance[10:])
    
    print("-" * 30)
    print("Feature Importance Analysis (Sparse):")
    print(f"Confounders (Green) Avg Weight: {avg_c:.4f}")
    print(f"Instruments (Red)   Avg Weight: {avg_i:.4f}")
    print(f"Noise (Gray)        Avg Weight: {avg_n:.4f}")
    print("-" * 30)

    # === Ablation Study: Calculate PEHE ===
    model.eval()
    with torch.no_grad():
        # 1. Get Test Data (using the same X since we are doing a simple benchmark on the whole dataset)
        # In a real scenario, we should use a hold-out test set, but for this benchmark we use the generated stress data
        X_test_tensor = X # Already on DEVICE
        true_te = data['true_te']
        
        # 2. Predict
        # model.predict returns CATE (y1 - y0) directly as a tensor
        cate_pred_tensor = model.predict(X_test_tensor)
        cate_pred = cate_pred_tensor.cpu().numpy().flatten()
        
        # 3. Calculate PEHE
        pehe_sparse = np.sqrt(np.mean((true_te - cate_pred)**2))
        
        # 3. Calculate PEHE
        pehe_sparse = np.sqrt(np.mean((true_te - cate_pred.flatten())**2))

    print("="*40)
    print(f"Ablation Study C (Full Method) Results:")
    print(f"CTRL-DML (lambda={LAMBDA_SPARSITY}) PEHE: {pehe_sparse:.4f}")
    print("="*40)

if __name__ == "__main__":
    train_sparse()
