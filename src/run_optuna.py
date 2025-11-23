import optuna
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from my_dragonnet import MyDragonNet as DragonNet
from optuna.visualization.matplotlib import plot_optimization_history

# === 1. Data Preparation (Train/Validation Split) ===
def get_data_for_tuning():
    # Generate data using the same logic as Stress Data
    n = 2000
    C = np.random.normal(0, 1, size=(n, 5)) 
    I = np.random.normal(0, 1, size=(n, 5))
    N = np.random.normal(0, 1, size=(n, 50)) # Fixed noise dimension at 50
    X = np.concatenate([C, I, N], axis=1)
    
    # Generate T and Y
    logit = np.sum(C, axis=1) + 2 * np.sum(I, axis=1)
    prop = 1 / (1 + np.exp(-logit))
    T = np.random.binomial(1, prop)
    true_te = 2 * np.sin(C[:, 0] * np.pi) + np.maximum(0, C[:, 1])
    y = np.sum(C, axis=1) + true_te * T + np.random.normal(0, 0.5, n)
    
    # Split Train / Val (80% / 20%)
    # Note: Never use Test Set for tuning!
    X_train, X_val, y_train, y_val, T_train, T_val = train_test_split(
        X, y, T, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val, T_train, T_val

# === 2. Define Objective Function ===
def objective(trial):
    # A. Suggest Hyperparameters
    # Search range: Learning Rate 1e-4 to 1e-2 (log scale)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # Search range: Sparsity Lambda 0.001 to 0.1
    lambda_sparsity = trial.suggest_float("lambda_sparsity", 0.001, 0.1)
    # Search range: Dropout 0.1 to 0.5
    dropout_p = trial.suggest_float("dropout_p", 0.1, 0.5)
    
    # B. Load Data
    X_train, X_val, y_train, y_val, T_train, T_val = get_data_for_tuning()
    
    # Convert to Tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X_train).float().to(device)
    y_t = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
    T_t = torch.from_numpy(T_train).long().to(device)
    
    X_v = torch.from_numpy(X_val).float().to(device)
    y_v = torch.from_numpy(y_val).float().unsqueeze(1).to(device)
    T_v = torch.from_numpy(T_val).long().to(device)
    
    # C. Initialize Model
    # Note: MyDragonNet uses n_unit_in and dropout_prob
    model = DragonNet(n_unit_in=X_train.shape[1], dropout_prob=dropout_p, n_iter=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # D. Fast Training (Fewer epochs for speed)
    epochs = 300 
    batch_size = 256
    
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_t.shape[0])
        for i in range(0, X_t.shape[0], batch_size):
            indices = permutation[i : i + batch_size]
            batch_X, batch_y, batch_T = X_t[indices], y_t[indices], T_t[indices]
            
            optimizer.zero_grad()
            
            # Forward pass
            po_preds, prop_preds, discr = model._step(batch_X, batch_T)
            
            # Calculate Loss
            base_loss = model.loss(po_preds, prop_preds, batch_y, batch_T, discr)
            reg_loss = model._repr_estimator[0].current_mask_penalty
            
            # Total Loss with suggested lambda
            loss = base_loss + (lambda_sparsity * reg_loss)
            
            loss.backward()
            optimizer.step()
            
        # [Optional] Pruning: Terminate early if performance is poor
        model.eval()
        with torch.no_grad():
            # Validation Loss
            po_preds_v, prop_preds_v, _ = model._step(X_v, T_v)
            # We use MSE on Y as the primary metric for tuning
            # Reconstruct predicted Y based on T
            y0_pred, y1_pred = po_preds_v[:, 0:1], po_preds_v[:, 1:2]
            T_v_expanded = T_v.unsqueeze(1).float()
            y_pred = T_v_expanded * y1_pred + (1 - T_v_expanded) * y0_pred
            val_loss = F.mse_loss(y_pred, y_v)
            
        model.train()
        
        trial.report(val_loss.item(), epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # E. Return Final Metric (Validation MSE Loss)
    return val_loss.item()

# === 3. Start Search ===
if __name__ == "__main__":
    print("🚀 Starting Hyperparameter Tuning with Optuna...")
    # Create Study, minimize Loss
    study = optuna.create_study(direction="minimize")
    
    # Run 5 trials for demonstration (increase to 50+ for real tuning)
    study.optimize(objective, n_trials=5)

    print("\n✅ Tuning Complete!")
    print("Best Trial:")
    print(f"  Value (Val Loss): {study.best_trial.value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
        
    # Visualization
    try:
        plt.figure(figsize=(10, 6))
        plot_optimization_history(study)
        plt.title("Optuna Optimization History")
        output_path = "optuna_history.pdf"
        plt.savefig(output_path, dpi=300)
        print(f"\nOptimization history saved to {output_path}")
    except Exception as e:
        print(f"Could not save visualization: {e}")
