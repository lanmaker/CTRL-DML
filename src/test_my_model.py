from my_dragonnet import MyDragonNet
import numpy as np
import torch

def test_my_model():
    # 1. Load stress data
    print("Loading stress_data.npz...")
    data = np.load("stress_data.npz")
    X, T, y, true_te = data['X'], data['T'], data['y'], data['true_te']
    
    # 2. Initialize CTRL-DML Model
    print("Initializing CTRL-DML (MyDragonNet)...")
    model = MyDragonNet(
        n_unit_in=X.shape[1], 
        n_iter=2000, 
        batch_size=256,
        lr=1e-3,
        val_split_prop=0.0 # Disable validation split
    )

    print("Surgery successful! Model initialized. Training CTRL-DML...")
    model.fit(X, y, T)

    # 3. Evaluate
    print("Predicting CATE...")
    pred = model.predict(X)
    
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy().flatten()
        
    pehe = np.sqrt(np.mean((true_te - pred)**2))
    print("-" * 30)
    print(f"CTRL-DML Performance on Stress Data")
    print(f"PEHE Score: {pehe:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    test_my_model()
