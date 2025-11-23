import sys
import os
import torch
import numpy as np

import torch
import numpy as np

try:
    from catenets.models.torch import DragonNet
    from catenets.datasets import load as load_dataset
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback to local import if installed in editable mode but not picked up
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'external', 'CATENets'))
    try:
        from catenets.models.torch import DragonNet
        from catenets.datasets import load as load_dataset
    except ImportError as e2:
        print(f"Could not import catenets even after path fix: {e2}")
        sys.exit(1)

def test_dragonnet():
    print("1. Loading IHDP dataset...")
    # This might download data
    try:
        # load returns: X, w, y, potential_outcomes_train, X_test, potential_outcomes_test
        X_train, w_train, y_train, _, X_test, po_test = load_dataset("ihdp", train_ratio=0.8)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Data loaded. Train shape: {X_train.shape}")
    print(f"Unique treatment values (w_train): {np.unique(w_train)}")
    print(f"w_train shape: {w_train.shape}")

    # 2. Initialize DragonNet
    print("2. Initializing DragonNet...")
    model = DragonNet(
        n_unit_in=X_train.shape[1],
        n_iter=2000, 
        batch_size=64, 
        lr=1e-3, 
        n_units_out_prop=100, 
        n_units_out=100,
        val_split_prop=0.0
    )

    # 3. Train
    print("3. Training DragonNet...")
    model.fit(X_train, y_train, w_train)

    # 4. Predict CATE
    print("4. Predicting CATE...")
    cate_pred = model.predict(X_test)
    
    # 5. Calculate PEHE (Precision in Estimation of Heterogeneous Effect)
    # po_test is (N, 2) where [:, 0] is mu0 and [:, 1] is mu1
    mu0_test = po_test[:, 0]
    mu1_test = po_test[:, 1]
    cate_true = mu1_test - mu0_test
    
    # cate_pred is tensor, convert to numpy
    if isinstance(cate_pred, torch.Tensor):
        cate_pred = cate_pred.detach().cpu().numpy().flatten()
        
    pehe = np.mean((cate_true - cate_pred) ** 2)
    print("-" * 30)
    print(f"Baseline Performance (DragonNet on IHDP)")
    print(f"PEHE Score: {pehe:.6f}")
    print("-" * 30)

    print(f"Prediction complete. First 5 CATE predictions: {cate_pred[:5]}")
    print(f"First 5 True CATE: {cate_true[:5]}")
    print("Success! CATENets is working.")

if __name__ == "__main__":
    test_dragonnet()
