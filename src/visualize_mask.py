import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path to import my_dragonnet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add external/CATENets to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'external', 'CATENets'))

from my_dragonnet import MyDragonNet # Import the modified model

def visualize_mask():
    # 1. Reload Data
    print("Loading stress_data.npz...")
    data = np.load("stress_data.npz")
    X = data['X']
    T = data['T']
    y = data['y']
    
    feature_names = [f"C{i}" for i in range(5)] + \
                    [f"I{i}" for i in range(5)] + \
                    [f"N{i}" for i in range(X.shape[1]-10)]

    # 2. Re-initialize and Train Model
    print("Training CTRL-DML for visualization (1000 iterations)...")
    model = MyDragonNet(
        n_unit_in=X.shape[1], 
        n_iter=1000, 
        batch_size=256,
        lr=1e-3,
        val_split_prop=0.0
    )
    model.fit(X, y, T)

    # === Core: Extract Mask Weights ===
    # Our TabularAttention is the first layer of _repr_estimator
    # _repr_estimator is a Sequential: [TabularAttention, Linear, ELU]
    attention_layer = model._repr_estimator[0] 

    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float()
        # mask_net outputs Sigmoid values (0-1)
        masks = attention_layer.mask_net(X_tensor).numpy()

    # 3. Calculate Mean Importance
    mean_importance = np.mean(masks, axis=0)

    # 4. Plotting
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(feature_names))

    # Colors: Confounder(Green), Instrument(Red), Noise(Gray)
    colors = ['green']*5 + ['red']*5 + ['gray']*(len(feature_names)-10)

    plt.bar(x_pos, mean_importance, color=colors)
    plt.xticks(x_pos, feature_names, rotation=90)
    plt.ylabel("Attention Weight (Importance)")
    plt.title("What did CTRL-DML learn? (Green=Confounder, Red=Instrument, Gray=Noise)")

    # Add vertical lines to separate groups
    plt.axvline(x=4.5, color='black', linestyle='--')
    plt.axvline(x=9.5, color='black', linestyle='--')

    plt.tight_layout()
    output_path = "feature_importance.pdf"
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")
    
    # 5. Numerical Verification
    avg_c = np.mean(mean_importance[:5])
    avg_i = np.mean(mean_importance[5:10])
    avg_n = np.mean(mean_importance[10:])
    
    print("-" * 30)
    print("Feature Importance Analysis:")
    print(f"Confounders (Green) Avg Weight: {avg_c:.4f}")
    print(f"Instruments (Red)   Avg Weight: {avg_i:.4f}")
    print(f"Noise (Gray)        Avg Weight: {avg_n:.4f}")
    print("-" * 30)
    
    if avg_c > avg_i and avg_c > avg_n:
        print("Hypothesis CONFIRMED: Confounders have the highest attention.")
    else:
        print("Hypothesis WEAK: Model might be overfitting to Instruments or Noise.")

    if avg_n < 0.1:
         print("Noise Filtering: EXCELLENT (Noise weights are very low)")
    elif avg_n < 0.3:
         print("Noise Filtering: GOOD")
    else:
         print("Noise Filtering: POOR")

if __name__ == "__main__":
    visualize_mask()
