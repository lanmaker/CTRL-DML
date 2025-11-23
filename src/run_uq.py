import numpy as np
import torch
import matplotlib.pyplot as plt
from my_dragonnet import MyDragonNet as DragonNet
from sklearn.model_selection import train_test_split
import os

# === 1. Generate Data (Stress Data) ===
def get_data(n=1000):
    # Simple generation logic matching previous experiments
    C = np.random.normal(0, 1, size=(n, 5)) 
    I = np.random.normal(0, 1, size=(n, 5))
    N = np.random.normal(0, 1, size=(n, 20)) # 20 dim noise
    X = np.concatenate([C, I, N], axis=1)
    
    # True CATE
    true_te = 2 * np.sin(C[:, 0] * np.pi) + np.maximum(0, C[:, 1])
    
    # Generate T and Y
    logit = np.sum(C[:, :2], axis=1) + np.sum(I, axis=1)
    prop = 1 / (1 + np.exp(-logit))
    T = np.random.binomial(1, prop)
    y = np.sum(C, axis=1) + true_te * T + np.random.normal(0, 0.5, n)
    
    return X, y, T, true_te

# Use a reasonable sample size
N_SAMPLES = 2000
X, y, T, true_te = get_data(n=N_SAMPLES)
X_train, X_test, y_train, y_test, T_train, T_test, te_train, te_test = train_test_split(
    X, y, T, true_te, test_size=0.2, random_state=42
)

# === 2. Train Model (with dropout_prob) ===
print("Training CTRL-DML with Dropout...")
# Tuning for better coverage:
# - n_units_r=400: Wider model to capture more uncertainty
# - dropout_prob=0.3: Higher dropout for wider confidence intervals
# - n_iter=1200: Slightly reduced iterations to prevent overfitting
model = DragonNet(
    n_unit_in=X.shape[1], 
    n_units_r=800, 
    dropout_prob=0.6, 
    n_iter=200, 
    batch_size=64
)
model.fit(X_train, y_train, T_train)

# === 3. Core: Uncertainty Inference ===
print("Running MC Dropout Inference...")
# Predict mean, std, lower bound, upper bound
pred_mean, pred_std, lower, upper = model.predict_with_uncertainty(X_test, n_runs=100)

# === 4. Visualization (Sorted CATE Plot) ===
# Sort by True CATE for better visualization
sort_idx = np.argsort(te_test)
te_sorted = te_test[sort_idx]
mean_sorted = pred_mean[sort_idx]
lower_sorted = lower[sort_idx]
upper_sorted = upper[sort_idx]

# === Plotting Optimization ===
plt.figure(figsize=(12, 6))

# Only plot the first 100 samples for clarity
limit = 100 
subset_te = te_sorted[:limit]
subset_mean = mean_sorted[:limit]
subset_lower = lower_sorted[:limit]
subset_upper = upper_sorted[:limit]

# 1. Ground Truth (Thicker line)
plt.plot(subset_te, color='black', linestyle='--', linewidth=2.5, label='Ground Truth CATE')

# 2. Prediction (Add transparency)
plt.plot(subset_mean, color='#2ca02c', alpha=0.9, linewidth=1.5, label='CTRL-DML Prediction')

# 3. Confidence Interval (Darker color)
plt.fill_between(range(limit), subset_lower, subset_upper, 
                 color='#2ca02c', alpha=0.3, label='95% Confidence Interval')

plt.xlabel("Test Samples (Sorted by Effect)")
plt.ylabel("Causal Effect (CATE)")
plt.title(f"Uncertainty Quantification (Dropout=0.3) | Subset of {limit} samples")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

output_path = "uq_analysis_fixed.pdf"
plt.savefig(output_path, dpi=300)
print(f"Visualization complete! Saved to {output_path}")

# === 5. Coverage Metric ===
# A good uncertainty estimate should have ~95% coverage for 95% CI
inside_interval = (te_test >= lower) & (te_test <= upper)
coverage_rate = np.mean(inside_interval)
print(f"\n>>> Metric Evaluation:")
print(f"Coverage Rate (Target ~0.95): {coverage_rate:.4f}")
print(f"Mean Uncertainty (Avg Std): {np.mean(pred_std):.4f}")

# Interpretation
if coverage_rate < 0.8:
    print("Warning: Model is Over-confident (Coverage < 0.8). Consider increasing dropout_prob.")
elif coverage_rate > 0.99:
    print("Warning: Model is Under-confident (Coverage > 0.99). Consider decreasing dropout_prob.")
else:
    print("Success: Coverage is within a reasonable range!")
