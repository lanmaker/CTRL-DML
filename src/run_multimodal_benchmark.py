import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from data_multimodal import get_multimodal_data, convert_text_to_tfidf
from model_multimodal import MultimodalCTRL

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. Prepare Data
print("Generating Multimodal Data (Clinical Notes Simulation)...")
n_samples = 3000
vocab_size = 1000
X_tab, X_text, Y, T, true_te = get_multimodal_data(n=n_samples, vocab_size=vocab_size)

# 2. Challenger 1: Causal Forest (Baseline)
# Must manually convert text to TF-IDF vectors and concatenate
print("Running Causal Forest (Baseline)...")
X_tfidf = convert_text_to_tfidf(X_text, vocab_size)
X_combined = np.concatenate([X_tab, X_tfidf], axis=1) # Dimension explosion: 10 + 1000

# Train CF
est = CausalForestDML(
    model_y=LassoCV(), model_t=LogisticRegressionCV(max_iter=1000), 
    n_estimators=100, discrete_treatment=True, random_state=42
)
est.fit(Y, T, X=X_combined)
pred_cf = est.effect(X_combined)
pehe_cf = np.sqrt(np.mean((true_te - pred_cf.flatten())**2))
print(f"Causal Forest PEHE: {pehe_cf:.4f}")


# 3. Challenger 2: Multimodal CTRL-DML (Ours)
print("Running Multimodal CTRL-DML (Ours)...")

# Convert to Tensor
xt_tab = torch.from_numpy(X_tab).float()
xt_text = torch.from_numpy(X_text).long() # Must be long for indices
yt = torch.from_numpy(Y).float().unsqueeze(1)
tt = torch.from_numpy(T).float().unsqueeze(1)

model = MultimodalCTRL(n_tab_input=10, vocab_size=vocab_size, embed_dim=32)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4) # Reduced LR, added decay

# Simple training loop
for epoch in range(150): # Reduced epochs to prevent overfitting
    optimizer.zero_grad()
    y0, y1, t_prob = model(xt_tab, xt_text)
    
    loss_t = F.binary_cross_entropy(t_prob, tt)
    y_pred = tt * y1 + (1 - tt) * y0
    loss_y = F.mse_loss(y_pred, yt)
    
    loss = loss_y + loss_t
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} loss: {loss.item():.4f}")

# Predict
model.eval()
with torch.no_grad():
    y0, y1, _ = model(xt_tab, xt_text)
    pred_ours = (y1 - y0).numpy()

pehe_ours = np.sqrt(np.mean((true_te - pred_ours.flatten())**2))
print(f"Multimodal CTRL-DML PEHE: {pehe_ours:.4f}")

# 4. Plotting
plt.figure(figsize=(8, 5))
models = ['Causal Forest\n(TF-IDF)', 'Multimodal\nCTRL-DML']
scores = [pehe_cf, pehe_ours]
colors = ['#1f77b4', '#2ca02c']

plt.bar(models, scores, color=colors, alpha=0.8)
plt.ylabel("PEHE Error (Lower is Better)")
plt.title("Performance on Text-Confounded Data")
plt.ylim(0, max(scores) * 1.2) # Add headroom for text labels
for i, v in enumerate(scores):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center', fontweight='bold')

output_path = "multimodal_result.pdf"
plt.savefig(output_path)
print(f"Done! Check {output_path}")
