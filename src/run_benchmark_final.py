import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim
from torch.nn import functional as F
from scipy.special import expit
import os
import argparse
from pathlib import Path

# === New challenger: EconML ===
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV

# === Existing baselines ===
from catenets.models.torch import DragonNet as StandardDragonNet
from my_dragonnet import MyDragonNet as CTRLDML
from catenets.models.torch.base import DEVICE

# === Dataset Loaders ===
from catenets.datasets.dataset_ihdp import load as load_ihdp
from catenets.datasets.dataset_acic2016 import load as load_acic
from catenets.datasets.dataset_twins import load as load_twins

FAST_RUN = os.environ.get("CTRL_DML_FAST", "0") == "1"
CTRL_EPOCHS = 800 if not FAST_RUN else 150
DRAGON_ITERS = 1000 if not FAST_RUN else 150
CF_TREES = 100 if not FAST_RUN else 48
N_SAMPLES = 2000 if not FAST_RUN else 1000

# === Helper functions ===
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_stress_data_dynamic(n_samples=2000, n_noise=10):
    # Keep the data-generation logic identical
    C = np.random.normal(0, 1, size=(n_samples, 5)) 
    I = np.random.normal(0, 1, size=(n_samples, 5)) 
    N = np.random.normal(0, 1, size=(n_samples, n_noise))
    X = np.concatenate([C, I, N], axis=1)
    
    logit = (np.sum(C, axis=1) + 0.5 * C[:, 0] * C[:, 1] + 2 * np.sum(I, axis=1))
    propensity = expit(logit)
    T = np.random.binomial(1, propensity)
    
    true_te = 2 * np.sin(C[:, 0] * np.pi) + np.maximum(0, C[:, 1])
    y0 = np.sum(C, axis=1) ** 2 + np.random.normal(0, 0.5, size=n_samples)
    y = y0 + true_te * T
    return X, T, y, true_te

# === Train CTRL-DML (Ours) ===
def train_ctrl_dml(X_np, y_np, T_np):
    X = torch.from_numpy(X_np).float().to(DEVICE)
    y = torch.from_numpy(y_np).float().to(DEVICE)
    T = torch.from_numpy(T_np).long().to(DEVICE).squeeze()
    
    model = CTRLDML(n_unit_in=X.shape[1], n_iter=1, batch_size=256, dropout_prob=0.35).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    lambda_sparsity = 0.072
    
    model.train()
    for epoch in range(CTRL_EPOCHS):
        permutation = torch.randperm(X.shape[0])
        for i in range(0, X.shape[0], 256):
            indices = permutation[i : i + 256]
            batch_X, batch_y, batch_T = X[indices], y[indices], T[indices]
            optimizer.zero_grad()
            po_preds, prop_preds, discr = model._step(batch_X, batch_T)
            base_loss = model.loss(po_preds, prop_preds, batch_y, batch_T, discr)
            reg_loss = model._repr_estimator[0].current_mask_penalty
            total_loss = base_loss + (lambda_sparsity * reg_loss)
            total_loss.backward()
            optimizer.step()
    return model

# === Benchmarks ===

def run_noise_robustness_benchmark(args):
    noise_levels = [10, 50, 100] 
    all_seeds = [42, 1024, 2023, 7, 999]
    seeds = all_seeds if not FAST_RUN else all_seeds[:2]

    # Results matrix: [Noise_Level, Seed]
    res_dragon = np.zeros((len(noise_levels), len(seeds)))
    res_cf = np.zeros((len(noise_levels), len(seeds))) # Added: Causal Forest
    res_ours = np.zeros((len(noise_levels), len(seeds)))

    print("Starting final benchmark: DragonNet vs CausalForest vs CTRL-DML (Noise Robustness)")
    if FAST_RUN:
        print("Fast mode enabled (CTRL_DML_FAST=1): fewer epochs/trees, smaller sample size, and fewer seeds for quicker verification.")

    for i, n_noise in enumerate(noise_levels):
        print(f"\n>>> Noise level: {n_noise}")
        for j, seed in enumerate(seeds):
            set_seed(seed)
            X, T, y, true_te = get_stress_data_dynamic(n_samples=N_SAMPLES, n_noise=n_noise)
            
            # 1. DragonNet (Red)
            m_base = StandardDragonNet(n_unit_in=X.shape[1], n_iter=DRAGON_ITERS, batch_size=256)
            m_base.fit(X, y, T)
            p_base_tensor = m_base.predict(torch.from_numpy(X).float().to(DEVICE))
            p_base = p_base_tensor.cpu().detach().numpy().flatten()
            res_dragon[i, j] = np.sqrt(np.mean((true_te - p_base)**2))
            
            # 2. Causal Forest (Blue) - The Strong Baseline
            # Use LassoCV for outcome and LogisticRegressionCV for treatment to avoid classifier warning
            est = CausalForestDML(model_y=LassoCV(), model_t=LogisticRegressionCV(max_iter=1000), 
                                  n_estimators=CF_TREES, discrete_treatment=True, random_state=seed)
            est.fit(y, T, X=X)
            p_cf = est.effect(X)
            res_cf[i, j] = np.sqrt(np.mean((true_te - p_cf.flatten())**2))
            
            # 3. CTRL-DML (Green)
            m_ours = train_ctrl_dml(X, y, T)
            m_ours.eval()
            with torch.no_grad():
                pred_ours_tensor = m_ours.predict(torch.from_numpy(X).float().to(DEVICE))
                p_ours = pred_ours_tensor.cpu().detach().numpy().flatten()
            res_ours[i, j] = np.sqrt(np.mean((true_te - p_ours)**2))
            
            print(f"   Seed {seed} | Dragon: {res_dragon[i,j]:.2f} | CF: {res_cf[i,j]:.2f} | Ours: {res_ours[i,j]:.2f}")

    # === Plotting ===
    d_mean, d_std = np.mean(res_dragon, axis=1), np.std(res_dragon, axis=1)
    c_mean, c_std = np.mean(res_cf, axis=1), np.std(res_cf, axis=1)
    o_mean, o_std = np.mean(res_ours, axis=1), np.std(res_ours, axis=1)
    offsets = np.linspace(-0.15, 0.15, len(seeds))

    plt.figure(figsize=(10, 6))

    # DragonNet (Red)
    plt.plot(noise_levels, d_mean, label='DragonNet (NN Baseline)', color='#d62728', linestyle='--', marker='x')
    plt.fill_between(noise_levels, d_mean - d_std, d_mean + d_std, color='#d62728', alpha=0.1)
    for idx, _ in enumerate(seeds):
        plt.scatter(np.array(noise_levels) + offsets[idx], res_dragon[:, idx], color='#d62728', alpha=0.35, s=40, label='_nolegend_')

    # Causal Forest (Blue)
    plt.plot(noise_levels, c_mean, label='Causal Forest (Tree Baseline)', color='#1f77b4', linestyle='--', marker='^')
    plt.fill_between(noise_levels, c_mean - c_std, c_mean + c_std, color='#1f77b4', alpha=0.1)
    for idx, _ in enumerate(seeds):
        plt.scatter(np.array(noise_levels) + offsets[idx], res_cf[:, idx], color='#1f77b4', alpha=0.35, s=40, label='_nolegend_')

    # CTRL-DML (Green)
    plt.plot(noise_levels, o_mean, label='CTRL-DML (Ours)', color='#2ca02c', linestyle='-', marker='s', linewidth=2.5)
    plt.fill_between(noise_levels, o_mean - o_std, o_mean + o_std, color='#2ca02c', alpha=0.2)
    for idx, _ in enumerate(seeds):
        plt.scatter(np.array(noise_levels) + offsets[idx], res_ours[:, idx], color='#2ca02c', alpha=0.35, s=40, label='_nolegend_')

    plt.xlabel("Noise Dimensions")
    plt.ylabel("PEHE Error")
    plt.title("Final Benchmark: Performance Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = "benchmark_final.pdf"
    plt.savefig(output_path, dpi=300)
    print(f"\nResults saved to {output_path}")

    # Delta view: negative means CTRL-DML beats Causal Forest
    delta = res_ours - res_cf
    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)

    plt.figure(figsize=(8, 5))
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.plot(noise_levels, delta_mean, label='CTRL-DML minus Causal Forest', color='#9467bd', marker='o', linewidth=2)
    plt.fill_between(noise_levels, delta_mean - delta_std, delta_mean + delta_std, color='#9467bd', alpha=0.15)
    for idx, _ in enumerate(seeds):
        plt.scatter(np.array(noise_levels) + offsets[idx], delta[:, idx], color='#9467bd', alpha=0.5, s=45, label='_nolegend_')
    plt.xlabel("Noise Dimensions")
    plt.ylabel("PEHE Delta (negative favors CTRL-DML)")
    plt.title("Gap vs. Causal Forest across Noise Levels")
    plt.legend()
    plt.grid(True, alpha=0.3)
    delta_path = "benchmark_final_delta.pdf"
    plt.savefig(delta_path, dpi=300)
    print(f"Delta view saved to {delta_path}")

    print("\nInterpretation guide:")
    print("- Noise=10: Causal Forest may lead thanks to strong nonlinear splits with low noise.")
    print("- Noise=100: CTRL-DML should stay competitive via sparsity when signal is sparse.")

def run_ihdp_benchmark(args):
    print("\n>>> Running IHDP Benchmark")
    n_exp = 10 if FAST_RUN else 100
    mses = []
    
    for i in range(1, n_exp + 1):
        try:
            # load() returns: X, w, y, pot_y, X_test, pot_y_test
            # pot_y is (n, 2), true CATE is pot_y[:,1] - pot_y[:,0]
            X, T, y, _, X_test, pot_y_test = load_ihdp(args.data_path, exp=i)
            true_te_test = pot_y_test[:, 1] - pot_y_test[:, 0]
            
            # Using only standard training set to keep it simple, evaluating on test set
            m_ours = train_ctrl_dml(X, y, T)
            m_ours.eval()
            with torch.no_grad():
                pred_ours_tensor = m_ours.predict(torch.from_numpy(X_test).float().to(DEVICE))
                p_ours = pred_ours_tensor.cpu().detach().numpy().flatten()
            
            pehe = np.sqrt(np.mean((true_te_test - p_ours)**2))
            mses.append(pehe)
            if i % 10 == 0 or i == 1:
                print(f"Exp {i}/{n_exp}: PEHE = {pehe:.4f}")
        except Exception as e:
            print(f"Error in Exp {i}: {e}")

    print(f"IHDP Result: Mean PEHE = {np.mean(mses):.4f} +/- {np.std(mses):.4f}")

def run_acic_benchmark(args):
    print("\n>>> Running ACIC2016 Benchmark")
    # ACIC has many simulations. We'll pick a few characteristic ones if not specified.
    # Currently just running simulation 1 for demonstration if not iterating all.
    # ACIC simulation files are complicated, using 'load' defaults to downloading preprocessed.
    
    # We will loop through 10 simulations for robustness check
    n_sims = 5 if FAST_RUN else 20
    mses_ours = []
    
    for i in range(1, n_sims + 1):
        try:
            # load() returns train/test splits derived from simulation
            # Note: setting 'exp' might be simulated by seeding in preprocess_simu within load?
            # Actually catenets ACIC loader is a bit complex. 
            # We will use 'preprocess_simu' mode (default) with different seeds (i)
            # load(data_path, i_exp=i) is not standard signature for acic loader in this repo, 
            # checking signature... it uses kwargs passed to preprocess.
            # preprocess_simu takes 'i_exp'.
            
            # Signature: load(data_path, preprocessed=True, original_acic_outcomes=False, i_exp=...)
            # Returns: X, w, y, pot_y, X_test, pot_y_test
            X, T, y, _, X_test, pot_y_test = load_acic(args.data_path, i_exp=i)
            true_te_test = pot_y_test[:, 1] - pot_y_test[:, 0]
            
            m_ours = train_ctrl_dml(X, y, T)
            m_ours.eval()
            with torch.no_grad():
                pred_ours_tensor = m_ours.predict(torch.from_numpy(X_test).float().to(DEVICE))
                p_ours = pred_ours_tensor.cpu().detach().numpy().flatten()
            
            pehe = np.sqrt(np.mean((true_te_test - p_ours)**2))
            mses_ours.append(pehe)
            print(f"Sim {i}/{n_sims}: PEHE = {pehe:.4f}")

        except Exception as e:
            print(f"Error in Sim {i}: {e}")
            
    print(f"ACIC2016 Result: Mean PEHE = {np.mean(mses_ours):.4f} +/- {np.std(mses_ours):.4f}")

def run_twins_benchmark(args):
    print("\n>>> Running Twins Benchmark")
    # Twins is one large dataset. We can bootstrap or just run one train/test split.
    # load() returns X_train, w_train, y_train, pot_y_train, X_test, pot_y_test
    # Default train_ratio=0.8
    seeds = [42, 1024, 7] if not FAST_RUN else [42]
    mses = []
    
    for s in seeds:
        X, T, y, _, X_test, pot_y_test = load_twins(args.data_path, seed=s, train_ratio=0.8)
        true_te_test = pot_y_test[:, 1] - pot_y_test[:, 0]
        
        m_ours = train_ctrl_dml(X, y, T)
        m_ours.eval()
        with torch.no_grad():
            pred_ours_tensor = m_ours.predict(torch.from_numpy(X_test).float().to(DEVICE))
            p_ours = pred_ours_tensor.cpu().detach().numpy().flatten()
        
        pehe = np.sqrt(np.mean((true_te_test - p_ours)**2))
        mses.append(pehe)
        print(f"Seed {s}: PEHE = {pehe:.4f}")
        
    print(f"Twins Result: Mean PEHE = {np.mean(mses):.4f} +/- {np.std(mses):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTRL-DML Final Benchmark Suite")
    parser.add_argument("--benchmark", type=str, choices=["noise", "ihdp", "acic", "twins"], default="noise", help="Which benchmark to run")
    parser.add_argument("--data-path", type=str, default="./data", help="Path to store/load datasets")
    parser.add_argument("--fast-run", action="store_true", help="Enable fast run for debugging")
    
    args = parser.parse_args()
    
    # Override global FAST_RUN if explicit argument is passed, ensuring consistency
    if args.fast_run:
        FAST_RUN = True
        # Re-set constants if needed (though Python executed them at top-level already, 
        # so this is mostly for local scope logic if we re-used them inside functions.
        # Since we use global vars inside functions, we should ideally not rely on the top-level
        # ones mutating, but for this script structure it's okay as long as functions read the global.
        # Better safety: Update global variables
        CTRL_EPOCHS = 150
        DRAGON_ITERS = 150
        CF_TREES = 48
        N_SAMPLES = 1000
    
    # Ensure data path exists
    data_path = Path(args.data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    
    if args.benchmark == "noise":
        run_noise_robustness_benchmark(args)
    elif args.benchmark == "ihdp":
        # Pass Path object, not string
        args.data_path = data_path
        run_ihdp_benchmark(args)
    elif args.benchmark == "acic":
        args.data_path = data_path
        run_acic_benchmark(args)
    elif args.benchmark == "twins":
        args.data_path = data_path
        run_twins_benchmark(args)
