# CTRL-DML: Robust & Multimodal Causal Effect Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> A deep learning estimator that learns to ignore noise and understand text, combining tabular attention, sparsity regularization, and uncertainty quantification.

---

## Story Arc

**Conflict.** Deep nets like DragonNet overfit noise; trees ignore unstructured text; DML can be unstable when nuisances are weak.  
**Solution.** CTRL-DML makes DML a first-class citizen: sparse tabular attention, modular nuisances, and an orthogonal head (ratio targets + warm-start + clipping) with optional distillation.  
**Claim.** In high-dimensional/weak-nuisance settings, the orthogonal head + sparsity yields more stable CATE than plug-in baselines.  
**Evidence.**  
- White-box: feature-role plots separate confounders/instruments/noise.  
- Robustness: stable PEHE as nuisance dimensions grow; orthogonal head degrades more gracefully when nuisances are weakened.  
- Scaling: with noise=50, CTRL-DML stays ahead of CF as $N$ grows (multi-seed).  
- Multimodal: dense text+tabular (and cross-attn) beat TF-IDF forests; sweep results now include multiple seeds.  
- Public baselines: TWINS/ACIC loaders + baselines included; semi-synthetic Yelp text+tabular benchmark with ground-truth CATE.  
**Reliability.** MC Dropout under-covers; conformal calibration restores nominal coverage (see `uq_conformal.pdf`, `uq_metrics_multi.csv`).

---

## Key Results (Robustness Mainline)

- **IHDP:** CTRL-DML PEHE 1.81 ± 1.27 vs CF 3.84 ± 6.06 (`results_ihdp.txt`).
- **Scaling (noise=50):** N=500 CTRL-DML 1.56 ± 0.05 vs CF 2.21 ± 0.13; N=1000 CTRL-DML 1.53 ± 0.01 vs CF 2.00 ± 0.08; N=2000 CTRL-DML 1.53 ± 0.00 vs CF 1.94 ± 0.46 (`scaling_dml.csv`, `scaling_dml.pdf`).
- **Nuisance misspecification:** Orthogonal head stays competitive when nuisance signal weakens (`nuisance_misspec.csv`, `nuisance_misspec.pdf`).
- **Interpretability:** Feature gating suppresses noise (`feature_roles.pdf`).
- **Boundary conditions:** Causal Forest wins at low noise and remains strong at noise=100 in current benchmark (`results_noise.txt`); ACIC fast-run (3 sims) shows CF best, CTRL-DML mid (`benchmark_acic.pdf`, `results_acic.txt`).
- **Extensions:** Multimodal sweep now has multi-seed summary (n=2) (`multimodal_sweep_summary.csv`, `multimodal_sweep.pdf`); UQ now has a 2-seed summary (`uq_metrics_multi.csv`) but still limited.

---

## Quick Start

### Install
```bash
git clone https://github.com/lanmaker/CTRL-DML.git
cd CTRL-DML
pip install -r requirements.txt
# Optional: install bundled CATENets in editable mode
pip install -e external/CATENets
```

### Benchmarks (DML-ready)

- **Noise robustness (DragonNet vs CTRL-DML):**
  ```bash
  python src/run_robustness_solid.py
  ```
  Outputs: `robustness_solid.pdf`.

- **Final benchmark (DragonNet vs Causal Forest vs CTRL-DML):**
  ```bash
  # Fast sanity run
  CTRL_DML_FAST=1 python src/run_benchmark_final.py
  # Full run
  python src/run_benchmark_final.py
  ```
  Outputs: `benchmark_final.pdf`, `benchmark_final_delta.pdf`.

- **Scaling law (sample size sweep, plug-in vs DML vs CF):**
  ```bash
  python src/run_scaling_dml.py --sample-sizes 500 1000 2000
  ```
  Outputs: `scaling_dml.pdf`, `scaling_dml.csv`.

- **Multimodal text + tabular (plug-in vs DML vs TF-IDF CF):**
  ```bash
  python src/run_multimodal_dml.py --n 1500 --vocab-size 500 --p-noise 0.0
  ```
  Outputs: `multimodal_dml.pdf`, `multimodal_dml.csv`.

- **Semi-synthetic Yelp text+tabular (ground-truth CATE):**
  ```bash
  python src/run_yelp_semisynth.py --yelp-dir "Yelp JSON/yelp_dataset" --n-rows 2000
  ```
  Outputs: `yelp_semisynth.pdf`, `yelp_semisynth.csv`.

- **Uncertainty quantification (MC Dropout + conformal):**
  ```bash
  python src/run_uq.py --metrics-csv uq_metrics.csv
  ```
  Outputs: `uq_conformal.pdf`, `uq_metrics.csv`.
- **Nuisance misspecification (plug-in vs DML sensitivity):**
  ```bash
  python src/run_nuisance_misspec.py --n-samples 1000 --n-noise 50 --seeds 42 7
  ```
  Outputs: `nuisance_misspec.pdf`, `nuisance_misspec.csv`.

---

## Methodology (at a glance)

- **Sparse attention:** `TabularAttention` with L1 penalty to mute noise.  
- **Two-tower fusion:** text embeddings + tabular dense tower for multimodal confounders.  
- **Targets:** heads for treatment, control, and propensity with targeted regularization.  
- **Tuning:** Optuna used for LR, dropout, weight decay.  
- **UQ:** Monte Carlo Dropout for epistemic uncertainty.

---

## Project Structure

```
CTRL-DML/
├── external/CATENets/         # Baseline library (vendor)
├── src/
│   ├── my_dragonnet.py        # CTRL-DML base estimator with sparse attention
│   ├── ctrl_orthogonal_learner.py # High-level staged DML wrapper (warm start, cross-fit, orthogonal head)
│   ├── model_multimodal.py    # Two-tower multimodal architecture
│   ├── data_multimodal.py     # Text + tabular data generators
│   ├── run_scaling_dml.py     # CF vs plug-in vs orthogonal across N
│   ├── run_multimodal_dml.py  # Multimodal benchmark with orthogonal head
│   ├── run_yelp_semisynth.py  # Semi-synthetic Yelp text+tabular with known CATE
│   ├── run_nuisance_misspec.py# Nuisance degradation study (plug-in vs DML)
│   ├── run_benchmark_final.py # Noise benchmark vs DragonNet & Causal Forest
│   ├── run_scaling.py         # Scaling law benchmark
│   ├── run_robustness_solid.py# Robustness to high-dimensional noise
│   ├── run_multimodal_benchmark.py
│   └── run_uq.py              # Uncertainty quantification + metrics CSV
├── requirements.txt
└── README.md
```

---

## Citation & Contact

This is an independent research effort on robust and multimodal causal inference. If it helps your work, please star the repo. For questions, open an issue or reach out via GitHub. 
