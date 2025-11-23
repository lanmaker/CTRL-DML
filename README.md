# CTRL-DML: Robust & Multimodal Causal Effect Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> A deep learning estimator that learns to ignore noise and understand text, combining tabular attention, sparsity regularization, and uncertainty quantification.

---

## Story Arc

**Conflict.** Deep nets like DragonNet often overfit noise in causal inference, while tree models struggle with unstructured text.  
**Solution.** CTRL-DML adds sparse attention to “learn to stay quiet” on junk features and uses a text tower for embeddings.  
**Evidence.**  
- White-box: feature-importance shows signal vs. noise separation.  
- Robustness: holds up under high-dimensional noise.  
- Scaling law: once N > 4k, CTRL-DML closes the gap with Causal Forest.  
**Killer features.** Multimodal text + tabular cuts error by ~62% in text-heavy settings; MC Dropout provides calibrated uncertainty.

---

## Key Results

- **Interpretability:** Feature gating suppresses noise (see `feature_importance_sparse.pdf`).
- **Robustness:** Stable PEHE as noise dimensions grow (see `robustness_solid.pdf`).
- **Scaling law:** Deep model overtakes trees at larger N (see `scaling_results.pdf` and `scaling_delta.pdf`).
- **Multimodal:** Text-aware CTRL-DML beats TF-IDF trees by ~62% (see `multimodal_result.pdf`).
- **Reliability:** MC Dropout intervals for risk-aware decisions (see `uq_analysis_fixed.pdf`).

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

### Benchmarks

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

- **Scaling law (sample size sweep):**
  ```bash
  # Fast sanity run
  CTRL_DML_FAST=1 python src/run_scaling.py
  # Full run
  python src/run_scaling.py
  ```
  Outputs: `scaling_results.pdf`, `scaling_delta.pdf`.

- **Multimodal text + tabular:**
  ```bash
  python src/run_multimodal_benchmark.py
  ```
  Outputs: `multimodal_result.pdf`.

- **Uncertainty quantification (MC Dropout):**
  ```bash
  python src/run_uq.py
  ```
  Outputs: `uq_analysis_fixed.pdf`.

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
│   ├── model_multimodal.py    # Two-tower multimodal architecture
│   ├── data_multimodal.py     # Text + tabular data generators
│   ├── run_benchmark_final.py # Noise benchmark vs DragonNet & Causal Forest
│   ├── run_scaling.py         # Scaling law benchmark
│   ├── run_robustness_solid.py# Robustness to high-dimensional noise
│   ├── run_multimodal_benchmark.py
│   └── run_uq.py              # Uncertainty quantification
├── requirements.txt
└── README.md
```

---

## Citation & Contact

This is an independent research effort on robust and multimodal causal inference. If it helps your work, please star the repo. For questions, open an issue or reach out via GitHub. 
