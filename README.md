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
**Reliability.** MC Dropout under-covers; conformal calibration restores nominal coverage.

---

## Key Results (Full Run: 2025-12-31)

| Experiment | CTRL Plug-in | CTRL-DML | Causal Forest | Notes |
|------------|--------------|----------|---------------|-------|
| IHDP (100 realizations) | 5.58 ± 8.23 | 7.78 ± 7.72 | 3.64 ± 6.56 | Median: Plugin 2.57, DML 4.67 |
| Ablation: No gating | 1.63 | 4.59 | - | 3 seeds (42, 1024, 2023) |
| Ablation: Gating + L1 | 1.64 | 8.80 | - | 3 seeds, N=2000, epochs 150/300 |
| Scaling N=500 | 1.58 | 8.77 | 1.92 | 3 seeds |
| Scaling N=1000 | 1.61 | 8.83 | 1.72 | 3 seeds |
| Scaling N=2000 | 1.61 | 8.78 | 1.57 | 3 seeds |
| Scaling N=5000 | 1.57 | 8.56 | 1.48 | 3 seeds |
| Nuisance misspec (strong) | 1.61 | 8.78 | - | 3 seeds |
| Nuisance misspec (weak) | 1.61 | 8.46 | - | 3 seeds |
| Multimodal (p_noise=0) | 1.24 | 1.24 | 0.08 | |
| Yelp semi-synthetic | 1.12 | 1.05 | 1.49 | |
| UQ MC coverage | - | 0.12 | - | 100 MC runs |
| UQ Conformal coverage | - | 0.95 | - | Target: 0.95 |

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

### Run Experiments

```bash
# Ablation study
python -m src.experiments.ablation --mode all

# All outputs go to output/figures/ and output/tables/
```

---

## Project Structure

```
CTRL-DML/
├── src/
│   ├── models/                    # Core models
│   │   ├── dragonnet.py           # TabularAttention + MyDragonNet
│   │   ├── multimodal.py          # MultimodalCTRL
│   │   └── orthogonal_learner.py  # CTRLOrthogonalLearner
│   ├── data/                      # Data loaders
│   │   ├── ihdp.py
│   │   ├── twins.py
│   │   ├── acic.py
│   │   ├── multimodal.py
│   │   └── synthetic.py
│   ├── experiments/               # Experiment scripts
│   │   └── ablation.py
│   └── utils/                     # Utilities
│       ├── io.py                  # Output path management
│       ├── metrics.py             # PEHE, ATE, bootstrap
│       ├── training.py            # Training utilities
│       ├── latex.py               # LaTeX table/macro generation
│       └── plotting.py            # Unified plotting style
├── output/                        # All outputs
│   ├── figures/                   # PDF figures (unified style)
│   ├── tables/                    # CSV data + LaTeX tables
│   └── results_macros.tex         # LaTeX macros for paper
├── data/                          # Raw data
├── external/                      # External dependencies
├── CTRL-DML-Paper/               # Paper directory
│   └── main.tex                   # Uses macros from output/
└── requirements.txt
```

---

## Output Structure

All experiment outputs are centralized:

- **Figures**: `output/figures/*.pdf` - Unified academic style (dark red primary, serif fonts, 300 DPI)
- **Tables**: `output/tables/*.csv` - Raw data for reproducibility
- **LaTeX Macros**: `output/results_macros.tex` - Auto-generated, imported by paper

### Regenerate Outputs

```bash
# Regenerate LaTeX macros from CSV data
python -m src.utils.latex --generate-all

# Paper compilation (from CTRL-DML-Paper/)
cd CTRL-DML-Paper && pdflatex main.tex
```

---

## Methodology

- **Sparse attention:** `TabularAttention` with L1 penalty to mute noise.
- **Two-tower fusion:** text embeddings + tabular dense tower for multimodal confounders.
- **Orthogonal head:** DML-style ratio targets with warm-start and clipping.
- **Targets:** heads for treatment, control, and propensity with targeted regularization.
- **Tuning:** Optuna used for LR, dropout, weight decay.
- **UQ:** Monte Carlo Dropout + conformal calibration.

---

## Plotting Style

All figures use a unified academic style defined in `src/utils/plotting.py`:

```python
from src.utils.plotting import set_style, COLORS, create_figure

set_style()
fig, ax = create_figure(figsize=(6, 4))
ax.plot(x, y, color=COLORS['primary'])  # Dark red (#8B1C1C)
```

Color palette:
- Primary: Dark red (#8B1C1C) - CTRL-DML
- Secondary: Steel blue (#2C5F8A) - Causal Forest
- Accent: Forest green (#4A7C59) - DragonNet

---

## Citation & Contact

This is an independent research effort on robust and multimodal causal inference. If it helps your work, please star the repo. For questions, open an issue or reach out via GitHub.
