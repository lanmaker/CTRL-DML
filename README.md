# CTRL-DML: Robust & Multimodal Causal Effect Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> A deep learning estimator that learns to ignore noise and understand text, combining tabular attention, sparsity regularization, and uncertainty quantification.

---

## Story Arc

**Conflict.** Deep nets like DragonNet overfit noise; trees ignore unstructured text; DML can be unstable when nuisances are weak.
**Solution.** CTRL-DML makes DML a first-class citizen: sparse tabular attention, modular nuisances, and an orthogonal head (ratio targets + warm-start + clipping) with optional distillation.
**Claim.** In high-dimensional/weak-nuisance settings, the orthogonal head + sparsity can yield more stable CATE than plug-in baselines.
**Evidence.**
- White-box: feature-role plots separate confounders/instruments/noise.
- Robustness: stable PEHE as nuisance dimensions grow; orthogonal head degrades more gracefully when nuisances are weakened.
- Scaling: compare CF vs CTRL-DML as $N$ grows (multi-seed).
- Multimodal: dense fusion, bag-of-embeddings, and cross-attention comparisons; sweep results include multiple seeds.
- Public baselines: TWINS/ACIC loaders + baselines included; semi-synthetic Yelp text+tabular benchmark with ground-truth CATE.
**Reliability.** MC Dropout under-covers; conformal calibration restores nominal coverage.

---

## Results & Reproducibility

All numbers and figures are generated from scripts. After running `./reproduce.sh`, see:
- `output/tables/` for raw CSV results.
- `output/figures/` for regenerated plots.
- `output/results_macros.tex` for LaTeX macros consumed by `CTRL-DML-Paper/main.tex`.

---

## Quick Start

### Install
```bash
git clone https://github.com/lanmaker/CTRL-DML.git
cd CTRL-DML
pip install -r requirements.txt
# Optional: install bundled catenets in editable mode
pip install -e external/catenets
```

### Run Experiments

```bash
# One-click reproduction (runs all experiments)
./reproduce.sh           # Full run (~hours)
./reproduce.sh --fast    # Fast mode for testing (~minutes)

# Or run individual experiments:
python -m src.experiments.ablation --mode all
python -m src.experiments.crossfit_ablation
python -m src.experiments.benchmarks --dataset all
python -m src.experiments.public_benchmarks
python -m src.experiments.feature_roles
python -m src.experiments.synthetic --analysis all
python -m src.experiments.robustness --test all
python -m src.experiments.multimodal --experiment all
python -m src.experiments.multimodal --experiment bert
python -m src.experiments.yelp_semisynth
python -m src.experiments.realdata --dataset all

# Regenerate LaTeX macros from results
python -m src.utils.latex --generate-macros

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
│   │   ├── ablation.py            # Core ablation study
│   │   ├── benchmarks.py          # IHDP/TWINS/ACIC benchmarks
│   │   ├── feature_roles.py       # Feature role decomposition
│   │   ├── synthetic.py           # Scaling, UQ, dynamics
│   │   ├── robustness.py          # Nuisance misspec, bias-variance
│   │   ├── multimodal.py          # Multimodal experiments
│   │   └── realdata.py            # LaLonde, STAR real data
│   └── utils/                     # Utilities
│       ├── io.py                  # Output path management
│       ├── metrics.py             # PEHE, ATE, bootstrap
│       ├── training.py            # Training utilities
│       └── latex.py               # LaTeX table/macro generation
├── output/                        # All outputs
│   ├── figures/                   # PDF figures (unified style)
│   ├── tables/                    # CSV data + LaTeX tables
│   └── results_macros.tex         # LaTeX macros for paper
├── data/                          # Raw data
├── external/                      # External dependencies
├── CTRL-DML-Paper/               # Paper directory
│   └── main.tex                   # Uses macros from output/
├── reproduce.sh                   # One-click reproduction script
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
# Full reproduction (runs all experiments + generates macros)
./reproduce.sh

# Or just regenerate LaTeX macros from existing CSV data
python -m src.utils.latex --generate-macros

# Paper compilation (from CTRL-DML-Paper/)
cd CTRL-DML-Paper && pdflatex main.tex
```

---

## Methodology

- **Sparse attention:** `TabularAttention` with L1 penalty to mute noise features.
- **Two-tower fusion:** text embeddings + tabular dense tower for multimodal confounders.
- **Orthogonal head:** DML-style R-learner with ratio targets, warm-start from plugin, and distillation.
- **Pseudo-outcome stabilization:** W-clipping (min |W|=0.05), Z-clipping (±5), weighted loss.
- **Targets:** heads for Y(0), Y(1), and propensity with targeted regularization.
- **UQ:** Monte Carlo Dropout + conformal calibration for interval estimates.

---

## Citation & Contact

This is an independent research effort on robust and multimodal causal inference. If it helps your work, please star the repo. For questions, open an issue or reach out via GitHub.
