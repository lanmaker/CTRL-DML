# CTRL-DML Benchmarks

Scripts to benchmark CTRL-DML against baselines (DragonNet, Causal Forest) and to study scaling with larger sample sizes. Plots are saved as PDFs.

## Setup
- Python 3.9+ recommended.
- Install dependencies: `pip install -r requirements.txt`
- If you use the vendored CATENets (under `external/CATENets`), install it in editable mode: `pip install -e external/CATENets`.

## Running the benchmarks
- Final benchmark across noise dimensions (saves `benchmark_final.pdf` and `benchmark_final_delta.pdf`):
  - Fast mode (quicker sanity run): `CTRL_DML_FAST=1 python src/run_benchmark_final.py`
  - Full mode (best quality): `python src/run_benchmark_final.py`
- Scaling benchmark across sample sizes (saves `scaling_results.pdf` and `scaling_delta.pdf`):
  - Fast mode: `CTRL_DML_FAST=1 python src/run_scaling.py`
  - Full mode: `python src/run_scaling.py`

Fast mode reduces epochs/trees/sample sizes and seed count for quick checks. Full mode is slower but higher fidelity.

## Outputs
- Main performance plots and delta (CTRL-DML minus Causal Forest) plots are emitted as PDFs in the repo root. PDFs are `.gitignore`’d by default.

## Notes
- Both scripts keep logs in English and use PDFs for publication-ready plots.
- Treatment model uses `LogisticRegressionCV` in Causal Forest to avoid classifier warnings; outcome model uses `LassoCV`.
