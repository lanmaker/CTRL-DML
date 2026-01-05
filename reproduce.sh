#!/bin/bash
# CTRL-DML: One-click reproduction script
# Runs all experiments and generates LaTeX assets
#
# Usage:
#   ./reproduce.sh           # Full run (slow, ~hours)
#   ./reproduce.sh --fast    # Fast run for testing (~minutes)
#
# Requirements:
#   - Python 3.8+ with dependencies: pip install -r requirements.txt
#   - For real data: econml or causalml for LaLonde dataset

set -e  # Exit on error

# Work around OpenMP shared-memory restrictions in constrained environments.
export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Parse arguments
FAST_FLAG=""
if [[ "$1" == "--fast" ]]; then
    export CTRL_DML_FAST=1
    FAST_FLAG="(FAST MODE)"
    echo "Running in FAST mode - reduced epochs/seeds for testing"
fi

echo "========================================"
echo "CTRL-DML Reproduction Script $FAST_FLAG"
echo "========================================"
echo ""

# Create output directories
mkdir -p output/figures output/tables

# Track start time
START_TIME=$(date +%s)

run_experiment() {
    local name=$1
    local cmd=$2
    echo ""
    echo ">>> Running: $name"
    echo "    Command: $cmd"
    eval $cmd
    echo "<<< Completed: $name"
}

# Core experiments
run_experiment "Ablation Study" \
    "python -m src.experiments.ablation --mode all"

run_experiment "Cross-fitting Ablation" \
    "python -m src.experiments.crossfit_ablation"

run_experiment "IHDP/TWINS/ACIC Benchmarks" \
    "python -m src.experiments.benchmarks --dataset all"

run_experiment "Public Benchmarks (TWINS/ACIC Baselines)" \
    "python -m src.experiments.public_benchmarks"

run_experiment "Feature Role Analysis" \
    "python -m src.experiments.feature_roles"

# Synthetic experiments
run_experiment "Scaling Analysis" \
    "python -m src.experiments.synthetic --analysis scaling"

run_experiment "Uncertainty Quantification" \
    "python -m src.experiments.synthetic --analysis uq"

run_experiment "Training Dynamics" \
    "python -m src.experiments.synthetic --analysis dynamics"

# Robustness experiments
run_experiment "Nuisance Misspecification" \
    "python -m src.experiments.robustness --test nuisance"

run_experiment "Bias-Variance Analysis" \
    "python -m src.experiments.robustness --test bias_variance"

run_experiment "Noise Robustness Sweep" \
    "python -m src.experiments.robustness_sweep"

# Multimodal experiments
run_experiment "Multimodal Benchmark" \
    "python -m src.experiments.multimodal --experiment benchmark"

run_experiment "Multimodal Noise Sweep" \
    "python -m src.experiments.multimodal --experiment noise_sweep"

run_experiment "Multimodal BERT Baselines" \
    "python -m src.experiments.multimodal --experiment bert"

run_experiment "Yelp Semi-synthetic" \
    "python -m src.experiments.yelp_semisynth"

# Sensitivity sweep
run_experiment "Sensitivity Heatmap" \
    "python -m src.experiments.sensitivity"

# Real data (may fail if datasets not available)
echo ""
echo ">>> Running: Real Data Experiments (may skip if data unavailable)"
python -m src.experiments.realdata --dataset all 2>/dev/null || \
    echo "    Skipped: Real data not available (install econml for LaLonde)"

# Generate LaTeX macros from all results
echo ""
echo ">>> Generating LaTeX macros"
python -m src.utils.latex --generate-macros

# Record run metadata for reproducibility
echo ""
echo ">>> Recording run metadata"
python - <<'PY'
import json
import os
import platform
import subprocess
from pathlib import Path

def cmd(command: str):
    try:
        return subprocess.check_output(command, shell=True, text=True).strip()
    except Exception:
        return None

try:
    import torch
    torch_cuda = torch.cuda.is_available()
except Exception:
    torch_cuda = None

meta = {
    "timestamp": cmd("date -u +%Y-%m-%dT%H:%M:%SZ"),
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "git_rev": cmd("git rev-parse HEAD"),
    "git_status": cmd("git status --porcelain"),
    "pip_freeze": cmd("python -m pip freeze"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_cuda_available": torch_cuda,
    "nvidia_smi": cmd("nvidia-smi -L"),
    "env_ctrl_dml": {k: v for k, v in os.environ.items() if k.startswith("CTRL_DML_")},
}

Path("output").mkdir(exist_ok=True)
Path("output/run_metadata.json").write_text(json.dumps(meta, indent=2))
print("Saved: output/run_metadata.json")
PY

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "========================================"
echo "Reproduction Complete!"
echo "========================================"
echo "Duration: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Outputs:"
echo "  - Figures:  output/figures/"
echo "  - Tables:   output/tables/"
echo "  - Macros:   output/results_macros.tex"
echo ""
echo "To use in LaTeX paper:"
echo "  \\input{../output/results_macros.tex}"
