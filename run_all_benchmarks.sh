#!/bin/bash
set -e

echo "Starting Full Benchmark Suite..."
echo "================================"

# 1. Twins (Smallest/Fastest new one)
echo ">>> Running Twins Benchmark..."
python src/run_benchmark_final.py --benchmark twins | tee results_twins.txt

# 2. ACIC (Medium)
echo ">>> Running ACIC2016 Benchmark (20 simulations)..."
python src/run_benchmark_final.py --benchmark acic | tee results_acic.txt

# 3. Noise Robustness (Original)
echo ">>> Running Noise Robustness Benchmark..."
python src/run_benchmark_final.py --benchmark noise | tee results_noise.txt

# 4. IHDP (Longest - 100 experiments)
echo ">>> Running IHDP Benchmark (100 experiments)..."
python src/run_benchmark_final.py --benchmark ihdp | tee results_ihdp.txt

echo "================================"
echo "All benchmarks completed."
