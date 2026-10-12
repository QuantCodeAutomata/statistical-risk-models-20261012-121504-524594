#!/usr/bin/env bash
# reproduce.sh — single-command reproduction entry point for all experiments.
# Usage: bash reproduce.sh
# Requires: MASSIVE_API_KEY set in .env (copy .env.example to .env first).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Statistical Risk Models — Reproduction Script ==="
echo "Paper: Kakushadze & Yu (2016), SSRN:2732453"
echo ""

# 1. Verify environment
if [ ! -f ".env" ]; then
    echo "ERROR: .env not found. Copy .env.example to .env and set MASSIVE_API_KEY."
    exit 1
fi

# 2. Run unit tests
echo "--- Running unit tests ---"
python -m pytest tests/ -v --tb=short 2>&1 | tee results/test_output.txt
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "ERROR: Unit tests failed. Fix before running experiments."
    exit 1
fi
echo ""

# 3. Run experiments sequentially
echo "--- Running Exp 1: K-selection algorithm comparison ---"
python -m exp.exp1_k_selection 2>&1 | tee results/exp1_log.txt

echo "--- Running Exp 2: PCA methods comparison ---"
python -m exp.exp2_pca_methods 2>&1 | tee results/exp2_log.txt

echo "--- Running Exp 3: Shrinkage models backtest ---"
python -m exp.exp3_shrinkage 2>&1 | tee results/exp3_log.txt

echo "--- Running Exp 4: Fixed-K and ad hoc sensitivity ---"
python -m exp.exp4_sensitivity 2>&1 | tee results/exp4_log.txt

echo ""
echo "=== Done. Results in results/ ==="
ls -lh results/
