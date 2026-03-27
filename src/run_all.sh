#!/usr/bin/env bash
# Run the full GP benchmark pipeline.
#
# Usage (from the project root):
#   bash src/run_all.sh
#
# Prerequisites
# -------------
# Edit MAIN_ENV and ESM_ENV below to point to your virtual/conda environments.
#   MAIN_ENV  – scikit-learn, smurff, scipy, pandas, rdkit, gseapy
#   ESM_ENV   – PyTorch, PyTorch Geometric, torch_sparse (for NN and GNN)

set -euo pipefail

MAIN_ENV=""   # e.g. /path/to/venv  or use: conda activate myenv
ESM_ENV=""    # e.g. /path/to/esm_env

# Feature sets
FEATURES_FULL="uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_2,uniport_bio,uniport_seq,uniport_esm"
FEATURES_PPI="uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019"
YEAR=2019

# Move to project root (parent of this script's directory)
cd "$(dirname "$0")/.."
mkdir -p logs

# ── Main environment ───────────────────────────────────────────────────────────
source "${MAIN_ENV}/bin/activate"

echo "=== Kernel (diffusion SVM) ==="
python src/main_kernel.py "${FEATURES_FULL}" "results/2019_kernel_full" ${YEAR} \
    2>&1 | tee logs/kernel_full.log
python src/main_kernel.py "${FEATURES_PPI}" "results/2019_kernel_ppi" ${YEAR} \
    2>&1 | tee logs/kernel_ppi.log

echo "=== Matrix Factorisation ==="
python src/main_mf.py "${FEATURES_FULL}" "results/2019_mf" ${YEAR} 200 500 \
    2>&1 | tee logs/mf.log

echo "=== Random Forest ==="
python src/main_rf.py "${FEATURES_FULL}" "results/2019_rf" ${YEAR} \
    2>&1 | tee logs/rf.log

# ── Deep-learning environment ──────────────────────────────────────────────────
source "${ESM_ENV}/bin/activate"

echo "=== Neural Network ==="
python src/main_nn.py "${FEATURES_FULL}" "results/2019_nn" ${YEAR} \
    2>&1 | tee logs/nn.log
python src/main_nn.py "${FEATURES_PPI}" "results/2019_nn_ppi" ${YEAR} \
    2>&1 | tee logs/nn_ppi.log

echo "=== Graph Neural Network ==="
python src/main_gnn.py "${FEATURES_FULL}" "results/2019_gnn" ${YEAR} \
    2>&1 | tee logs/gnn.log

echo ""
echo "All pipelines complete. Results are in results/."
