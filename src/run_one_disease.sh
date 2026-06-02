#!/bin/bash
# One-disease test: runs each model on a single specified disease to verify
# paths and end-to-end execution without launching the full benchmark.

if [ -z "$1" ]; then
    echo "Usage: $0 <ICD10_DISEASE_ID>"
    echo ""
    echo "Example:"
    echo "  ./src/run_one_disease.sh ICD10_M41"
    echo ""
    echo "The disease ID must exist in the dataset and meet the temporal split criteria"
    echo "(≥15 GDAs total, ≥5 associations before the 2019 cutoff, ≥1 after)."
    exit 1
fi

DISEASE_ID="$1"
export ONE_DISEASE_ID="$DISEASE_ID"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/../logs/one_disease"
mkdir -p "$LOG_DIR"

echo "=== GP_benchmark one-disease test ==="
echo "Disease : $DISEASE_ID"
echo "Repo    : $(dirname "$SCRIPT_DIR")"
echo "Logs    : $LOG_DIR"
echo ""

# --- Environment 1: .venv (kernel method / mf / rf) ---
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

echo "[1/5] kernel method (diffusion) ..."
python -u main_diffusion.py \
  'diffusion_2019' \
  "results/one_disease/${DISEASE_ID}/diffusion" 2019 'disgenet' \
  2>&1 | tee "$LOG_DIR/diffusion.log"
[ "${PIPESTATUS[0]}" -eq 0 ] && echo "      [OK]" || echo "      [FAIL] — see $LOG_DIR/diffusion.log"

# echo "[1/5] kernel method (diffusion) ..."
# python main_diffusion.py \
#   'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019' \
#   "results/one_disease/${DISEASE_ID}/diffusion" 2019 'disgenet' \
#   > "$LOG_DIR/diffusion.log" 2>&1 \
#   && echo "      [OK]" || echo "      [FAIL] — see $LOG_DIR/diffusion.log"

# echo "[2/5] matrix factorisation ..."
# python main_mf.py \
#   'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_pca' \
#   "results/one_disease/${DISEASE_ID}/mf" 2019 'disgenet' 200 500 \
#   > "$LOG_DIR/mf.log" 2>&1 \
#   && echo "      [OK]" || echo "      [FAIL] — see $LOG_DIR/mf.log"

# echo "[3/5] random forest ..."
# python main_rf.py \
#   'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_2' \
#   "results/one_disease/${DISEASE_ID}/rf" 2019 \
#   > "$LOG_DIR/rf.log" 2>&1 \
#   && echo "      [OK]" || echo "      [FAIL] — see $LOG_DIR/rf.log"

# # --- Environment 2: new_esm_env (nn / gnn) ---
# source /itf-fi-ml/shared/users/ziyuzh/new_esm_env/bin/activate

# echo "[4/5] neural network ..."
# python main_nn_non_para.py \
#   'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_2' \
#   "results/one_disease/${DISEASE_ID}/nn" 2019 \
#   > "$LOG_DIR/nn.log" 2>&1 \
#   && echo "      [OK]" || echo "      [FAIL] — see $LOG_DIR/nn.log"

# echo "[5/5] graph neural network ..."
# python main_gnn.py \
#   'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_pca' \
#   "results/one_disease/${DISEASE_ID}/gnn" 2019 \
#   > "$LOG_DIR/gnn.log" 2>&1 \
#   && echo "      [OK]" || echo "      [FAIL] — see $LOG_DIR/gnn.log"

echo ""
echo "One-disease test complete."
echo "Results : $(dirname "$SCRIPT_DIR")/results/one_disease/$DISEASE_ID/"
echo "Logs    : $LOG_DIR/"
