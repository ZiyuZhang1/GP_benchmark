#!/bin/bash
# Smoke test: runs each model on exactly ONE disease to verify path correctness.
# Much faster than run_all.sh; does not change production outputs.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export SMOKE_TEST=1

LOG_DIR="$SCRIPT_DIR/../logs/smoke"
mkdir -p "$LOG_DIR"

echo "=== GP_benchmark smoke test ==="
echo "Repo: $(dirname "$SCRIPT_DIR")"
echo "Logs: $LOG_DIR"
echo ""

# --- Environment 1: .venv (diffusion / mf / rf) ---
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

echo "[1/5] diffusion ..."
python main_diffusion.py \
  'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019' \
  'results/smoke/diffusion' 2019 'disgenet' \
  > "$LOG_DIR/smoke_diffusion.log" 2>&1 \
  && echo "      [OK]" || echo "      [FAIL] — see $LOG_DIR/smoke_diffusion.log"

echo "[2/5] mf ..."
python main_mf.py \
  'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_pca' \
  'results/smoke/mf' 2019 'disgenet' 20 50 \
  > "$LOG_DIR/smoke_mf.log" 2>&1 \
  && echo "      [OK]" || echo "      [FAIL] — see $LOG_DIR/smoke_mf.log"

echo "[3/5] rf ..."
python main_rf.py \
  'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_2' \
  'results/smoke/rf' 2019 \
  > "$LOG_DIR/smoke_rf.log" 2>&1 \
  && echo "      [OK]" || echo "      [FAIL] — see $LOG_DIR/smoke_rf.log"

# --- Environment 2: new_esm_env (nn / gnn) ---
source /itf-fi-ml/shared/users/ziyuzh/new_esm_env/bin/activate

echo "[4/5] nn ..."
python main_nn_non_para.py \
  'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_2' \
  'results/smoke/nn' 2019 \
  > "$LOG_DIR/smoke_nn.log" 2>&1 \
  && echo "      [OK]" || echo "      [FAIL] — see $LOG_DIR/smoke_nn.log"

echo "[5/5] gnn ..."
python main_gnn.py \
  'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_pca' \
  'results/smoke/gnn' 2019 \
  > "$LOG_DIR/smoke_gnn.log" 2>&1 \
  && echo "      [OK]" || echo "      [FAIL] — see $LOG_DIR/smoke_gnn.log"

echo ""
echo "Smoke test complete. Results in $(dirname "$SCRIPT_DIR")/results/smoke/"
