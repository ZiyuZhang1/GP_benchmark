#!/bin/bash

# Resolve the directory containing this script so the script works from any cwd.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"

# --- Environment 1: .venv (diffusion / mf / rf) ---
source /itf-fi-ml/shared/users/ziyuzh/.venv/bin/activate

python main_diffusion.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019,uniport_bio,uniport_seq,uniport_esm' "results/2019_df_oob_mid" 2019 'disgenet' > "$LOG_DIR/2019_df_oob_mid.log" 2>&1
python main_diffusion.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019' "results/2019_df_oob_ppi_mid" 2019 'disgenet' > "$LOG_DIR/2019_df_oob_ppi_mid.log" 2>&1

python main_mf.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_pca,uniport_bio,uniport_seq,uniport_esm' "results/2019_mf" 2019 'disgenet' 200 500 > "$LOG_DIR/2019_mf.log" 2>&1

python main_rf.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_2,uniport_bio,uniport_seq,uniport_esm' "results/2019_rf" 2019 > "$LOG_DIR/2019_rf.log" 2>&1

# --- Environment 2: new_esm_env (nn / gnn) ---
source /itf-fi-ml/shared/users/ziyuzh/new_esm_env/bin/activate

# Stay in GP_benchmark/src — no cd to the old svm/src
python main_nn_non_para.py 'uniport_ppi_2019,ppi_2019_dw_40,uniport_bio,uniport_seq,uniport_esm,diffusion_2019_2' "results/2019_nn_oob" 2019 > "$LOG_DIR/2019_nn_oob.log" 2>&1
python main_nn_non_para.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_2' "results/2019_nn_oob_ppi" 2019 > "$LOG_DIR/2019_nn_oob_ppi.log" 2>&1
python main_gnn.py 'uniport_ppi_2019,ppi_2019_dw_40,diffusion_2019_pca,uniport_bio,uniport_seq,uniport_esm' "results/2019_gnn" 2019 > "$LOG_DIR/2019_gnn.log" 2>&1
