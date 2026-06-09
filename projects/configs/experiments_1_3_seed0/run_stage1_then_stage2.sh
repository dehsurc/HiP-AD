#!/usr/bin/env bash
# Run E1 stage1 → E2 stage2 sequentially on the 1/3 subset (seed 0).
# Usage:
#   bash projects/configs/experiments_1_3_seed0/run_stage1_then_stage2.sh
# Run from the HiP-AD repo root.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PORT=${PORT:-28651}
GPUS=2

CFG_DIR="projects/configs/experiments_1_3_seed0"
STAGE1_CFG="${CFG_DIR}/E1_stage1_12ep_1_3_seed0.py"
STAGE2_CFG="${CFG_DIR}/E2_E1_stage2_18ep_1_3_seed0.py"
STAGE1_CKPT="work_dirs/exp/E1_stage1_12ep_1_3_seed0/latest.pth"

LOG_DIR="work_dirs/exp/_chain_logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)

echo "[chain] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  PORT=${PORT}  GPUS=${GPUS}"
echo "[chain] $(date) -- starting STAGE1: ${STAGE1_CFG}"
bash tools/dist_train.sh "${STAGE1_CFG}" ${GPUS} \
    2>&1 | tee "${LOG_DIR}/stage1_${TS}.log"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
    echo "[chain] ERROR: stage1 checkpoint not found at ${STAGE1_CKPT}; aborting stage2." >&2
    exit 1
fi

echo "[chain] $(date) -- STAGE1 done; starting STAGE2: ${STAGE2_CFG}"
bash tools/dist_train.sh "${STAGE2_CFG}" ${GPUS} \
    2>&1 | tee "${LOG_DIR}/stage2_${TS}.log"

echo "[chain] $(date) -- ALL DONE."
