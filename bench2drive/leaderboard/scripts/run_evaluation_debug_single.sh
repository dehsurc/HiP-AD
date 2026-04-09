#!/bin/bash
# Single scenario debug script for diagnosing ground collision issues
# Usage: bash bench2drive/leaderboard/scripts/run_evaluation_debug_single.sh [GPU_RANK] [TASK_ID]
#   GPU_RANK: GPU index (default: 0)
#   TASK_ID:  split index (default: 1, which contains the route_dev failures)

BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True

CONFIG_NAME=hipad_b2d_stage2

TEAM_AGENT=bench2drive/leaderboard/team_code/hipad_b2d_agent.py
TEAM_CONFIG=/home/yongjae/e2e/HiP-AD/projects/configs/$CONFIG_NAME.py+\
/home/yongjae/e2e/HiP-AD/ckpts/HiP-AD_stage2.pth

PLANNER_TYPE=traj
BASE_ROUTES=bench2drive/leaderboard/data/splits16/bench2drive220

SAVE_PATH=evaluation/${CONFIG_NAME}_debug
BASE_CHECKPOINT_ENDPOINT=evaluation/${CONFIG_NAME}_debug/${CONFIG_NAME}_debug

GPU_RANK=${1:-0}
TASK_ID=${2:-1}

mkdir -p "$SAVE_PATH"

PORT=$((BASE_PORT + TASK_ID * 200))
TM_PORT=$((BASE_TM_PORT + TASK_ID * 200))
ROUTES="${BASE_ROUTES}_${TASK_ID}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}_${TASK_ID}.json"

echo -e "\033[33m==================== DEBUG SINGLE RUN ====================\033[0m"
echo -e "GPU_RANK: $GPU_RANK"
echo -e "TASK_ID:  $TASK_ID"
echo -e "PORT:     $PORT"
echo -e "TM_PORT:  $TM_PORT"
echo -e "ROUTES:   $ROUTES"
echo -e "\033[33m==========================================================\033[0m"

# Run in foreground (not background) for easier debugging
bash bench2drive/leaderboard/scripts/run_evaluation.sh \
    $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG \
    $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 | tee ${CHECKPOINT_ENDPOINT%.json}.log
