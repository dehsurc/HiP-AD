#!/bin/bash
# Auto-monitoring evaluation script with crash recovery
# - Launches evaluation tasks on small maps only (Town12/13 excluded)
# - Monitors each task; if CARLA crashes (Signal 11, exit -1), auto-restarts
# - Logs GPU status periodically
# - Max retries per task to avoid infinite loops

set -o pipefail

MAX_RETRIES=5
MONITOR_INTERVAL=30  # seconds between health checks
GPU_LOG_INTERVAL=120  # seconds between GPU status logs

CONFIG_NAME=hipad_b2d_stage2
SAVE_PATH=evaluation/$CONFIG_NAME
BASE_CHECKPOINT_ENDPOINT=evaluation/$CONFIG_NAME/$CONFIG_NAME
MONITOR_LOG="${SAVE_PATH}/monitor.log"

TEAM_AGENT=bench2drive/leaderboard/team_code/hipad_b2d_agent.py
TEAM_CONFIG="/workspace/HiP-AD/projects/configs/${CONFIG_NAME}.py+/workspace/HiP-AD/ckpts/HiP-AD_Stage_2.pth"
PLANNER_TYPE=traj
IS_BENCH2DRIVE=True
BASE_ROUTES=bench2drive/leaderboard/data/splits16/bench2drive220

mkdir -p "$SAVE_PATH"

# ──────────────────────────────────────────────
# Configuration: which tasks on which GPUs
# ──────────────────────────────────────────────
TASK_LIST=(10 11 12 13 14 15)
GPU_LIST=(0  1  2  3  4  5)

# Port assignment: each task gets a unique base port range
port_for_task() {
    local idx=$1
    echo $((30000 + idx * 200))
}
tm_port_for_task() {
    local idx=$1
    echo $((50000 + idx * 200))
}
routes_for_task() {
    local task_id=$1
    if [ "$task_id" -eq 10 ] || [ "$task_id" -eq 15 ]; then
        echo "${BASE_ROUTES}_${task_id}_notown12_13.xml"
    else
        echo "${BASE_ROUTES}_${task_id}.xml"
    fi
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

# ──────────────────────────────────────────────
# Kill any leftover CARLA/eval processes on a GPU
# ──────────────────────────────────────────────
cleanup_gpu() {
    local gpu=$1
    # Kill CARLA on this GPU
    ps aux | grep "graphicsadapter=${gpu}" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    sleep 1
}

# ──────────────────────────────────────────────
# Launch a single evaluation task
# ──────────────────────────────────────────────
declare -A TASK_PIDS
declare -A TASK_RETRIES

launch_task() {
    local slot=$1
    local task_id=${TASK_LIST[$slot]}
    local gpu=${GPU_LIST[$slot]}
    local port=$(port_for_task $slot)
    local tm_port=$(tm_port_for_task $slot)
    local routes=$(routes_for_task $task_id)
    local checkpoint="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.json"
    local task_log="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.log"
    local retry_count=${TASK_RETRIES[$slot]:-0}

    log "LAUNCH task=$task_id gpu=$gpu port=$port retry=$retry_count routes=$(basename $routes)"

    cleanup_gpu $gpu

    bash bench2drive/leaderboard/scripts/run_evaluation.sh \
        $port $tm_port $IS_BENCH2DRIVE $routes $TEAM_AGENT "$TEAM_CONFIG" \
        $checkpoint $SAVE_PATH $PLANNER_TYPE $gpu \
        > "$task_log" 2>&1 &

    TASK_PIDS[$slot]=$!
    log "  PID=${TASK_PIDS[$slot]}"
}

# ──────────────────────────────────────────────
# Check if a task is still running, finished, or crashed
# Returns: "running", "done", "crashed"
# ──────────────────────────────────────────────
check_task() {
    local slot=$1
    local pid=${TASK_PIDS[$slot]}
    local task_id=${TASK_LIST[$slot]}

    if [ -z "$pid" ]; then
        echo "not_started"
        return
    fi

    if kill -0 "$pid" 2>/dev/null; then
        echo "running"
    else
        # Process exited — check exit code
        wait "$pid" 2>/dev/null
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "done"
        else
            echo "crashed"
        fi
    fi
}

# ──────────────────────────────────────────────
# Log GPU status
# ──────────────────────────────────────────────
log_gpu_status() {
    log "GPU STATUS:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | while read line; do
        log "  GPU $line"
    done
}

# ──────────────────────────────────────────────
# Log progress from checkpoint JSONs
# ──────────────────────────────────────────────
log_progress() {
    for slot in "${!TASK_LIST[@]}"; do
        local task_id=${TASK_LIST[$slot]}
        local checkpoint="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.json"
        if [ -f "$checkpoint" ]; then
            local progress=$(python3 -c "
import json
try:
    with open('$checkpoint') as f:
        data = json.load(f)
    done = len([r for r in data.get('_checkpoint', {}).get('records', []) if r])
    total = data.get('_checkpoint', {}).get('progress', [0,0])
    print(f'task={$task_id} progress={total[0]}/{total[1]}')
except: print(f'task={$task_id} progress=?')
" 2>/dev/null)
            log "  $progress"
        fi
    done
}

# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────
log "=========================================="
log "Starting monitored evaluation"
log "Tasks: ${TASK_LIST[*]}"
log "GPUs:  ${GPU_LIST[*]}"
log "Max retries per task: $MAX_RETRIES"
log "=========================================="

# Initialize retry counts
for slot in "${!TASK_LIST[@]}"; do
    TASK_RETRIES[$slot]=0
done

# Launch all tasks
for slot in "${!TASK_LIST[@]}"; do
    launch_task $slot
    sleep 10  # stagger launches to reduce simultaneous GPU pressure
done

log_gpu_status

# Monitor loop
last_gpu_log=$(date +%s)
all_done=false

while [ "$all_done" = false ]; do
    sleep $MONITOR_INTERVAL
    all_done=true
    any_running=false

    for slot in "${!TASK_LIST[@]}"; do
        local_task_id=${TASK_LIST[$slot]}
        status=$(check_task $slot)

        case $status in
            running)
                all_done=false
                any_running=true
                ;;
            done)
                # Already finished successfully — skip
                ;;
            crashed)
                all_done=false
                retry=${TASK_RETRIES[$slot]:-0}
                retry=$((retry + 1))
                TASK_RETRIES[$slot]=$retry

                if [ $retry -le $MAX_RETRIES ]; then
                    log "CRASH DETECTED task=$local_task_id (retry $retry/$MAX_RETRIES) — restarting in 15s..."
                    sleep 15
                    launch_task $slot
                    any_running=true
                else
                    log "GIVING UP on task=$local_task_id after $MAX_RETRIES retries"
                fi
                ;;
            not_started)
                all_done=false
                ;;
        esac
    done

    # Periodic GPU logging
    now=$(date +%s)
    if [ $((now - last_gpu_log)) -ge $GPU_LOG_INTERVAL ]; then
        log_gpu_status
        log_progress
        last_gpu_log=$now
    fi

    if [ "$any_running" = false ] && [ "$all_done" = false ]; then
        # Nothing running but not all done — everything either finished or gave up
        all_done=true
    fi
done

log "=========================================="
log "All tasks completed. Final progress:"
log_progress
log_gpu_status
log "=========================================="
