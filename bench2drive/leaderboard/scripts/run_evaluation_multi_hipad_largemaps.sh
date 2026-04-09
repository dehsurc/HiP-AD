#!/bin/bash
# Auto-monitoring evaluation script for Town12/Town13 (large maps) only
# - 12 tasks (splits 0-9, 10, 15), run 4 at a time (2 per GPU on GPU 2 and 3)
# - Monitors each task; if CARLA crashes, auto-restarts on a free GPU slot
# - Queue-based: when a slot finishes/crashes, it picks the next pending task
# - Max retries per task to avoid infinite loops

set -o pipefail

MAX_RETRIES=5
MAX_CONCURRENT=1  # 1 GPU only (GPU 0)
MONITOR_INTERVAL=30  # seconds between health checks
GPU_LOG_INTERVAL=120  # seconds between GPU status logs

CONFIG_NAME=hipad_b2d_stage2
SAVE_PATH=evaluation/${CONFIG_NAME}_largemaps
BASE_CHECKPOINT_ENDPOINT=evaluation/${CONFIG_NAME}_largemaps/${CONFIG_NAME}_largemaps
MONITOR_LOG="${SAVE_PATH}/monitor.log"

TEAM_AGENT=bench2drive/leaderboard/team_code/hipad_b2d_agent.py
TEAM_CONFIG="/home/yongjae/e2e/HiP-AD/projects/configs/${CONFIG_NAME}.py+/home/yongjae/e2e/HiP-AD/ckpts/HiP-AD_stage2.pth"
PLANNER_TYPE=traj
IS_BENCH2DRIVE=True
BASE_ROUTES=bench2drive/leaderboard/data/splits16/bench2drive220

mkdir -p "$SAVE_PATH"

# ────────���─────────────────────────────────────
# Configuration
# ───────────────��──────────────────────────────
TASK_LIST=(2 3 8)

# 1 GPU slot: GPU 0 only (GPU 1 CARLA crash)
GPU_SLOTS=(0)

port_for_task() {
    local task_id=$1
    echo $((30000 + task_id * 200))
}
tm_port_for_task() {
    local task_id=$1
    echo $((50000 + task_id * 200))
}
routes_for_task() {
    local task_id=$1
    if [ "$task_id" -eq 10 ] || [ "$task_id" -eq 15 ]; then
        echo "${BASE_ROUTES}_${task_id}_town12_13_only.xml"
    else
        echo "${BASE_ROUTES}_${task_id}.xml"
    fi
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

# ──���───────────────────────────────────────────
# Task state tracking (indexed by task slot in TASK_LIST)
# ─────���─────────────────────────────��──────────
declare -A SLOT_PID       # PID of running task
declare -A SLOT_GPU       # which GPU it's on
declare -A SLOT_RETRIES   # retry count
declare -A SLOT_STATUS    # pending / running / done / failed

for slot in "${!TASK_LIST[@]}"; do
    SLOT_RETRIES[$slot]=0
    SLOT_STATUS[$slot]="pending"
done

# GPU slot tracking: which task slot is running on each GPU slot
declare -A GPU_SLOT_TASK  # GPU_SLOT_TASK[gpu_slot_idx] = task_slot or ""
for i in "${!GPU_SLOTS[@]}"; do
    GPU_SLOT_TASK[$i]=""
done

# ──────────────────────────────────────────────
# Skip the crashing route in checkpoint JSON
# Inserts a dummy record with status "Failed - CARLA crashed"
# which the resume logic treats as completed (skips over)
# ──────────────────────────────────────────────
skip_crashing_route() {
    local task_id=$1
    local checkpoint="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.json"
    local routes=$(routes_for_task $task_id)

    if [ ! -f "$checkpoint" ]; then
        log "  SKIP: no checkpoint for task=$task_id"
        return 1
    fi

    python3 << PYEOF
import json, xml.etree.ElementTree as ET

ckpt_path = "$checkpoint"
xml_path = "$routes"

with open(ckpt_path) as f:
    data = json.load(f)

progress = data['_checkpoint']['progress']
done_count = progress[0]
total = progress[1]

if done_count >= total:
    print(f"Task $task_id: already complete ({done_count}/{total}), nothing to skip")
    exit(0)

tree = ET.parse(xml_path)
routes = tree.getroot().findall('route')
crash_route = routes[done_count]
route_id = crash_route.get('id')
town = crash_route.get('town')

dummy = {
    "route_id": f"RouteScenario_{route_id}_rep0",
    "index": done_count,
    "status": "Failed - CARLA crashed",
    "infractions": {k: [] for k in [
        "collisions_layout","collisions_pedestrian","collisions_vehicle",
        "outside_route_lanes","red_light","route_dev","route_timeout",
        "stop_infraction","vehicle_blocked","min_speed_infractions",
        "yield_emergency_vehicle_infractions"]},
    "scores": {"score_composed": 0.0, "score_penalty": 0.0, "score_route": 0.0},
    "meta": {"town": town, "skipped": "CARLA Signal 11 crash"}
}

data['_checkpoint']['records'].append(dummy)
data['_checkpoint']['progress'] = [done_count + 1, total]

with open(ckpt_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Task $task_id: skipped RouteScenario_{route_id} ({town}), progress={done_count+1}/{total}")
PYEOF
}

# ──────────────────────────────────────────────
# Kill a specific task's processes (not the whole GPU)
# ──────────────────────────────────────────────
cleanup_task() {
    local slot=$1
    local pid=${SLOT_PID[$slot]}
    if [ -n "$pid" ]; then
        # Kill the bash wrapper and its children (CARLA + python)
        pkill -P "$pid" 2>/dev/null
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi
}

# ─────────��────────────────────────────────────
# Launch a task on a specific GPU slot
# ──────��─────────────���─────────────────────────
launch_task() {
    local slot=$1
    local gpu_slot_idx=$2
    local gpu=${GPU_SLOTS[$gpu_slot_idx]}
    local task_id=${TASK_LIST[$slot]}
    local port=$(port_for_task $task_id)
    local tm_port=$(tm_port_for_task $task_id)
    local routes=$(routes_for_task $task_id)
    local checkpoint="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.json"
    local task_log="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.log"
    local retry_count=${SLOT_RETRIES[$slot]:-0}

    log "LAUNCH task=$task_id gpu=$gpu gpu_slot=$gpu_slot_idx port=$port retry=$retry_count routes=$(basename $routes)"

    bash bench2drive/leaderboard/scripts/run_evaluation.sh \
        $port $tm_port $IS_BENCH2DRIVE $routes $TEAM_AGENT "$TEAM_CONFIG" \
        $checkpoint $SAVE_PATH $PLANNER_TYPE $gpu \
        > "$task_log" 2>&1 &

    SLOT_PID[$slot]=$!
    SLOT_GPU[$slot]=$gpu
    SLOT_STATUS[$slot]="running"
    GPU_SLOT_TASK[$gpu_slot_idx]=$slot
    log "  PID=${SLOT_PID[$slot]}"
}

# ──��─────────���─────────────────────────────────
# Check if a task is still running, finished, or crashed
# ───���──────────────────────────────────────────
check_task() {
    local slot=$1
    local pid=${SLOT_PID[$slot]}

    if [ -z "$pid" ]; then
        echo "not_started"
        return
    fi

    if kill -0 "$pid" 2>/dev/null; then
        echo "running"
    else
        wait "$pid" 2>/dev/null
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "done"
        else
            echo "crashed"
        fi
    fi
}

# ───���──────────────────────────────────────────
# Find next pending task slot, or return empty
# ───��──────────────────────────────────────────
next_pending_slot() {
    for slot in "${!TASK_LIST[@]}"; do
        if [ "${SLOT_STATUS[$slot]}" = "pending" ]; then
            echo "$slot"
            return
        fi
    done
    echo ""
}

# ─────────────────────��────────────────────────
# Find the GPU slot index for a given task slot
# ───────────────���──────────────────────────────
find_gpu_slot_for_task() {
    local task_slot=$1
    for i in "${!GPU_SLOTS[@]}"; do
        if [ "${GPU_SLOT_TASK[$i]}" = "$task_slot" ]; then
            echo "$i"
            return
        fi
    done
    echo ""
}

# ────���───────────────────��─────────────────────
# Log GPU status
# ──────────────���───────────────────────────��───
log_gpu_status() {
    log "GPU STATUS:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | while read line; do
        log "  GPU $line"
    done
}

# ──���──────────��────────────────────────────────
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
    total = data.get('_checkpoint', {}).get('progress', [0,0])
    print(f'task={$task_id} progress={total[0]}/{total[1]} status=${SLOT_STATUS[$slot]}')
except: print(f'task={$task_id} progress=? status=${SLOT_STATUS[$slot]}')
" 2>/dev/null)
            log "  $progress"
        else
            log "  task=${task_id} progress=not_started status=${SLOT_STATUS[$slot]}"
        fi
    done
}

# ────���──────────────��──────────────────────────
# Main
# ────────────��─────────────────────────────────
log "=========================================="
log "Starting monitored large-map evaluation"
log "Tasks: ${TASK_LIST[*]}"
log "GPUs:  0, 1 (1 task per GPU, 2 parallel)"
log "Max retries per task: $MAX_RETRIES"
log "=========================================="

# Initial launch: fill all 4 GPU slots
for gpu_slot_idx in "${!GPU_SLOTS[@]}"; do
    slot=$(next_pending_slot)
    if [ -n "$slot" ]; then
        launch_task "$slot" "$gpu_slot_idx"
        sleep 10
    fi
done

log_gpu_status

# Monitor loop
last_gpu_log=$(date +%s)

while true; do
    sleep $MONITOR_INTERVAL

    any_active=false

    for slot in "${!TASK_LIST[@]}"; do
        [ "${SLOT_STATUS[$slot]}" = "done" ] && continue
        [ "${SLOT_STATUS[$slot]}" = "failed" ] && continue
        [ "${SLOT_STATUS[$slot]}" = "pending" ] && { any_active=true; continue; }

        # Status is "running" — check actual state
        status=$(check_task $slot)
        task_id=${TASK_LIST[$slot]}
        gpu=${SLOT_GPU[$slot]}
        gpu_slot_idx=$(find_gpu_slot_for_task $slot)

        case $status in
            running)
                any_active=true
                ;;
            done)
                log "DONE task=$task_id on gpu=$gpu (slot $gpu_slot_idx)"
                SLOT_STATUS[$slot]="done"
                GPU_SLOT_TASK[$gpu_slot_idx]=""
                # GPU slot is free �� pick next pending task
                next=$(next_pending_slot)
                if [ -n "$next" ]; then
                    sleep 5
                    launch_task "$next" "$gpu_slot_idx"
                    any_active=true
                fi
                ;;
            crashed)
                retry=${SLOT_RETRIES[$slot]:-0}
                retry=$((retry + 1))
                SLOT_RETRIES[$slot]=$retry

                if [ $retry -le $MAX_RETRIES ]; then
                    log "CRASH task=$task_id on gpu=$gpu (retry $retry/$MAX_RETRIES) — skipping route and restarting..."
                    cleanup_task $slot
                    # Skip the crashing route in checkpoint so we don't repeat it
                    skip_crashing_route $task_id
                    sleep 15
                    launch_task "$slot" "$gpu_slot_idx"
                    any_active=true
                else
                    log "GIVING UP on task=$task_id after $MAX_RETRIES retries"
                    SLOT_STATUS[$slot]="failed"
                    SLOT_PID[$slot]=""
                    GPU_SLOT_TASK[$gpu_slot_idx]=""
                    # GPU slot is free — pick next pending task
                    next=$(next_pending_slot)
                    if [ -n "$next" ]; then
                        sleep 5
                        launch_task "$next" "$gpu_slot_idx"
                        any_active=true
                    fi
                fi
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

    # Exit if nothing active
    if [ "$any_active" = false ]; then
        break
    fi
done

log "=========================================="
log "All tasks completed. Final progress:"
log_progress
log_gpu_status
log "=========================================="
