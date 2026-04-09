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
SPARE_GPUS=(6 7)  # fallback GPUs when a GPU repeatedly crashes
GPU_CRASH_COUNT=()  # per-GPU crash counter
SAME_GPU_CRASH_THRESHOLD=2  # switch GPU after this many crashes on same GPU

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
# Try to assign a spare GPU for a repeatedly crashing task
# ──────────────────────────────────────────────
try_reassign_gpu() {
    local slot=$1
    local old_gpu=${GPU_LIST[$slot]}
    local crash_key="gpu_${old_gpu}"
    local count=${GPU_CRASH_COUNT[$crash_key]:-0}
    count=$((count + 1))
    GPU_CRASH_COUNT[$crash_key]=$count

    if [ $count -ge $SAME_GPU_CRASH_THRESHOLD ] && [ ${#SPARE_GPUS[@]} -gt 0 ]; then
        local new_gpu=${SPARE_GPUS[0]}
        SPARE_GPUS=("${SPARE_GPUS[@]:1}")  # pop first spare
        SPARE_GPUS+=($old_gpu)  # old GPU becomes spare (maybe works for other tasks)
        GPU_LIST[$slot]=$new_gpu
        log "GPU REASSIGN task=${TASK_LIST[$slot]}: gpu $old_gpu -> $new_gpu (crashed $count times on gpu $old_gpu)"
        GPU_CRASH_COUNT[$crash_key]=0  # reset counter for old GPU
    fi
}

# ──────────────────────────────────────────────
# Skip the current crashing route in checkpoint JSON
# ──────────────────────────────────────────────
skip_crashing_route() {
    local slot=$1
    local task_id=${TASK_LIST[$slot]}
    local checkpoint="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.json"
    local routes_file=$(routes_for_task $task_id)

    python3 << PYEOF
import json, xml.etree.ElementTree as ET
try:
    with open('$checkpoint') as f:
        data = json.load(f)
    progress = data['_checkpoint']['progress']
    done_count = progress[0]

    tree = ET.parse('$routes_file')
    routes = tree.getroot().findall('route')
    if done_count >= len(routes):
        print(f"  task=$task_id: no more routes to skip")
    else:
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
        data['_checkpoint']['progress'] = [done_count + 1, len(routes)]
        with open('$checkpoint', 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  task=$task_id: SKIPPED RouteScenario_{route_id} ({town}), progress={done_count+1}/{len(routes)}")
except Exception as e:
    print(f"  task=$task_id: skip error: {e}")
PYEOF
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
        local retry=${TASK_RETRIES[$slot]:-0}
        if [ $retry -ge $MAX_RETRIES ]; then
            echo "done"  # gave up, treat as done to stop checking
        else
            echo "not_started"
        fi
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
                    log "CRASH DETECTED task=$local_task_id (retry $retry/$MAX_RETRIES) — skipping crashed route and restarting..."
                    skip_crashing_route $slot 2>&1 | while read line; do log "$line"; done
                    try_reassign_gpu $slot
                    sleep 15
                    launch_task $slot
                    any_running=true
                else
                    log "GIVING UP on task=$local_task_id after $MAX_RETRIES retries"
                    TASK_PIDS[$slot]=""  # clear PID so we stop checking
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
