#!/bin/bash
# Evaluate ALL small maps (task 0~15, full routes including Town12/13)
# Uses GPUs 0, 1
# Runs 2 tasks at a time, cycling through all 16 tasks

set -o pipefail

MAX_RETRIES=5
MAX_SKIP_RETRY_ROUNDS=3  # max rounds to retry skipped routes
MONITOR_INTERVAL=30
GPU_LOG_INTERVAL=120

CONFIG_NAME=code_epoch18
SAVE_PATH=evaluation/${CONFIG_NAME}
BASE_CHECKPOINT_ENDPOINT=evaluation/${CONFIG_NAME}/${CONFIG_NAME}
MONITOR_LOG="${SAVE_PATH}/monitor.log"

TEAM_AGENT=bench2drive/leaderboard/team_code/hipad_b2d_agent.py
TEAM_CONFIG="/home/yongjae/e2e/HiP-AD/ckpts/HiP-AD-Stage2_code.py+/home/yongjae/e2e/HiP-AD/ckpts/code_epoch18.pth"
PLANNER_TYPE=traj
IS_BENCH2DRIVE=True
BASE_ROUTES=bench2drive/leaderboard/data/splits16/bench2drive220

AVAILABLE_GPUS=(0 1)
SPARE_GPUS=()  # fallback GPUs when a GPU repeatedly crashes
SAME_GPU_CRASH_THRESHOLD=2
declare -A GPU_CRASH_COUNT  # per-GPU crash counter

mkdir -p "$SAVE_PATH"

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
    echo "${BASE_ROUTES}_${task_id}.xml"
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

cleanup_gpu() {
    local gpu=$1
    # Kill CARLA servers on this GPU
    ps aux | grep "graphicsadapter=${gpu}" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    # Kill any eval python processes on this GPU (CUDA_VISIBLE_DEVICES=gpu)
    ps aux | grep "leaderboard_evaluator" | grep "CUDA_VISIBLE_DEVICES=${gpu}\b" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    sleep 1
}

# Check if CARLA server is alive for a given GPU
is_carla_alive() {
    local gpu=$1
    ps aux | grep "graphicsadapter=${gpu}" | grep -v grep | grep -q "CarlaUE4"
}

# Check if eval python process is alive
is_eval_alive() {
    local pid=$1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# Kill orphan eval process (CARLA died, eval still running)
kill_orphan_eval() {
    local pid=$1
    local task_id=$2
    if [ -n "$pid" ]; then
        log "ORPHAN EVAL task=$task_id: CARLA dead but eval PID=$pid alive — killing eval"
        kill -9 "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
    fi
}

# Kill orphan CARLA (eval died, CARLA still running)
kill_orphan_carla() {
    local gpu=$1
    local task_id=$2
    log "ORPHAN CARLA task=$task_id: eval dead but CARLA on gpu=$gpu alive — killing CARLA"
    ps aux | grep "graphicsadapter=${gpu}" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    sleep 1
}

skip_crashing_route() {
    local task_id=$1
    local checkpoint="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.json"
    local routes_file=$(routes_for_task $task_id)

    python3 << PYEOF
import json, os, xml.etree.ElementTree as ET
try:
    tree = ET.parse('$routes_file')
    routes = tree.getroot().findall('route')
    total = len(routes)

    # Create checkpoint if it doesn't exist yet (CARLA died before writing anything)
    if not os.path.exists('$checkpoint'):
        data = {
            "_checkpoint": {
                "global_record": {},
                "progress": [0, total],
                "records": []
            }
        }
    else:
        with open('$checkpoint') as f:
            data = json.load(f)

    progress = data['_checkpoint']['progress']
    done_count = progress[0]

    if done_count >= total:
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
        data['_checkpoint']['progress'] = [done_count + 1, total]
        with open('$checkpoint', 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  task=$task_id: SKIPPED RouteScenario_{route_id} ({town}), progress={done_count+1}/{total}")
except Exception as e:
    print(f"  task=$task_id: skip error: {e}")
PYEOF
}

# ──────────────────────────────────────────────
# Check if a task is already fully completed
# ──────────────────────────────────────────────
is_task_complete() {
    local task_id=$1
    local checkpoint="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.json"
    local routes_file=$(routes_for_task $task_id)
    CKPT_PATH="$checkpoint" ROUTES_PATH="$routes_file" python3 << 'PYEOF'
import json, xml.etree.ElementTree as ET, os
try:
    with open(os.environ['CKPT_PATH']) as f:
        data = json.load(f)
    progress = data['_checkpoint']['progress']
    tree = ET.parse(os.environ['ROUTES_PATH'])
    total_routes = len(tree.getroot().findall('route'))
    complete = progress[0] >= total_routes and total_routes > 0
except Exception:
    complete = False
exit(0 if complete else 1)
PYEOF
}

# ──────────────────────────────────────────────
# Run a batch of tasks (up to 3) with monitoring
# ──────────────────────────────────────────────
run_batch() {
    local -a batch_tasks=("$@")
    local n=${#batch_tasks[@]}

    declare -A BATCH_PIDS
    declare -A BATCH_RETRIES
    declare -A BATCH_GPUS

    log "------------------------------------------"
    log "BATCH: tasks ${batch_tasks[*]}"
    log "------------------------------------------"

    # Assign GPUs and launch (skip already completed tasks)
    local any_launched=false
    for i in "${!batch_tasks[@]}"; do
        local task_id=${batch_tasks[$i]}
        local gpu=${AVAILABLE_GPUS[$i]}
        BATCH_GPUS[$i]=$gpu
        BATCH_RETRIES[$i]=0

        # Skip if already complete
        if is_task_complete $task_id; then
            log "SKIP task=$task_id (already complete)"
            BATCH_PIDS[$i]=""
            continue
        fi

        local port=$(port_for_task $task_id)
        local tm_port=$(tm_port_for_task $task_id)
        local routes=$(routes_for_task $task_id)
        local checkpoint="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.json"
        local task_log="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.log"

        log "LAUNCH task=$task_id gpu=$gpu port=$port routes=$(basename $routes)"
        cleanup_gpu $gpu

        bash bench2drive/leaderboard/scripts/run_evaluation.sh \
            $port $tm_port $IS_BENCH2DRIVE $routes $TEAM_AGENT "$TEAM_CONFIG" \
            $checkpoint $SAVE_PATH $PLANNER_TYPE $gpu \
            > "$task_log" 2>&1 &

        BATCH_PIDS[$i]=$!
        any_launched=true
        log "  PID=${BATCH_PIDS[$i]}"
        sleep 10
    done

    if [ "$any_launched" = false ]; then
        log "BATCH COMPLETE: all tasks already done"
        return
    fi

    # Monitor loop
    local last_gpu_log=$(date +%s)
    local all_done=false

    while [ "$all_done" = false ]; do
        sleep $MONITOR_INTERVAL
        all_done=true
        local any_running=false

        for i in "${!batch_tasks[@]}"; do
            local task_id=${batch_tasks[$i]}
            local pid=${BATCH_PIDS[$i]}

            [ -z "$pid" ] && continue

            local gpu=${BATCH_GPUS[$i]}
            local eval_alive=false
            local carla_alive=false
            is_eval_alive "$pid" && eval_alive=true
            is_carla_alive "$gpu" && carla_alive=true

            if [ "$eval_alive" = true ] && [ "$carla_alive" = false ]; then
                # Case 1: CARLA died, eval is zombie — check if task already done
                local ckpt_file="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.json"
                local routes_file=$(routes_for_task $task_id)
                local task_done=$(CKPT_PATH="$ckpt_file" ROUTES_PATH="$routes_file" python3 << 'PYCHECK'
import json, xml.etree.ElementTree as ET, os
try:
    with open(os.environ['CKPT_PATH']) as f: d = json.load(f)
    p = d['_checkpoint']['progress']
    t = len(ET.parse(os.environ['ROUTES_PATH']).getroot().findall('route'))
    print('yes' if p[0] >= t and t > 0 else 'no')
except Exception: print('no')
PYCHECK
)
                if [ "$task_done" = "yes" ]; then
                    kill_orphan_eval "$pid" "$task_id"
                    is_carla_alive "$gpu" && cleanup_gpu "$gpu"
                    log "DONE task=$task_id (completed despite CARLA crash)"
                    BATCH_PIDS[$i]=""
                    continue
                fi
                kill_orphan_eval "$pid" "$task_id"
                eval_alive=false
            elif [ "$eval_alive" = false ] && [ "$carla_alive" = true ]; then
                # Case 2: Eval died, CARLA is zombie — kill CARLA
                kill_orphan_carla "$gpu" "$task_id"
            fi

            if [ "$eval_alive" = true ]; then
                all_done=false
                any_running=true
            else
                wait "$pid" 2>/dev/null
                local exit_code=$?

                if [ $exit_code -eq 0 ]; then
                    # Cleanup any leftover CARLA even on success
                    is_carla_alive "$gpu" && cleanup_gpu "$gpu"
                    log "DONE task=$task_id"
                    BATCH_PIDS[$i]=""
                else
                    # Non-zero exit but task already complete → treat as success
                    if is_task_complete $task_id; then
                        is_carla_alive "$gpu" && cleanup_gpu "$gpu"
                        log "DONE task=$task_id (completed despite non-zero exit)"
                        BATCH_PIDS[$i]=""
                        continue
                    fi

                    all_done=false
                    local retry=${BATCH_RETRIES[$i]}
                    retry=$((retry + 1))
                    BATCH_RETRIES[$i]=$retry

                    if [ $retry -le $MAX_RETRIES ]; then
                        log "CRASH task=$task_id (retry $retry/$MAX_RETRIES) — skipping route and restarting..."
                        skip_crashing_route $task_id 2>&1 | while read line; do log "$line"; done

                        # After skip, check if task is now complete (all routes done/skipped)
                        if is_task_complete $task_id; then
                            is_carla_alive "$gpu" && cleanup_gpu "$gpu"
                            log "DONE task=$task_id (all routes completed after skip)"
                            BATCH_PIDS[$i]=""
                            continue
                        fi

                        # Try to reassign GPU if it keeps crashing
                        local gpu=${BATCH_GPUS[$i]}
                        local crash_key="gpu_${gpu}"
                        local gpu_crashes=${GPU_CRASH_COUNT[$crash_key]:-0}
                        gpu_crashes=$((gpu_crashes + 1))
                        GPU_CRASH_COUNT[$crash_key]=$gpu_crashes

                        if [ $gpu_crashes -ge $SAME_GPU_CRASH_THRESHOLD ] && [ ${#SPARE_GPUS[@]} -gt 0 ]; then
                            local new_gpu=${SPARE_GPUS[0]}
                            SPARE_GPUS=("${SPARE_GPUS[@]:1}")
                            SPARE_GPUS+=($gpu)
                            BATCH_GPUS[$i]=$new_gpu
                            GPU_CRASH_COUNT[$crash_key]=0
                            log "GPU REASSIGN task=$task_id: gpu $gpu -> $new_gpu (crashed $gpu_crashes times)"
                            gpu=$new_gpu
                        fi
                        local port=$(port_for_task $task_id)
                        local tm_port=$(tm_port_for_task $task_id)
                        local routes=$(routes_for_task $task_id)
                        local checkpoint="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.json"
                        local task_log="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.log"

                        sleep 15
                        cleanup_gpu $gpu

                        bash bench2drive/leaderboard/scripts/run_evaluation.sh \
                            $port $tm_port $IS_BENCH2DRIVE $routes $TEAM_AGENT "$TEAM_CONFIG" \
                            $checkpoint $SAVE_PATH $PLANNER_TYPE $gpu \
                            > "$task_log" 2>&1 &

                        BATCH_PIDS[$i]=$!
                        any_running=true
                        log "  Restarted task=$task_id PID=${BATCH_PIDS[$i]}"
                    else
                        log "GIVING UP task=$task_id after $MAX_RETRIES retries"
                        BATCH_PIDS[$i]=""
                    fi
                fi
            fi
        done

        # Periodic GPU/progress logging
        local now=$(date +%s)
        if [ $((now - last_gpu_log)) -ge $GPU_LOG_INTERVAL ]; then
            log "GPU STATUS:"
            nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | while read line; do
                log "  GPU $line"
            done
            for i in "${!batch_tasks[@]}"; do
                local task_id=${batch_tasks[$i]}
                local checkpoint="${BASE_CHECKPOINT_ENDPOINT}_${task_id}.json"
                if [ -f "$checkpoint" ]; then
                    CKPT_PATH="$checkpoint" TASK_ID="$task_id" python3 << 'PYPROG' 2>/dev/null | while read line; do log "$line"; done
import json, os
try:
    with open(os.environ['CKPT_PATH']) as f:
        d = json.load(f)
    p = d.get('_checkpoint',{}).get('progress',[0,0])
    tid = os.environ['TASK_ID']
    print(f'  task={tid} progress={p[0]}/{p[1]}')
except:
    tid = os.environ.get('TASK_ID','?')
    print(f'  task={tid} progress=?')
PYPROG
                fi
            done
            last_gpu_log=$now
        fi

        if [ "$any_running" = false ]; then
            all_done=true
        fi
    done

    log "BATCH COMPLETE: tasks ${batch_tasks[*]}"
}

# ──────────────────────────────────────────────
# Find tasks with skipped routes, generate retry
# XML files, and reset checkpoint for those routes
# Returns task IDs with skipped routes via stdout
# ──────────────────────────────────────────────
find_and_prepare_skipped_retries() {
    local round=$1
    ROUND_NUM="$round" BASE_ROUTES_ENV="$BASE_ROUTES" BASE_CKPT_ENV="$BASE_CHECKPOINT_ENDPOINT" SAVE_PATH_ENV="$SAVE_PATH" \
    python3 << 'PYEOF'
import json, sys, os
import xml.etree.ElementTree as ET

round_num = int(os.environ['ROUND_NUM'])
base_routes = os.environ['BASE_ROUTES_ENV']
base_ckpt = os.environ['BASE_CKPT_ENV']
save_path = os.environ['SAVE_PATH_ENV']

retry_tasks = []

for task_id in range(16):
    ckpt_path = f"{base_ckpt}_{task_id}.json"
    routes_path = f"{base_routes}_{task_id}.xml"

    if not os.path.exists(ckpt_path):
        continue

    with open(ckpt_path) as f:
        data = json.load(f)

    records = data.get('_checkpoint', {}).get('records', [])

    # Find skipped route IDs
    skipped_route_ids = set()
    for rec in records:
        if not rec:
            continue
        meta = rec.get('meta', {})
        if meta.get('skipped') or 'CARLA crashed' in rec.get('status', ''):
            rid = rec.get('route_id', '')
            parts = rid.replace('RouteScenario_', '').replace('_rep0', '')
            if parts:
                skipped_route_ids.add(parts)

    if not skipped_route_ids:
        continue

    # Parse original XML and extract only skipped routes
    tree = ET.parse(routes_path)
    root = tree.getroot()
    all_routes = root.findall('route')

    retry_routes = [r for r in all_routes if r.get('id') in skipped_route_ids]
    if not retry_routes:
        continue

    # Write retry XML
    retry_xml_path = os.path.join(save_path, f"retry_round{round_num}_task{task_id}.xml")
    new_root = ET.Element('routes')
    for r in retry_routes:
        new_root.append(r)
    ET.ElementTree(new_root).write(retry_xml_path, xml_declaration=True, encoding='utf-8')

    # Create fresh checkpoint for retry
    retry_ckpt_path = f"{base_ckpt}_retry{round_num}_{task_id}.json"
    fresh_ckpt = {
        "_checkpoint": {
            "global_record": {},
            "progress": [0, len(retry_routes)],
            "records": []
        }
    }
    with open(retry_ckpt_path, 'w') as f:
        json.dump(fresh_ckpt, f, indent=2)

    print(f"{task_id}|{len(retry_routes)}|{retry_xml_path}|{retry_ckpt_path}")
    retry_tasks.append(task_id)

if not retry_tasks:
    sys.exit(1)
PYEOF
}

# ──────────────────────────────────────────────
# Merge retry results back into original checkpoint
# ──────────────────────────────────────────────
merge_retry_results() {
    local round=$1
    ROUND_NUM="$round" BASE_CKPT_ENV="$BASE_CHECKPOINT_ENDPOINT" \
    python3 << 'PYEOF'
import json, sys, os

round_num = int(os.environ['ROUND_NUM'])
base_ckpt = os.environ['BASE_CKPT_ENV']

for task_id in range(16):
    orig_path = f"{base_ckpt}_{task_id}.json"
    retry_path = f"{base_ckpt}_retry{round_num}_{task_id}.json"

    if not os.path.exists(retry_path) or not os.path.exists(orig_path):
        continue

    with open(orig_path) as f:
        orig = json.load(f)
    with open(retry_path) as f:
        retry = json.load(f)

    retry_records = retry.get('_checkpoint', {}).get('records', [])
    if not retry_records:
        continue

    retry_map = {}
    for rec in retry_records:
        if rec and rec.get('route_id'):
            retry_map[rec['route_id']] = rec

    replaced = 0
    orig_records = orig.get('_checkpoint', {}).get('records', [])
    for i, rec in enumerate(orig_records):
        if not rec:
            continue
        rid = rec.get('route_id', '')
        meta = rec.get('meta', {})
        if rid in retry_map and (meta.get('skipped') or 'CARLA crashed' in rec.get('status', '')):
            new_rec = retry_map[rid]
            if 'CARLA crashed' not in new_rec.get('status', ''):
                orig_records[i] = new_rec
                replaced += 1

    orig['_checkpoint']['records'] = orig_records
    with open(orig_path, 'w') as f:
        json.dump(orig, f, indent=2)

    if replaced > 0:
        print(f"  task={task_id}: merged {replaced} retry results into original checkpoint")
PYEOF
}

# ──────────────────────────────────────────────
# Main: split 16 tasks into batches of 3
# ──────────────────────────────────────────────
ALL_TASKS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
BATCH_SIZE=${#AVAILABLE_GPUS[@]}

log "=========================================="
log "Starting full small-map evaluation"
log "Tasks: ${ALL_TASKS[*]}"
log "GPUs:  ${AVAILABLE_GPUS[*]}"
log "Batch size: $BATCH_SIZE"
log "Total batches: $(( (${#ALL_TASKS[@]} + BATCH_SIZE - 1) / BATCH_SIZE ))"
log "=========================================="

# ── Round 0: initial full evaluation ──
for ((start=0; start<${#ALL_TASKS[@]}; start+=BATCH_SIZE)); do
    batch=("${ALL_TASKS[@]:$start:$BATCH_SIZE}")
    run_batch "${batch[@]}"
done

log "=========================================="
log "Initial evaluation complete. Checking for skipped routes..."
log "=========================================="

# ── Retry rounds: re-evaluate skipped routes ──
for ((round=1; round<=MAX_SKIP_RETRY_ROUNDS; round++)); do
    log "=========================================="
    log "SKIP RETRY ROUND $round / $MAX_SKIP_RETRY_ROUNDS"
    log "=========================================="

    # Find tasks with skipped routes and prepare retry files
    retry_info=$(find_and_prepare_skipped_retries $round 2>/dev/null)
    if [ $? -ne 0 ] || [ -z "$retry_info" ]; then
        log "No skipped routes found. All routes evaluated successfully!"
        break
    fi

    # Parse retry info and collect tasks to retry
    declare -A RETRY_ROUTES_MAP   # task_id -> retry xml path
    declare -A RETRY_CKPT_MAP     # task_id -> retry checkpoint path
    RETRY_TASKS=()

    while IFS='|' read -r task_id n_routes retry_xml retry_ckpt; do
        log "  task=$task_id: $n_routes skipped routes to retry"
        RETRY_ROUTES_MAP[$task_id]=$retry_xml
        RETRY_CKPT_MAP[$task_id]=$retry_ckpt
        RETRY_TASKS+=($task_id)
    done <<< "$retry_info"

    log "Retrying ${#RETRY_TASKS[@]} tasks: ${RETRY_TASKS[*]}"

    # Override routes_for_task for retry batches
    # We run retry tasks through run_batch, but need custom routes/checkpoint
    # So we launch them manually in batches
    for ((start=0; start<${#RETRY_TASKS[@]}; start+=BATCH_SIZE)); do
        batch_slice=("${RETRY_TASKS[@]:$start:$BATCH_SIZE}")

        declare -A BATCH_PIDS
        declare -A BATCH_RETRIES
        declare -A BATCH_GPUS

        log "------------------------------------------"
        log "RETRY BATCH: tasks ${batch_slice[*]}"
        log "------------------------------------------"

        for i in "${!batch_slice[@]}"; do
            local_task_id=${batch_slice[$i]}
            local_gpu=${AVAILABLE_GPUS[$i]}
            BATCH_GPUS[$i]=$local_gpu
            BATCH_RETRIES[$i]=0

            local_port=$(port_for_task $local_task_id)
            local_tm_port=$(tm_port_for_task $local_task_id)
            local_routes=${RETRY_ROUTES_MAP[$local_task_id]}
            local_checkpoint=${RETRY_CKPT_MAP[$local_task_id]}
            local_task_log="${BASE_CHECKPOINT_ENDPOINT}_retry${round}_${local_task_id}.log"

            log "LAUNCH retry task=$local_task_id gpu=$local_gpu routes=$(basename $local_routes)"
            cleanup_gpu $local_gpu

            bash bench2drive/leaderboard/scripts/run_evaluation.sh \
                $local_port $local_tm_port $IS_BENCH2DRIVE $local_routes $TEAM_AGENT "$TEAM_CONFIG" \
                $local_checkpoint $SAVE_PATH $PLANNER_TYPE $local_gpu \
                > "$local_task_log" 2>&1 &

            BATCH_PIDS[$i]=$!
            log "  PID=${BATCH_PIDS[$i]}"
            sleep 10
        done

        # Simple monitor for retry batch
        while true; do
            sleep $MONITOR_INTERVAL
            local_all_done=true

            for i in "${!batch_slice[@]}"; do
                local_pid=${BATCH_PIDS[$i]}
                [ -z "$local_pid" ] && continue

                local_task_id=${batch_slice[$i]}
                local_gpu=${BATCH_GPUS[$i]}
                local_eval_alive=false
                local_carla_alive=false
                is_eval_alive "$local_pid" && local_eval_alive=true
                is_carla_alive "$local_gpu" && local_carla_alive=true

                # Orphan detection
                if [ "$local_eval_alive" = true ] && [ "$local_carla_alive" = false ]; then
                    kill_orphan_eval "$local_pid" "$local_task_id"
                    local_eval_alive=false
                elif [ "$local_eval_alive" = false ] && [ "$local_carla_alive" = true ]; then
                    kill_orphan_carla "$local_gpu" "$local_task_id"
                fi

                if [ "$local_eval_alive" = true ]; then
                    local_all_done=false
                else
                    wait "$local_pid" 2>/dev/null
                    local_exit_code=$?
                    is_carla_alive "$local_gpu" && cleanup_gpu "$local_gpu"
                    if [ $local_exit_code -eq 0 ]; then
                        log "DONE retry task=$local_task_id"
                    else
                        log "FAILED retry task=$local_task_id (exit=$local_exit_code)"
                    fi
                    BATCH_PIDS[$i]=""
                fi
            done

            [ "$local_all_done" = true ] && break
        done

        log "RETRY BATCH COMPLETE: tasks ${batch_slice[*]}"
    done

    # Merge retry results back into original checkpoints
    log "Merging retry round $round results..."
    merge_retry_results $round 2>&1 | while read line; do log "$line"; done

    unset RETRY_ROUTES_MAP RETRY_CKPT_MAP RETRY_TASKS
done

log "=========================================="
log "ALL EVALUATION COMPLETE (including skip retries)"
log "Results in: $SAVE_PATH"
log "=========================================="
