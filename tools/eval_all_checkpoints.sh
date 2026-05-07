#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Evaluate every iter_*.pth checkpoint in a work_dir sequentially.
# Usage:
#   bash tools/eval_all_checkpoints.sh [CONFIG] [WORK_DIR] [GPUS]
# Defaults target E9 GradNorm stage2 on GPU 2,3.
#
# Output:
#   <WORK_DIR>/eval_all/<ckpt>.log    — full stdout/stderr of test run
#   <WORK_DIR>/eval_all/summary.csv   — iter, mAP / NDS / etc. scraped from log
# Env overrides:
#   CUDA_VISIBLE_DEVICES (default "2,3")
#   PORT                 (default 29700, incremented per ckpt)
#   EXTRA_TEST_ARGS      (forwarded to tools/test.py, e.g. "--eval bbox")
# ─────────────────────────────────────────────────────────────
set -u -o pipefail

# ── args & defaults ───────────────────────────────────────────
CONFIG=${1:-projects/configs/experiments/E9_E2_E1_stage2_18ep_GN.py}
WORK_DIR=${2:-work_dirs/exp/E9_E2_E1_stage2_18ep_GN}
GPUS=${3:-2}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
BASE_PORT=${PORT:-29700}
EXTRA_TEST_ARGS=${EXTRA_TEST_ARGS:-"--eval bbox"}

# Resolve paths relative to repo root (where this script is invoked from).
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

[[ -f "$CONFIG"   ]] || { echo "[eval_all] config not found: $CONFIG";   exit 1; }
[[ -d "$WORK_DIR" ]] || { echo "[eval_all] work_dir not found: $WORK_DIR"; exit 1; }

OUT_DIR="$WORK_DIR/eval_all"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.csv"

# ── discover checkpoints (numeric iter sort, skip latest/best symlinks) ──
mapfile -t CKPTS < <(
    find "$WORK_DIR" -maxdepth 1 -type f -name 'iter_*.pth' \
        | awk -F'iter_|\\.pth' '{print $2 "\t" $0}' \
        | sort -n \
        | cut -f2
)

if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "[eval_all] no iter_*.pth under $WORK_DIR"
    exit 1
fi

echo "[eval_all] config    : $CONFIG"
echo "[eval_all] work_dir  : $WORK_DIR"
echo "[eval_all] gpus      : $GPUS (visible: $CUDA_VISIBLE_DEVICES)"
echo "[eval_all] ckpt count: ${#CKPTS[@]}"
echo "[eval_all] out dir   : $OUT_DIR"
echo

# ── summary header ────────────────────────────────────────────
if [[ ! -f "$SUMMARY" ]]; then
    echo "iter,ckpt,status,elapsed_sec,log" > "$SUMMARY"
fi

# ── evaluate each checkpoint sequentially ─────────────────────
idx=0
for CKPT in "${CKPTS[@]}"; do
    idx=$((idx + 1))
    CKPT_BASE="$(basename "$CKPT" .pth)"
    ITER_NUM="${CKPT_BASE#iter_}"
    LOG="$OUT_DIR/${CKPT_BASE}.log"
    PORT_I=$((BASE_PORT + idx))

    # Skip if already evaluated successfully.
    if [[ -f "$LOG" ]] && grep -q "Evaluation finished\|EVAL_DONE" "$LOG" 2>/dev/null; then
        echo "[eval_all] [${idx}/${#CKPTS[@]}] $CKPT_BASE  — already done, skip"
        echo "${ITER_NUM},${CKPT},cached,0,${LOG}" >> "$SUMMARY"
        continue
    fi

    echo "[eval_all] [${idx}/${#CKPTS[@]}] $CKPT_BASE  (port=${PORT_I}) → $LOG"
    t0=$(date +%s)

    PORT=$PORT_I bash tools/dist_test.sh \
        "$CONFIG" "$CKPT" "$GPUS" \
        $EXTRA_TEST_ARGS \
        > "$LOG" 2>&1
    rc=$?

    echo "EVAL_DONE" >> "$LOG"
    t1=$(date +%s)
    elapsed=$((t1 - t0))

    status=$([[ $rc -eq 0 ]] && echo ok || echo "fail(rc=$rc)")
    echo "${ITER_NUM},${CKPT},${status},${elapsed},${LOG}" >> "$SUMMARY"
    echo "[eval_all]   status=${status} elapsed=${elapsed}s"
    echo
done

# ── done ──────────────────────────────────────────────────────
echo "[eval_all] all checkpoints processed."
echo "[eval_all] summary : $SUMMARY"
echo "[eval_all] logs    : $OUT_DIR/*.log"
