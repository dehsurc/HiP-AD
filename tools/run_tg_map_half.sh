#!/usr/bin/env bash
# Run TG experiment with map loss weight = 0.5x
#
# Usage:
#   bash tools/run_tg_map_half.sh              # Full 9ep run (~1h)
#   SMOKE=1 bash tools/run_tg_map_half.sh      # Smoke test (~5min)
#   GPUS=2 bash tools/run_tg_map_half.sh       # Use 2 GPUs
#
# Output:
#   work_dirs/exp/stage2_tg_map_half/tg_map_half/iter_15822.pth
#   work_dirs/exp/stage2_tg_map_half/eval/tg_map_half.log
set -e -o pipefail
cd /home/yongjae/e2e/HiP-AD-pcgrad
export PATH=/home/yongjae/miniconda3/envs/hipad/bin:$PATH

GPUS=${GPUS:-4}
PORT=${PORT:-29700}
SEED=${SEED:-0}
SMOKE=${SMOKE:-0}

# Use 9ep checkpoint (where gradient dynamics are most active)
CKPT=${CKPT:-/home/yongjae/e2e/HiP-AD/work_dirs/E2_rev/iter_15822.pth}
CKPT_TAG="9ep"

CONFIG="projects/configs/stage2_tg/tg_map_half.py"
TG_ROOT="work_dirs/exp/stage2_tg_map_half"
WORK="$TG_ROOT/tg_map_half"
EVAL_DIR="$TG_ROOT/eval"

mkdir -p "$WORK" "$EVAL_DIR"

# Training iterations (1 epoch = 1758 iters for 4 GPUs x batch 4)
FINAL_ITER=1758
TRAIN_EXTRA="load_from=$CKPT"

if [[ "$SMOKE" == "1" ]]; then
    FINAL_ITER=30
    TRAIN_EXTRA="$TRAIN_EXTRA runner.max_iters=30 lr_config.warmup_iters=10 checkpoint_config.interval=30 log_config.interval=5"
    echo "[run_tg_map_half] SMOKE mode: max_iters=$FINAL_ITER"
fi

OUT_CKPT="$WORK/iter_${FINAL_ITER}.pth"

echo "=============================================="
echo "TG Map Half (lambda_map = 0.5x) Experiment"
echo "=============================================="
echo "Config:     $CONFIG"
echo "Checkpoint: $CKPT (tag=$CKPT_TAG)"
echo "Output:     $WORK"
echo "GPUs:       $GPUS"
echo "=============================================="

# --- Training ---
if [[ -f "$OUT_CKPT" ]]; then
    echo "[train] $OUT_CKPT exists, skipping training"
else
    echo "[train] Starting training..."
    PORT=$PORT bash tools/dist_train.sh "$CONFIG" "$GPUS" \
        --no-validate --seed "$SEED" --deterministic \
        --work-dir "$WORK" \
        --cfg-options $TRAIN_EXTRA \
        2>&1 | tee "$TG_ROOT/tg_map_half_train.log"
fi

# --- Evaluation ---
EVAL_LOG="$EVAL_DIR/tg_map_half.log"
if grep -q "EVAL_DONE" "$EVAL_LOG" 2>/dev/null; then
    echo "[eval] Cached, skipping"
else
    echo "[eval] Running evaluation..."
    PORT=$((PORT + 100)) bash tools/dist_test.sh "$CONFIG" "$OUT_CKPT" "$GPUS" \
        --cfg-options work_dir="$EVAL_DIR/artifacts_tg_map_half" \
        > "$EVAL_LOG" 2>&1
    echo "EVAL_DONE" >> "$EVAL_LOG"
fi

# --- Extract Results ---
echo ""
echo "=============================================="
echo "Results"
echo "=============================================="
# Extract planning metrics from eval log
if [[ -f "$EVAL_LOG" ]]; then
    echo "Planning metrics from $EVAL_LOG:"
    grep -E "(L2|collision|obj_box_col)" "$EVAL_LOG" | tail -10 || echo "(parsing pending)"
fi

echo ""
echo "Compare with baseline (tg_full from 9ep):"
echo "  tg_full L2:     0.6457"
echo "  tg_no_map L2:   0.6124 (best)"
echo ""
echo "If tg_map_half L2 < 0.6457, reducing map weight helps planning!"
echo "=============================================="
