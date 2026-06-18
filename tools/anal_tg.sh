#!/usr/bin/env bash
# Transfer-Gain (leave-one-aux-out) driver — 2026-06-12
#
# For each variant in projects/configs/stage2_tg/ :
#   1) 1-epoch fine-tune from the converged 18ep ckpt (iter_31644),
#      one task's loss gradients cut via SparseDetector.ablate_tasks
#   2) planning eval (tools/test.py) on nuScenes val
# Then evaluates the UN-finetuned iter_31644 once (pretrain_ref) and
# aggregates TG(task) = P(no_task) - P(tg_full) into a CSV + report.
#
# Usage:
#   bash tools/anal_tg.sh                # full run (4 GPUs, ~4-5 h)
#   SMOKE=1 bash tools/anal_tg.sh        # pipeline smoke (~10 min)
#   VARIANTS="tg_no_det" bash tools/anal_tg.sh   # subset
# Env overrides: GPUS (4), PORT (29610), SEED (0), SMOKE (0)
set -e -o pipefail
cd /home/yongjae/e2e/HiP-AD-pcgrad
# dist_train.sh / dist_test.sh call bare `python3` — point it at the hipad env.
export PATH=/home/yongjae/miniconda3/envs/hipad/bin:$PATH

VARIANTS=${VARIANTS:-"tg_full tg_no_det tg_no_map tg_no_motion tg_no_det_motion"}
GPUS=${GPUS:-4}
PORT_BASE=${PORT:-29610}
SEED=${SEED:-0}
SMOKE=${SMOKE:-0}
# Starting checkpoint for the fine-tune. NOTE: a CONVERGED ckpt (18ep) is
# saturated -> TG ~ 0 by construction; measure from a still-plastic mid ckpt
# (3-9ep) where the aux<->plan dynamics are actually active (see analysis A).
# Sweep example: for t in 3ep 9ep 18ep; do CKPT_TAG=$t CKPT=.../iter_<n>.pth bash tools/anal_tg.sh; done
CKPT=${CKPT:-/home/yongjae/e2e/HiP-AD/work_dirs/E2_rev/iter_31644.pth}
CKPT_TAG=${CKPT_TAG:-18ep}
# LR override (empty = config default 2e-5). For mid ckpts use a larger LR so
# 1 epoch actually moves the model (e.g. LR=1e-4), else TG is null by construction.
LR=${LR:-}
PRETRAIN_CKPT=$CKPT

if [[ "$SMOKE" == "1" ]]; then
    TG_ROOT=${TG_ROOT:-work_dirs/exp/stage2_tg_smoke}   # keep real runs clean
else
    TG_ROOT=${TG_ROOT:-work_dirs/exp/stage2_tg_${CKPT_TAG}}
fi
EVAL_DIR=$TG_ROOT/eval
mkdir -p "$EVAL_DIR"
echo "[anal_tg] ckpt=$CKPT (tag=$CKPT_TAG)  lr=${LR:-config-default}  root=$TG_ROOT"

FINAL_ITER=1758
TRAIN_EXTRA=""
EVAL_EXTRA=""
if [[ "$SMOKE" == "1" ]]; then
    FINAL_ITER=30
    TRAIN_EXTRA="runner.max_iters=30 lr_config.warmup_iters=10 checkpoint_config.interval=30 log_config.interval=5"
    # NOTE: do NOT subsample val with load_interval — planning GT needs
    # consecutive future frames (get_ann_info: data_infos[index + i]);
    # subsampling breaks continuity and raises IndexError. Full val it is.
    echo "[anal_tg] SMOKE mode: max_iters=$FINAL_ITER (full val eval)"
fi

# Pin the starting checkpoint (overrides the config's load_from) for every
# variant so a checkpoint sweep just changes CKPT/CKPT_TAG. Optional LR override.
TRAIN_EXTRA="$TRAIN_EXTRA load_from=$CKPT"
[[ -n "$LR" ]] && TRAIN_EXTRA="$TRAIN_EXTRA optimizer.lr=$LR"

run_eval () {  # $1=tag  $2=config  $3=ckpt  $4=port
    local LOG="$EVAL_DIR/$1.log"
    if grep -q "EVAL_DONE" "$LOG" 2>/dev/null; then
        echo "[anal_tg] eval $1 — cached, skip"
        return 0
    fi
    if [[ ! -f "$3" ]]; then
        echo "[anal_tg] ERROR: checkpoint missing for eval $1: $3" >&2
        return 1
    fi
    echo "[anal_tg] eval $1  ckpt=$3  port=$4 -> $LOG"
    # Per-tag work_dir: evaluate() writes results.pkl/results_nusc.json to
    # cfg.work_dir — without this, pretrain_ref (same config as tg_full)
    # would clobber tg_full's artifacts, and SMOKE evals would write into
    # the real run dirs.
    PORT=$4 bash tools/dist_test.sh "$2" "$3" "$GPUS" \
        --cfg-options work_dir="$EVAL_DIR/artifacts_$1" ${EVAL_EXTRA:-} \
        > "$LOG" 2>&1
    echo "EVAL_DONE" >> "$LOG"
}

i=0
for v in $VARIANTS; do
    i=$((i + 1))
    CONFIG=projects/configs/stage2_tg/${v}.py
    WORK=$TG_ROOT/${v}
    OUT_CKPT=$WORK/iter_${FINAL_ITER}.pth   # produced ckpt (distinct from start CKPT)
    PORT_I=$((PORT_BASE + i))

    if [[ -f "$OUT_CKPT" ]]; then
        echo "[anal_tg] train $v — $OUT_CKPT exists, skip"
    else
        echo "[anal_tg] train $v  (gpus=$GPUS port=$PORT_I) -> $WORK"
        PORT=$PORT_I bash tools/dist_train.sh "$CONFIG" "$GPUS" \
            --no-validate --seed "$SEED" --deterministic \
            --work-dir "$WORK" \
            ${TRAIN_EXTRA:+--cfg-options $TRAIN_EXTRA} \
            2>&1 | tee "$TG_ROOT/${v}_train.log"
        rm -f "$EVAL_DIR/${v}.log"   # fresh weights -> stale eval cache invalid
    fi

    run_eval "$v" "$CONFIG" "$OUT_CKPT" $((PORT_BASE + 100 + i))
done

# Un-finetuned reference (drift check: how much does tg_full itself move?)
run_eval "pretrain_ref" "projects/configs/stage2_tg/tg_full.py" \
         "$PRETRAIN_CKPT" $((PORT_BASE + 199))

# --- weight snapshots for later qualitative before/after visualization ---
# AFTER weights are already saved by mmcv (<variant>/iter_<final>.pth, kept via
# checkpoint_config.max_keep_ckpts); BEFORE weights are the shared start ckpt.
# Here we just expose a clean before/after interface (symlinks + manifest.json)
# for a downstream viz site. Pure bookkeeping — does not touch training/eval.
WEIGHTS_DIR=$TG_ROOT/weights
mkdir -p "$WEIGHTS_DIR"
ln -sf "$(realpath "$CKPT")" "$WEIGHTS_DIR/before_${CKPT_TAG}.pth" 2>/dev/null || true
for v in $VARIANTS; do
    after="$TG_ROOT/$v/iter_${FINAL_ITER}.pth"
    [[ -f "$after" ]] && ln -sf "$(realpath "$after")" "$WEIGHTS_DIR/${v}_after.pth"
done
TG_ROOT="$TG_ROOT" CKPT="$CKPT" CKPT_TAG="$CKPT_TAG" LR="${LR:-config-default}" \
FINAL_ITER="$FINAL_ITER" VARIANTS="$VARIANTS" \
    /home/yongjae/miniconda3/envs/vad/bin/python tools/gradient_analysis/write_tg_weights_manifest.py || \
    echo "[anal_tg] WARN: weight manifest step failed (non-fatal)"

echo "[anal_tg] aggregating -> $TG_ROOT/tg_summary.csv"
/home/yongjae/miniconda3/envs/vad/bin/python tools/gradient_analysis/aggregate_tg.py \
    --eval-dir "$EVAL_DIR" --out-dir "$TG_ROOT"
echo "[anal_tg] done."
