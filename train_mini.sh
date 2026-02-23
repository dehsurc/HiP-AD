#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

# Quick smoke test for nuScenes mini (dataloader + forward pass)
CONFIG="projects/configs/hipad_nusc_stage2.py"
WORK_DIR="work_dirs/debug_nusc_mini"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH+:$PYTHONPATH}"

python tools/train.py "$CONFIG" \
  --work-dir "$WORK_DIR" \
  --gpus 1 \
  --no-validate \
  --cfg-options \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=1 \
    runner.max_iters=2 \
    log_config.interval=1 \
    data.train.ann_file=/data/infos/mini/nuscenes_infos_train.pkl \
    data.val.ann_file=/data/infos/mini/nuscenes_infos_val.pkl \
    data.test.ann_file=/data/infos/mini/nuscenes_infos_val.pkl \
    data.train.version=v1.0-mini \
    data.val.version=v1.0-mini \
    data.test.version=v1.0-mini \
    eval_config.version=v1.0-mini \
    eval_config.ann_file=/data/infos/mini/nuscenes_infos_val.pkl \
    data.train.data_root=/data/nuscenes/ \
    data.val.data_root=/data/nuscenes/ \
    data.test.data_root=/data/nuscenes/
