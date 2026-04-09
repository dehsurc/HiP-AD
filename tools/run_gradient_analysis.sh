#!/bin/bash
# Gradient Conflict Analysis for HiP-AD

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CUDA_VISIBLE_DEVICES=2 python tools/analyze_gradient_conflict.py \
    --config projects/configs/hipad_b2d_stage2.py \
    --checkpoint ckpts/code_epoch1.pth \
    --num-batches 128 \
    --num-splits 2 \
    --analysis-mode full \
    --random-baseline \
    --output-dir gradient_analysis_results/e1 \
    "$@"

CUDA_VISIBLE_DEVICES=2 python tools/analyze_gradient_conflict.py \
    --config projects/configs/hipad_b2d_stage2.py \
    --checkpoint ckpts/code_epoch2.pth \
    --num-batches 128 \
    --num-splits 2 \
    --analysis-mode full \
    --random-baseline \
    --output-dir gradient_analysis_results/e2 \
    "$@"

CUDA_VISIBLE_DEVICES=2 python tools/analyze_gradient_conflict.py \
    --config projects/configs/hipad_b2d_stage2.py \
    --checkpoint ckpts/code_epoch3.pth \
    --num-batches 128 \
    --num-splits 2 \
    --analysis-mode full \
    --random-baseline \
    --output-dir gradient_analysis_results/e3 \
    "$@"

CUDA_VISIBLE_DEVICES=2 python tools/analyze_gradient_conflict.py \
    --config projects/configs/hipad_b2d_stage2.py \
    --checkpoint ckpts/code_epoch18.pth \
    --num-batches 128 \
    --num-splits 2 \
    --analysis-mode full \
    --random-baseline \
    --output-dir gradient_analysis_results/e18 \
    "$@"
