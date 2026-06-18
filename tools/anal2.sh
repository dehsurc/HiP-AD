#!/usr/bin/env bash
# GPU 1 / ckpt 3ep / num_batches=2000 (b2000 yaml).
# Variants: raw + pcgrad_strict + pcgrad_normpreserved (see yaml `probe.variants`).
# Output dir bumped to *_pcgrad so 5/25 (raw,normalized) data is preserved.
set -e
cd /home/yongjae/e2e/HiP-AD-pcgrad
mkdir -p gradient_analysis_results
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  PYTHONPATH=. /home/yongjae/miniconda3/envs/hipad/bin/python tools/run_gradient_analysis.py \
  --config configs/gradient_analysis_b2000.yaml --checkpoints 3ep \
  --output-root gradient_analysis_results/inter_gnn_b1000_pcgrad \
  --modules M2,M3,M_PI \
  --probe-layers dec0_inter_gnn_0,dec1_inter_gnn_0,dec2_inter_gnn_0 \
  --no-supplementary \
  2>&1 | tee gradient_analysis_results/inter_gnn_b1000_pcgrad_3ep.log
