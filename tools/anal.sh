CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  PYTHONPATH=. /home/yongjae/miniconda3/envs/hipad/bin/python tools/run_gradient_analysis.py \
  --config configs/gradient_analysis.yaml --checkpoints 1ep,3ep,6ep,18ep \
  --output-root gradient_analysis_results/tier1 \
  --probe-layers dec0_ffn_0_fc1,dec1_ffn_0_fc1,dec2_ffn_0_fc1,dec3_ffn_0_fc1,dec4_ffn_0_fc1,dec5_ffn_0_fc1,fc_before,fc_after \
  --no-supplementary \
  2>&1 | tee gradient_analysis_results/tier1_shared.log