mkdir -p gradient_analysis_results
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  PYTHONPATH=. /home/yongjae/miniconda3/envs/hipad/bin/python tools/run_gradient_analysis.py \
  --config configs/gradient_analysis.yaml --checkpoints 1ep,3ep,6ep,18ep \
  --output-root gradient_analysis_results/v3 \
  --modules M2,M3,M4,M5,M6,M7,M_N1,M_N2,M_Q \
  --probe-layers dec0_norm_0,dec0_norm_1,dec1_norm_0,dec1_norm_1,dec2_norm_0,dec2_norm_1,dec3_norm_0,dec3_norm_1,dec4_norm_0,dec4_norm_1,dec5_norm_0,dec5_norm_1,dec0_ffn_0_pre_norm,dec1_ffn_0_pre_norm,dec2_ffn_0_pre_norm,dec3_ffn_0_pre_norm,dec4_ffn_0_pre_norm,dec5_ffn_0_pre_norm \
  --no-supplementary \
  2>&1 | tee gradient_analysis_results/tier1_shared.log