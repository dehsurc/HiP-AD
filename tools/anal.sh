mkdir -p gradient_analysis_results
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  PYTHONPATH=. /home/yongjae/miniconda3/envs/hipad/bin/python tools/run_gradient_analysis.py \
  --config configs/gradient_analysis.yaml --checkpoints 1ep,3ep,9ep,18ep \
  --output-root gradient_analysis_results/inter_gnn_v1 \
  --modules M2,M3,M_PI \
  --probe-layers dec0_inter_gnn_0,dec1_inter_gnn_0,dec2_inter_gnn_0,dec3_inter_gnn_0,dec4_inter_gnn_0,dec5_inter_gnn_0 \
  --no-supplementary \
  2>&1 | tee gradient_analysis_results/inter_gnn_v1.log
