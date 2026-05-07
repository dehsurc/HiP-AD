CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$(pwd)" python tools/test.py /home/yongjae/e2e/HiP-AD/work_dirs/exp/E9_E2_E1_stage2_18ep_GN/E9_E2_E1_stage2_18ep_GN.py /home/yongjae/e2e/HiP-AD/work_dirs/exp/E9_E2_E1_stage2_18ep_GN/iter_780.pth --eval bbox
&&
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$(pwd)" python tools/test.py /home/yongjae/e2e/HiP-AD/work_dirs/exp/E9_E2_E1_stage2_18ep_GN/E9_E2_E1_stage2_18ep_GN.py /home/yongjae/e2e/HiP-AD/work_dirs/exp/E9_E2_E1_stage2_18ep_GN/iter_1560.pth --eval bbox
&&
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$(pwd)" python tools/test.py /home/yongjae/e2e/HiP-AD/work_dirs/exp/E9_E2_E1_stage2_18ep_GN/E9_E2_E1_stage2_18ep_GN.py /home/yongjae/e2e/HiP-AD/work_dirs/exp/E9_E2_E1_stage2_18ep_GN/iter_2340.pth --eval bbox
&&
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$(pwd)" python tools/test.py /home/yongjae/e2e/HiP-AD/work_dirs/exp/E9_E2_E1_stage2_18ep_GN/E9_E2_E1_stage2_18ep_GN.py /home/yongjae/e2e/HiP-AD/work_dirs/exp/E9_E2_E1_stage2_18ep_GN/iter_4680.pth --eval bbox