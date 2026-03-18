"""Cache BEVFusion teacher outputs for distillation.

Run this script in the BEVFusion environment to pre-compute teacher
predictions for all training samples. The cached outputs are loaded
during HiP-AD distillation training.

Usage:
    cd /home/yongjae/e2e/bevfusion
    torchpack dist-run -np 4 python /home/yongjae/e2e/HiP-AD/tools/cache_teacher_outputs.py \
        configs/bench2drive/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml \
        --checkpoint runs/stage/latest.pth \
        --out /home/yongjae/e2e/HiP-AD/data/teacher_cache
"""

import argparse
import os
import pickle

import torch
import numpy as np
from tqdm import tqdm

from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval
from mmcv import Config
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDistributedDataParallel
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache BEVFusion teacher outputs for distillation"
    )
    parser.add_argument("config", help="BEVFusion config file path")
    parser.add_argument("--checkpoint", required=True, help="BEVFusion checkpoint path")
    parser.add_argument("--out", required=True, help="Output directory for cached outputs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    rank, world_size = get_dist_info()
    os.makedirs(args.out, exist_ok=True)

    # Load BEVFusion config
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # Skip backbone pretrained init since we load full checkpoint below
    cfg.model.encoders.camera.backbone.init_cfg = None

    # Build model (following BEVFusion test.py pattern)
    cfg.model.train_cfg = None
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    # Wrap in MMDistributedDataParallel (handles DataContainer scattering)
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )
    model.eval()

    # Build dataset (training set without augmentation - use test pipeline)
    cfg.data.test.ann_file = cfg.data.train.dataset.ann_file  # use train annotations
    dataset = build_dataset(cfg.data.test)

    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=args.workers,
        dist=True,
        shuffle=False,
    )

    # Cache outputs
    all_outputs = {}
    if rank == 0:
        print(f"Caching teacher outputs for {len(dataset)} samples...")

    iterator = tqdm(dataloader) if rank == 0 else dataloader

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(iterator):
            # Use model(**data) — MMDDP handles DataContainer unwrapping & GPU placement
            # We need raw head outputs, so we call the inner model's methods
            inner = model.module

            # Standard forward handles DataContainer via mmcv scatter
            # But we need intermediate outputs, so use the standard forward first
            # to let MMDDP scatter data, then access internal state
            results = model(return_loss=False, rescale=True, **batch_data)

            # Unfortunately, the standard forward only returns post-processed results.
            # We need to re-run the head to get raw predictions.
            # But the features are computed during forward — we can hook into the model.
            # For now, let's just cache the post-processed detection results.
            metas = batch_data.get("metas", batch_data.get("img_metas"))
            if hasattr(metas, "data"):
                metas = metas.data[0]

            for i, result in enumerate(results):
                if isinstance(metas, list) and len(metas) > i:
                    token = metas[i].get("token", f"batch{batch_idx}_sample{i}")
                else:
                    token = f"batch{batch_idx}_sample{i}"

                # Cache detection results (boxes, scores, labels)
                sample_output = {}
                for key, val in result.items():
                    if isinstance(val, torch.Tensor):
                        sample_output[key] = val.cpu().half()
                    elif isinstance(val, np.ndarray):
                        sample_output[key] = torch.from_numpy(val).half()
                    else:
                        sample_output[key] = val
                all_outputs[token] = sample_output

            # Save periodically to avoid memory issues
            if (batch_idx + 1) % 500 == 0:
                save_path = os.path.join(args.out, f"teacher_cache_rank{rank}_part_{batch_idx}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(all_outputs, f)
                if rank == 0:
                    print(f"Saved {len(all_outputs)} samples to {save_path}")

    # Final save (per-rank to avoid conflicts)
    save_path = os.path.join(args.out, f"teacher_cache_rank{rank}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_outputs, f)
    print(f"[Rank {rank}] Done! Cached {len(all_outputs)} samples to {save_path}")


if __name__ == "__main__":
    main()
