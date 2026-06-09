"""Analyze gradient conflict between real GT loss and pseudo GT loss in pgt_plus mode.

Two analyses:
1. Target mismatch: How different are the Hungarian matching results for real GT vs pseudo GT?
2. Gradient cosine similarity: Do the two losses push shared parameters in opposite directions?

Usage:
    python tools/analyze_grad_conflict.py \
        work_dirs/hipad_nusc_stage2_distill_pgt_plus/hipad_nusc_stage2_distill_pgt_plus.py \
        work_dirs/hipad_nusc_stage2_distill_pgt_plus/iter_2344.pth \
        --num-batches 10
"""
import argparse
import sys
import os
import torch
import numpy as np
from collections import defaultdict

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def get_param_groups(head):
    """Define parameter groups to analyze."""
    groups = {}
    # det_refine layers (direct conflict site)
    for i, refine in enumerate(head.det_refine):
        groups[f"det_refine[{i}].cls_layers"] = list(refine.cls_layers.parameters())
        groups[f"det_refine[{i}].layers"] = list(refine.layers.parameters())
        if hasattr(refine, "quality_layers"):
            groups[f"det_refine[{i}].quality_layers"] = list(refine.quality_layers.parameters())

    # Upstream shared modules
    if hasattr(head, "det_anchor_encoder"):
        groups["det_anchor_encoder"] = list(head.det_anchor_encoder.parameters())
    for i, deform in enumerate(head.det_deformable):
        groups[f"det_deformable[{i}]"] = list(deform.parameters())

    return groups


def collect_grads(params):
    """Flatten all gradients in a param group into a single vector."""
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads)


def compute_metrics(grad_a, grad_b):
    """Compute conflict metrics between two gradient vectors."""
    # Cosine similarity
    cos = torch.nn.functional.cosine_similarity(grad_a.unsqueeze(0), grad_b.unsqueeze(0)).item()

    # Sign agreement ratio
    nonzero = (grad_a != 0) & (grad_b != 0)
    if nonzero.sum() > 0:
        sign_agree = ((grad_a[nonzero].sign() == grad_b[nonzero].sign()).float().mean().item())
    else:
        sign_agree = float("nan")

    # Gradient magnitude ratio
    mag_a = grad_a.norm().item()
    mag_b = grad_b.norm().item()

    # Dot product (scalar interference)
    dot = torch.dot(grad_a, grad_b).item()

    return {
        "cosine": cos,
        "sign_agree": sign_agree,
        "mag_gt": mag_a,
        "mag_pgt": mag_b,
        "mag_ratio": mag_a / (mag_b + 1e-8),
        "dot": dot,
    }


# ============================================================
# Analysis 1: Target mismatch
# ============================================================
def analyze_target_mismatch(head, det_output, data):
    """Compare Hungarian matching results between real GT and pseudo GT."""
    cls_scores = det_output["classification"]
    reg_preds = det_output["prediction"]

    # Use last decoder layer (distill_last_layer_only=True)
    cls = cls_scores[-1]
    reg = reg_preds[-1][..., :len(head.det_reg_weights)]

    # Real GT matching
    gt_labels = data["gt_labels_3d"]
    gt_bboxes = data["gt_bboxes_3d"]
    head.det_sampler.sample(cls, reg, gt_labels, gt_bboxes)
    real_indices = [(idx[0].clone(), idx[1].clone()) if idx[0] is not None else (None, None)
                    for idx in head.det_sampler.indices]

    # Pseudo GT matching
    pgt_labels, pgt_bboxes = head._build_pseudo_gt(data)
    head.det_sampler.sample(cls, reg, pgt_labels, pgt_bboxes)
    pgt_indices = [(idx[0].clone(), idx[1].clone()) if idx[0] is not None else (None, None)
                   for idx in head.det_sampler.indices]

    # Restore real GT indices
    head.det_sampler.sample(cls, reg, gt_labels, gt_bboxes)

    # Compare: which query indices are assigned as positive in each?
    results = []
    bs = cls.shape[0]
    for b in range(bs):
        real_pred_idx, real_gt_idx = real_indices[b]
        pgt_pred_idx, pgt_gt_idx = pgt_indices[b]

        if real_pred_idx is None or pgt_pred_idx is None:
            continue

        real_set = set(real_pred_idx.cpu().numpy().tolist())
        pgt_set = set(pgt_pred_idx.cpu().numpy().tolist())

        # Overlap: queries that are positive in both matchings
        overlap = real_set & pgt_set
        only_real = real_set - pgt_set
        only_pgt = pgt_set - real_set

        # For overlapping queries, check if they got the same class
        cls_match = 0
        cls_total = 0
        for q in overlap:
            real_pos = (real_pred_idx == q).nonzero(as_tuple=True)[0][0]
            pgt_pos = (pgt_pred_idx == q).nonzero(as_tuple=True)[0][0]
            real_cls = gt_labels[b][real_gt_idx[real_pos]].item()
            pgt_cls = pgt_labels[b][pgt_gt_idx[pgt_pos]].item()
            cls_total += 1
            if real_cls == pgt_cls:
                cls_match += 1

        results.append({
            "num_real_gt": len(gt_labels[b]),
            "num_pgt": len(pgt_labels[b]),
            "num_real_pos": len(real_set),
            "num_pgt_pos": len(pgt_set),
            "overlap": len(overlap),
            "only_real": len(only_real),
            "only_pgt": len(only_pgt),
            "cls_match": cls_match,
            "cls_total": cls_total,
        })

    return results


# ============================================================
# Analysis 2: Gradient cosine similarity
# ============================================================
def analyze_grad_conflict(model, head, det_output, data, param_groups):
    """Compute separate gradients for real GT loss and pseudo GT loss, then compare."""
    cls_scores = det_output["classification"]
    reg_preds = det_output["prediction"]
    quality = det_output["quality"]

    decoder_idx = len(cls_scores) - 1  # last layer (distill_last_layer_only=True)
    cls = cls_scores[decoder_idx]
    reg = reg_preds[decoder_idx][..., :len(head.det_reg_weights)]
    qt = quality[decoder_idx]

    gt_labels = data["gt_labels_3d"]
    gt_bboxes = data["gt_bboxes_3d"]
    pgt_labels, pgt_bboxes = head._build_pseudo_gt(data)

    all_params = []
    for params in param_groups.values():
        all_params.extend(params)

    # --- Compute real GT loss gradient ---
    model.zero_grad()
    cls_target, reg_target, reg_weights = head.det_sampler.sample(cls, reg, gt_labels, gt_bboxes)
    reg_target = reg_target[..., :len(head.det_reg_weights)]
    mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))

    from projects.mmdet3d_plugin.models.sparse_onedecoder import reduce_mean
    num_pos = max(reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0)

    cls_flat = cls.flatten(end_dim=1)
    cls_target_flat = cls_target.flatten(end_dim=1)
    gt_cls_loss = head.loss_det_cls(cls_flat, cls_target_flat, avg_factor=num_pos)

    mask_flat = mask.reshape(-1)
    rw = reg_weights * reg.new_tensor(head.det_reg_weights)
    rt = reg_target.flatten(end_dim=1)[mask_flat]
    rp = reg.flatten(end_dim=1)[mask_flat]
    rw_flat = rw.flatten(end_dim=1)[mask_flat]
    rt = torch.where(rt.isnan(), rp.new_tensor(0.0), rt)
    ct = cls_target_flat[mask_flat]
    qt_flat = qt.flatten(end_dim=1)[mask_flat] if qt is not None else None

    gt_reg_loss_dict = head.loss_det_reg(
        rp, rt, weight=rw_flat, avg_factor=num_pos,
        prefix="det_", suffix=f"_{decoder_idx}", quality=qt_flat, cls_target=ct)

    gt_loss = gt_cls_loss
    for v in gt_reg_loss_dict.values():
        gt_loss = gt_loss + v
    gt_loss.backward(retain_graph=True)

    gt_grads = {}
    for name, params in param_groups.items():
        gt_grads[name] = collect_grads(params)

    # --- Compute pseudo GT loss gradient ---
    model.zero_grad()
    pgt_cls_target, pgt_reg_target, pgt_reg_weights = head.det_sampler.sample(
        cls, reg, pgt_labels, pgt_bboxes)
    pgt_reg_target = pgt_reg_target[..., :len(head.det_reg_weights)]
    pgt_mask = torch.logical_not(torch.all(pgt_reg_target == 0, dim=-1))
    pgt_num_pos = max(reduce_mean(torch.sum(pgt_mask).to(dtype=reg.dtype)), 1.0)

    pgt_cls_flat = cls.flatten(end_dim=1)
    pgt_cls_target_flat = pgt_cls_target.flatten(end_dim=1)
    pgt_cls_loss = head.loss_det_cls(pgt_cls_flat, pgt_cls_target_flat, avg_factor=pgt_num_pos)

    pgt_mask_flat = pgt_mask.reshape(-1)
    pgt_rw = pgt_reg_weights * reg.new_tensor(head.det_reg_weights)
    pgt_rt = pgt_reg_target.flatten(end_dim=1)[pgt_mask_flat]
    pgt_rp = reg.flatten(end_dim=1)[pgt_mask_flat]
    pgt_rw_flat = pgt_rw.flatten(end_dim=1)[pgt_mask_flat]
    pgt_rt = torch.where(pgt_rt.isnan(), pgt_rp.new_tensor(0.0), pgt_rt)
    pgt_ct = pgt_cls_target_flat[pgt_mask_flat]
    pgt_qt = qt.flatten(end_dim=1)[pgt_mask_flat] if qt is not None else None

    pgt_reg_loss_dict = head.loss_det_reg(
        pgt_rp, pgt_rt, weight=pgt_rw_flat, avg_factor=pgt_num_pos,
        prefix="det_pgt_", suffix=f"_{decoder_idx}", quality=pgt_qt, cls_target=pgt_ct)

    pgt_loss = pgt_cls_loss
    for v in pgt_reg_loss_dict.values():
        pgt_loss = pgt_loss + v
    pgt_loss.backward(retain_graph=True)

    pgt_grads = {}
    for name, params in param_groups.items():
        pgt_grads[name] = collect_grads(params)

    # Restore
    model.zero_grad()
    head.det_sampler.sample(cls, reg, gt_labels, gt_bboxes)

    # Compare
    metrics = {}
    for name in param_groups:
        metrics[name] = compute_metrics(gt_grads[name], pgt_grads[name])

    return metrics


def main():
    args = parse_args()

    # Build model
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.to(args.device)
    model.eval()

    # Need grad for backward
    for p in model.parameters():
        p.requires_grad_(True)

    # Get the head (SparseOneDecoder)
    head = model.head
    if hasattr(head, "onedecoder_head"):
        head = head.onedecoder_head

    # Build dataloader (use train set since teacher cache is for train)
    cfg.data.train.test_mode = False
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,  # minimize GPU memory
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
    )

    param_groups = get_param_groups(head)
    print(f"\nParameter groups:")
    for name, params in param_groups.items():
        total = sum(p.numel() for p in params)
        print(f"  {name}: {total:,} params")

    # Run analysis
    all_target_results = []
    all_grad_metrics = defaultdict(lambda: defaultdict(list))

    for batch_idx, data in enumerate(data_loader):
        if batch_idx >= args.num_batches:
            break

        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1}/{args.num_batches}")

        # Move data to device
        for k, v in data.items():
            if hasattr(v, "data"):
                # DataContainer
                if isinstance(v.data, list):
                    if isinstance(v.data[0], torch.Tensor):
                        data[k] = v.data[0].to(args.device)
                    else:
                        data[k] = v.data[0]
                elif isinstance(v.data, torch.Tensor):
                    data[k] = v.data.to(args.device)

        # Forward (detection head only)
        with torch.enable_grad():
            # We need the model forward to get det_output
            img = data.get("img")
            if isinstance(img, list):
                img = img[0]
            if hasattr(img, "data"):
                img = img.data
                if isinstance(img, list):
                    img = img[0]
            img = img.to(args.device)

            # Full forward
            det_output, map_output, ego_output, plan_output, motion_output, scenes_output = model.head(
                img_feats=model.extract_feat(img),
                data=data,
            )

        # Analysis 1: Target mismatch (no grad needed, uses same det_output)
        mismatch = analyze_target_mismatch(head, det_output, data)
        all_target_results.extend(mismatch)

        for m in mismatch:
            overlap_rate = m["overlap"] / max(min(m["num_real_pos"], m["num_pgt_pos"]), 1)
            cls_rate = m["cls_match"] / max(m["cls_total"], 1)
            print(f"  [Target] real_gt={m['num_real_gt']}, pgt={m['num_pgt']}, "
                  f"overlap={m['overlap']}/{min(m['num_real_pos'], m['num_pgt_pos'])}"
                  f" ({overlap_rate:.1%}), cls_match={cls_rate:.1%}")

        # Analysis 2: Gradient conflict (reuse same det_output)
        with torch.enable_grad():
            grad_metrics = analyze_grad_conflict(model, head, det_output, data, param_groups)

        for name, metrics in grad_metrics.items():
            for k, v in metrics.items():
                all_grad_metrics[name][k].append(v)

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Target mismatch summary
    print(f"\n--- Target Mismatch ({len(all_target_results)} samples) ---")
    if all_target_results:
        avg_real = np.mean([r["num_real_gt"] for r in all_target_results])
        avg_pgt = np.mean([r["num_pgt"] for r in all_target_results])
        avg_overlap = np.mean([r["overlap"] for r in all_target_results])
        avg_real_pos = np.mean([r["num_real_pos"] for r in all_target_results])
        avg_pgt_pos = np.mean([r["num_pgt_pos"] for r in all_target_results])
        total_cls_match = sum(r["cls_match"] for r in all_target_results)
        total_cls_total = sum(r["cls_total"] for r in all_target_results)
        cls_rate = total_cls_match / max(total_cls_total, 1)

        print(f"  Avg real GT objects: {avg_real:.1f}")
        print(f"  Avg pseudo GT objects: {avg_pgt:.1f}")
        print(f"  Avg positive queries (real): {avg_real_pos:.1f}")
        print(f"  Avg positive queries (pgt): {avg_pgt_pos:.1f}")
        print(f"  Avg query overlap: {avg_overlap:.1f}")
        print(f"  Class agreement (overlapping queries): {cls_rate:.1%} ({total_cls_match}/{total_cls_total})")

    # Gradient conflict summary
    print(f"\n--- Gradient Conflict (per parameter group) ---")
    print(f"{'Group':<35} {'Cosine':>8} {'SignAgr':>8} {'|g_gt|':>10} {'|g_pgt|':>10} {'Dot':>12}")
    print("-" * 85)
    for name in sorted(all_grad_metrics.keys()):
        m = all_grad_metrics[name]
        cos = np.mean(m["cosine"])
        sign = np.mean(m["sign_agree"])
        mag_gt = np.mean(m["mag_gt"])
        mag_pgt = np.mean(m["mag_pgt"])
        dot = np.mean(m["dot"])
        print(f"{name:<35} {cos:>8.4f} {sign:>8.4f} {mag_gt:>10.4f} {mag_pgt:>10.4f} {dot:>12.4f}")

    # Interpretation
    print(f"\n--- Interpretation ---")
    conflict_groups = []
    aligned_groups = []
    for name in sorted(all_grad_metrics.keys()):
        cos = np.mean(all_grad_metrics[name]["cosine"])
        if cos < 0:
            conflict_groups.append((name, cos))
        elif cos > 0.5:
            aligned_groups.append((name, cos))

    if conflict_groups:
        print("  CONFLICT detected (cosine < 0):")
        for name, cos in conflict_groups:
            print(f"    {name}: cosine = {cos:.4f}")
    else:
        print("  No clear gradient conflict detected (all cosine >= 0)")

    if aligned_groups:
        print("  Well-aligned groups (cosine > 0.5):")
        for name, cos in aligned_groups:
            print(f"    {name}: cosine = {cos:.4f}")


if __name__ == "__main__":
    main()
