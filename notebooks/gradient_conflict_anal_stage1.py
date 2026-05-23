#!/usr/bin/env python
"""
Gradient Conflict Analysis Script for HiP-AD Stage1 (nuScenes)

Stage1 variant of gradient_conflict_anal_nusc.py focused on diagnosing the
effect of detection knowledge-distillation (KD) loss on other tasks (notably
mapping). KD loss is split out as its own task ``det_distill`` so its
conflict / dilution effect on ``det``, ``map``, ``depth`` can be measured
independently.

Stage1 active tasks (verified from config + logs):
  det:          det_loss_{cls, box, cns, yns}
  det_distill:  det_loss_kd_{cls, box, cns, yns}     # only present in E4
  map:          map_loss_{cls, line}
  depth:        loss_dense_depth

Excluded — present in logs but loss_weight == 0 in stage1 config
(see E1_stage1_12ep.py: loss_ego_status / loss_plan_cls / loss_plan_reg
all have loss_weight=0.0; with_motion=False, with_planning=False):
  ego_loss_status, plan_loss_temp_*

Note on prefix matching: ``det_loss_kd_*`` does NOT collide with ``det_loss_cls/box/cns/yns``
under startswith-matching, so no longest-prefix logic is required — we just keep
the two task groups disjoint by listing exact full prefixes.

Two levels of analysis:
  1. Weight gradient: How tasks compete for shared parameter updates (optimizer perspective)
  2. Activation gradient: How tasks want to change shared intermediate representations
     (FPN features, deformable outputs, inter_gnn layers)

Key metrics:
  - Raw cosine similarity: Standard full-vector cosine between task gradients
  - Active-overlap cosine: Weighted cosine on coordinates where both tasks have non-zero
    gradient, distinguishing disjoint / true_orthogonal / hidden_conflict
  - Interference: ||g_src|| * max(0, -cos(g_src, g_vic))
  - Decomposition: Cooperative vs conflicting gradient components

Analysis modes:
  basic     - Overall pairwise cosine similarity across all shared parameters
  full      - Per-layer analysis + advanced metrics + active-overlap
  direction - PCA-based gradient direction visualization + all of 'full'

Usage:
    # E1 baseline (no distill)
    python notebooks/gradient_conflict_anal_stage1.py \
        --config work_dirs/exp/E1_stage1_12ep/hipad_nusc_stage1.py \
        --checkpoint work_dirs/exp/E1_stage1_12ep/latest.pth \
        --num-batches 50 --batch-size 1 --analysis-mode full \
        --output-dir gradient_analysis_results_nusc/E1

    # E4 (det + distill)
    python notebooks/gradient_conflict_anal_stage1.py \
        --config work_dirs/exp/E4_stage1_12ep_distill_w_det/E4_stage1_12ep_distill_w_det.py \
        --checkpoint work_dirs/exp/E4_stage1_12ep_distill_w_det/latest.pth \
        --num-batches 50 --batch-size 1 --analysis-mode full \
        --output-dir gradient_analysis_results_nusc/E4
"""

import os
import csv
import sys
import json
import math
import pickle
import tempfile
import argparse
import contextlib
from functools import partial
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats as sp_stats

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_mmdet

# Import HiP-AD custom modules
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.models.blocks import DeformableFeatureAggregation


# ============================================================================
# Constants
# ============================================================================

# Stage1 active tasks only — ego/plan are in logs but loss_weight=0 (no grad
# contribution) and motion/planning heads are not built (with_motion=False,
# with_planning=False). See module docstring for details.
TASK_GROUPS = {
    'det':         ['det_loss_cls', 'det_loss_box', 'det_loss_cns', 'det_loss_yns'],
    'det_distill': ['det_loss_kd_cls', 'det_loss_kd_box', 'det_loss_kd_cns', 'det_loss_kd_yns'],
    'map':         ['map_loss_cls', 'map_loss_line'],
    'depth':       ['loss_dense_depth'],
}

# Pairs ordered to put the diagnostic-of-interest pairs first.
# (a) det_distill vs map      — does KD's gradient direction conflict with map?
# (b) det_distill vs det      — does KD point the same way as det's own gradient?
# (c) det_distill vs depth    — does KD interfere with backbone-shared depth supervision?
# (d) det vs map / det vs depth / map vs depth — baseline conflicts to compare against E1.
TASK_PAIRS = [
    ('det_distill', 'map'),
    ('det_distill', 'det'),
    ('det_distill', 'depth'),
    ('det', 'map'),
    ('det', 'depth'),
    ('map', 'depth'),
]

TASK_COLORS = {
    'det':         '#e74c3c',
    'det_distill': '#c0392b',
    'map':         '#2ecc71',
    'depth':       '#34495e',
}


# ============================================================================
# Utility Functions
# ============================================================================

@contextlib.contextmanager
def _selective_eval(model):
    """Forward dispatch는 train 유지, Dropout/BN/DeformableFeatureAggregation만 eval로 전환."""
    switched = []
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                          nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm,
                          DeformableFeatureAggregation)):
            if m.training:
                m.eval()
                switched.append(m)
    try:
        yield
    finally:
        for m in switched:
            m.train()


def match_loss_key(loss_key: str, task_prefixes: List[str]) -> bool:
    for prefix in task_prefixes:
        if loss_key.startswith(prefix):
            return True
    return False


def compute_cosine_similarity(grad1: torch.Tensor, grad2: torch.Tensor) -> float:
    if grad1 is None or grad2 is None:
        return float('nan')
    norm1 = torch.norm(grad1)
    norm2 = torch.norm(grad2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return float('nan')
    return (torch.dot(grad1, grad2) / (norm1 * norm2)).item()


def compute_active_overlap_cosine(
    grad1: torch.Tensor, grad2: torch.Tensor, quantile: float = 0.99,
) -> Dict[str, float]:
    """Compute active-overlap weighted cosine similarity."""
    if grad1 is None or grad2 is None:
        return {'overlap_cosine': float('nan'), 'overlap_ratio': 0.0,
                'raw_cosine': float('nan'), 'interpretation': 'N/A'}

    raw_cos = compute_cosine_similarity(grad1, grad2)

    abs1 = grad1.abs()
    abs2 = grad2.abs()
    _MAX_QUANTILE_ELEMS = 2_000_000
    if abs1.numel() > _MAX_QUANTILE_ELEMS:
        idx = torch.randint(abs1.numel(), (_MAX_QUANTILE_ELEMS,), device=abs1.device)
        q1 = abs1.view(-1)[idx].quantile(quantile) + 1e-8
        q2 = abs2.view(-1)[idx].quantile(quantile) + 1e-8
    else:
        q1 = abs1.quantile(quantile) + 1e-8
        q2 = abs2.quantile(quantile) + 1e-8
    norm1 = (abs1 / q1).clamp(max=1.0)
    norm2 = (abs2 / q2).clamp(max=1.0)

    w = torch.min(norm1, norm2)

    active_mask = w > 0.01
    overlap_ratio = active_mask.float().mean().item()

    w_sum = w.sum()
    if w_sum < 1e-8:
        overlap_cos = float('nan')
    else:
        numerator = (grad1 * grad2 * w).sum()
        denom = ((grad1 ** 2 * w).sum().sqrt() * (grad2 ** 2 * w).sum().sqrt() + 1e-8)
        overlap_cos = (numerator / denom).item()

    if np.isnan(overlap_cos) or np.isnan(raw_cos):
        interp = 'N/A'
    elif overlap_ratio < 0.05:
        interp = 'disjoint'
    elif overlap_cos < -0.1:
        interp = 'hidden_conflict' if abs(raw_cos) < 0.1 else 'conflict'
    elif overlap_cos > 0.1:
        interp = 'cooperative'
    else:
        interp = 'true_orthogonal'

    return {
        'overlap_cosine': float(overlap_cos),
        'overlap_ratio': float(overlap_ratio),
        'raw_cosine': float(raw_cos),
        'interpretation': interp,
    }


def compute_interference(grad_source: torch.Tensor, grad_victim: torch.Tensor) -> float:
    """Interference of source on victim: ||g_src|| * max(0, -cos(g_src, g_vic))."""
    cos = compute_cosine_similarity(grad_source, grad_victim)
    if np.isnan(cos):
        return float('nan')
    return torch.norm(grad_source).item() * max(0.0, -cos)


def compute_decomposition(grad_a: torch.Tensor, grad_b: torch.Tensor) -> Tuple[float, float]:
    """Decompose grad_a into cooperative and conflicting components w.r.t. grad_b."""
    if grad_a is None or grad_b is None:
        return (float('nan'), float('nan'))
    norm_b = torch.norm(grad_b)
    if norm_b < 1e-8:
        return (float('nan'), float('nan'))
    proj_scalar = torch.dot(grad_a, grad_b) / (norm_b * norm_b)
    proj_mag = proj_scalar.item() * norm_b.item()
    if proj_mag >= 0:
        return (proj_mag, 0.0)
    else:
        return (0.0, abs(proj_mag))


def compute_subspace_principal_angles(
    grad1: torch.Tensor, grad2: torch.Tensor, weight_shape: Tuple[int, int],
    top_k: int = 5,
) -> Dict[str, object]:
    """Compute principal angles between gradient subspaces via SVD."""
    out_dim, in_dim = weight_shape
    if grad1.numel() != out_dim * in_dim or grad2.numel() != out_dim * in_dim:
        return None

    _svd_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    G1 = grad1.reshape(out_dim, in_dim).float().to(_svd_device)
    G2 = grad2.reshape(out_dim, in_dim).float().to(_svd_device)

    if not torch.isfinite(G1).all() or not torch.isfinite(G2).all():
        return None
    if G1.abs().max() < 1e-12 or G2.abs().max() < 1e-12:
        return None

    try:
        U1, S1, _ = torch.linalg.svd(G1, full_matrices=False)
        U2, S2, _ = torch.linalg.svd(G2, full_matrices=False)
    except (torch._C._LinAlgError, RuntimeError):
        return None

    k = min(top_k, U1.shape[1], U2.shape[1])
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]

    M = U1_k.T @ U2_k
    try:
        sigmas = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    except (torch._C._LinAlgError, RuntimeError):
        return None
    angles_rad = torch.acos(sigmas)
    angles_deg = torch.rad2deg(angles_rad)

    total_var1 = (S1 ** 2).sum()
    total_var2 = (S2 ** 2).sum()

    return {
        'principal_angles_deg': angles_deg.tolist(),
        'mean_angle_deg': float(angles_deg.mean().item()),
        'min_angle_deg': float(angles_deg.min().item()),
        'max_angle_deg': float(angles_deg.max().item()),
        'top_k': k,
        'explained_var_ratio_1': (S1[:k] ** 2 / (total_var1 + 1e-12)).tolist(),
        'explained_var_ratio_2': (S2[:k] ** 2 / (total_var2 + 1e-12)).tolist(),
    }


def get_group_weight_info(
    param_groups: OrderedDict,
) -> Dict[str, Dict]:
    """For each parameter group, find the largest 2D weight and its offset."""
    info = {}
    for gk, params in param_groups.items():
        offset = 0
        best = None
        for name, param in params.items():
            if param.dim() == 2:
                numel = param.numel()
                if best is None or numel > best[2]:
                    best = (tuple(param.shape), offset, numel)
            offset += param.numel()
        if best is not None:
            info[gk] = {'shape': best[0], 'offset': best[1], 'numel': best[2]}
    return info


def scatter_data_to_device(data: dict, gpu_id: int) -> dict:
    """Use mmcv scatter_kwargs to move DataContainer data to GPU."""
    _, scatter_kwargs_out = scatter_kwargs(
        inputs=None, kwargs=data, target_gpus=[gpu_id]
    )
    return scatter_kwargs_out[0]


# ============================================================================
# Model / Decoder Helpers
# ============================================================================

def get_decoder(model: nn.Module):
    if hasattr(model, 'module'):
        model = model.module
    head = model.head
    if hasattr(head, 'onedecoder_head'):
        return head.onedecoder_head
    if hasattr(head, 'sparse_head'):
        return head.sparse_head
    if hasattr(head, 'operation_order'):
        return head
    raise AttributeError("Could not find unified decoder in model structure")


def compute_decoder_layer_mapping(operation_order: List[str]) -> Dict[int, int]:
    """Map each operation index to its decoder layer index (0-based)."""
    mapping = {}
    dec_idx = 0
    for i, op in enumerate(operation_order):
        mapping[i] = dec_idx
        if op == 'refine':
            dec_idx += 1
    return mapping


# ============================================================================
# Grouped Parameter Extraction
# ============================================================================

def _split_submodule_params(
    group_key: str, params: OrderedDict, op_type: str, meta: dict,
) -> List[Tuple[str, OrderedDict, dict]]:
    """Split an operation's params into MGCM-style fine-grained sub-modules."""
    results = []

    if op_type in ('gnn', 'temp_gnn', 'inter_gnn'):
        attn_subgroups = defaultdict(lambda: defaultdict(OrderedDict))
        other_params = OrderedDict()
        for name, param in params.items():
            parts = name.split('.')
            if 'attns' in parts:
                attn_pos = parts.index('attns')
                attn_idx = parts[attn_pos + 1]
                remainder = '.'.join(parts[attn_pos + 2:])
                if 'in_proj' in remainder:
                    attn_subgroups[attn_idx]['qkv'][name] = param
                elif 'out_proj' in remainder:
                    attn_subgroups[attn_idx]['o'][name] = param
                else:
                    attn_subgroups[attn_idx]['other'][name] = param
            else:
                other_params[name] = param

        for attn_idx in sorted(attn_subgroups.keys()):
            for part_name, part_params in attn_subgroups[attn_idx].items():
                if part_params:
                    sub_key = f"{group_key}_attn{attn_idx}_{part_name}"
                    sub_meta = {**meta, 'sub_module': f'attn{attn_idx}_{part_name}'}
                    results.append((sub_key, part_params, sub_meta))
        if other_params:
            sub_key = f"{group_key}_other"
            results.append((sub_key, other_params, {**meta, 'sub_module': 'other'}))

    elif op_type == 'ffn':
        import re
        w1_params = OrderedDict()
        w2_params = OrderedDict()
        pre_norm_params = OrderedDict()
        other_params = OrderedDict()
        ffn_layers_re = re.compile(r'\.layers\.(\d+)')
        for name, param in params.items():
            if 'pre_norm' in name:
                pre_norm_params[name] = param
            else:
                matches = list(ffn_layers_re.finditer(name))
                if len(matches) >= 2:
                    ffn_idx = int(matches[-1].group(1))
                elif len(matches) == 1:
                    ffn_idx = int(matches[0].group(1))
                else:
                    ffn_idx = -1

                if ffn_idx == 0:
                    w1_params[name] = param
                elif ffn_idx >= 1:
                    w2_params[name] = param
                else:
                    other_params[name] = param

        if pre_norm_params:
            results.append((f"{group_key}_prenorm", pre_norm_params,
                          {**meta, 'sub_module': 'pre_norm'}))
        if w1_params:
            results.append((f"{group_key}_w1", w1_params,
                          {**meta, 'sub_module': 'w1'}))
        if w2_params:
            results.append((f"{group_key}_w2", w2_params,
                          {**meta, 'sub_module': 'w2'}))
        if other_params:
            results.append((f"{group_key}_other", other_params,
                          {**meta, 'sub_module': 'other'}))

    else:
        results.append((group_key, params, meta))

    return results


def get_shared_parameters_grouped(
    model: nn.Module,
    shared_layer_names: List[str],
    last_n_layers: Optional[int] = None,
    modular: bool = False,
) -> Tuple[OrderedDict, OrderedDict]:
    """Extract shared parameters grouped by (decoder_layer, operation_type)."""
    raw_model = model.module if hasattr(model, 'module') else model

    param_groups = OrderedDict()
    group_meta = OrderedDict()

    # ---- Non-decoder shared modules: backbone, neck ----
    non_decoder_modules = {
        'backbone': getattr(raw_model, 'img_backbone', None),
        'neck': getattr(raw_model, 'img_neck', None),
    }
    for mod_name, mod in non_decoder_modules.items():
        if mod_name not in shared_layer_names or mod is None:
            continue
        if mod_name == 'backbone':
            stage_names = ['layer1', 'layer2', 'layer3', 'layer4']
            stage_prefixes = set(stage_names)
            stem_params = OrderedDict()
            for name, param in mod.named_parameters():
                if param.requires_grad and not any(name.startswith(s) for s in stage_prefixes):
                    stem_params[f"backbone.{name}"] = param
            if stem_params:
                param_groups['backbone_stem'] = stem_params
                group_meta['backbone_stem'] = {
                    'decoder_idx': -1,
                    'op_type': 'backbone_stem',
                    'global_layer_idx': -1,
                }
            for stage_name in stage_names:
                stage = getattr(mod, stage_name, None)
                if stage is None:
                    continue
                stage_params = OrderedDict()
                for name, param in stage.named_parameters():
                    if param.requires_grad:
                        stage_params[f"backbone.{stage_name}.{name}"] = param
                if stage_params:
                    gk = f"backbone_{stage_name}"
                    param_groups[gk] = stage_params
                    group_meta[gk] = {
                        'decoder_idx': -1,
                        'op_type': f'backbone_{stage_name}',
                        'global_layer_idx': -1,
                    }
        else:
            params = OrderedDict()
            for name, param in mod.named_parameters():
                if param.requires_grad:
                    params[f"{mod_name}.{name}"] = param
            if params:
                param_groups[mod_name] = params
                group_meta[mod_name] = {
                    'decoder_idx': -1,
                    'op_type': mod_name,
                    'global_layer_idx': -1,
                }

    # ---- Decoder operation layers ----
    decoder = get_decoder(model)
    operation_order = decoder.operation_order
    dec_mapping = compute_decoder_layer_mapping(operation_order)

    if last_n_layers is not None:
        refine_indices = [i for i, op in enumerate(operation_order) if op == 'refine']
        start_idx = refine_indices[-last_n_layers] if last_n_layers <= len(refine_indices) else 0
    else:
        start_idx = 0

    op_count_per_dec: Dict[Tuple[int, str], int] = defaultdict(int)

    for i, (op, layer) in enumerate(zip(operation_order, decoder.layers)):
        if layer is None or i < start_idx:
            continue
        if op not in shared_layer_names:
            continue

        dec_idx = dec_mapping[i]
        count = op_count_per_dec[(dec_idx, op)]
        op_count_per_dec[(dec_idx, op)] += 1
        group_key = f"dec{dec_idx}_{op}_{count}"

        params = OrderedDict()
        for name, param in layer.named_parameters():
            if param.requires_grad:
                params[f"layers.{i}.{name}"] = param

        if params:
            param_groups[group_key] = params
            group_meta[group_key] = {
                'decoder_idx': dec_idx,
                'op_type': op,
                'global_layer_idx': i,
            }

    # fc_before / fc_after
    for fc_name in ['fc_before', 'fc_after']:
        if fc_name not in shared_layer_names:
            continue
        fc_mod = getattr(decoder, fc_name, None)
        if fc_mod is None or isinstance(fc_mod, nn.Identity):
            continue
        params = OrderedDict()
        for name, param in fc_mod.named_parameters():
            if param.requires_grad:
                params[f"{fc_name}.{name}"] = param
        if params:
            param_groups[fc_name] = params
            group_meta[fc_name] = {
                'decoder_idx': -1,
                'op_type': fc_name,
                'global_layer_idx': -1,
            }

    if modular:
        modular_groups = OrderedDict()
        modular_meta = OrderedDict()
        for gk, params in param_groups.items():
            meta = group_meta[gk]
            op_type = meta['op_type']
            for sub_key, sub_params, sub_meta in _split_submodule_params(
                gk, params, op_type, meta
            ):
                modular_groups[sub_key] = sub_params
                modular_meta[sub_key] = sub_meta
        return modular_groups, modular_meta

    return param_groups, group_meta


def get_shared_parameters(model, shared_layer_names, last_n_layers=None):
    """Backward-compatible flat parameter dict."""
    param_groups, _ = get_shared_parameters_grouped(model, shared_layer_names, last_n_layers)
    flat = OrderedDict()
    for gk, params in param_groups.items():
        flat.update(params)
    return flat


# ============================================================================
# Gradient Computation
# ============================================================================

def _sum_task_loss(loss_dict, task_name):
    """Sum all losses for a given task."""
    prefixes = TASK_GROUPS.get(task_name, [])
    task_loss = None
    for key, val in loss_dict.items():
        if match_loss_key(key, prefixes) and isinstance(val, torch.Tensor) and val.requires_grad:
            task_loss = val if task_loss is None else task_loss + val
    return task_loss


def compute_task_gradient_grouped(
    model: nn.Module,
    loss_dict: Dict[str, torch.Tensor],
    task_name: str,
    param_groups: OrderedDict,
    retain_graph: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    """Compute per-group gradient vectors for a task via .backward()."""
    task_loss = _sum_task_loss(loss_dict, task_name)
    if task_loss is None:
        return None

    all_params = []
    for gk, params in param_groups.items():
        all_params.extend(params.values())

    if not all_params:
        return None

    for p in all_params:
        if p.grad is not None:
            p.grad = None

    try:
        task_loss.backward(retain_graph=retain_graph)
    except RuntimeError as e:
        print(f"Warning: gradient computation failed for {task_name}: {e}")
        return None

    device = all_params[0].device
    dtype = all_params[0].dtype

    result = {}
    for gk, params in param_groups.items():
        group_grads = []
        for p in params.values():
            if p.grad is not None:
                group_grads.append(p.grad.detach().flatten())
            else:
                group_grads.append(torch.zeros(p.numel(), device=device, dtype=dtype))
        result[gk] = torch.cat(group_grads)

    valid_chunks = [v for v in result.values() if not torch.isnan(v).any()]
    result['_all'] = torch.cat(valid_chunks) if valid_chunks else torch.cat(list(result.values()))

    return result


def compute_task_gradient(model, loss_dict, task_name, shared_params, retain_graph=True):
    """Backward-compatible: returns single flat gradient tensor."""
    wrapper = OrderedDict([('_flat', shared_params)])
    result = compute_task_gradient_grouped(model, loss_dict, task_name, wrapper, retain_graph)
    if result is None:
        return None
    return result['_all']


# ============================================================================
# Per-Group Metrics
# ============================================================================

def compute_per_group_metrics(task_grads_grouped, group_keys):
    """Compute cosine, norms, interference, decomposition, active-overlap for all groups."""
    tasks = list(task_grads_grouped.keys())
    per_group_cosine = {}
    per_group_norms = {}
    per_group_conflict = {}
    per_group_interference = {}
    per_group_decomposition = {}
    per_group_active_overlap = {}

    for gk in group_keys:
        cosines = {}
        norms = {}
        conflicts = {}
        interferences = {}
        decompositions = {}
        active_overlaps = {}

        for task in tasks:
            g = task_grads_grouped[task].get(gk)
            norms[task] = torch.norm(g).item() if g is not None else 0.0

        for t1, t2 in TASK_PAIRS:
            pk = f"{t1}_vs_{t2}"
            g1 = task_grads_grouped.get(t1, {}).get(gk)
            g2 = task_grads_grouped.get(t2, {}).get(gk)
            if g1 is not None and g2 is not None:
                c = compute_cosine_similarity(g1, g2)
                cosines[pk] = c
                conflicts[pk] = c < 0 if not np.isnan(c) else False
                interferences[f"{t1}_on_{t2}"] = compute_interference(g1, g2)
                interferences[f"{t2}_on_{t1}"] = compute_interference(g2, g1)
                coop, conf = compute_decomposition(g1, g2)
                decompositions[f"{t1}_wrt_{t2}"] = (coop, conf)
                coop2, conf2 = compute_decomposition(g2, g1)
                decompositions[f"{t2}_wrt_{t1}"] = (coop2, conf2)
                active_overlaps[pk] = compute_active_overlap_cosine(g1, g2)

        per_group_cosine[gk] = cosines
        per_group_norms[gk] = norms
        per_group_conflict[gk] = conflicts
        per_group_interference[gk] = interferences
        per_group_decomposition[gk] = decompositions
        per_group_active_overlap[gk] = active_overlaps

    return (per_group_cosine, per_group_norms, per_group_conflict,
            per_group_interference, per_group_decomposition, per_group_active_overlap)


# ============================================================================
# Single Batch Analysis
# ============================================================================

def analyze_single_batch(
    model, data, param_groups, group_meta, gpu_id,
    fp16=False, analysis_mode='full', store_gradients=False,
    selective_eval=True,
):
    results = {
        'pairwise_cosine': {},
        'gradient_norms': {},
        'conflict_flags': {},
    }

    # nuScenes: use scatter_kwargs for DataContainer handling
    data_gpu = scatter_data_to_device(data, gpu_id)
    raw_model = model.module if hasattr(model, 'module') else model

    model.train()

    _ctx = _selective_eval(model) if selective_eval else contextlib.nullcontext()
    with _ctx:
        # nuScenes: use forward_train with float() to avoid fp16 mismatch
        img = data_gpu.pop('img')
        outputs = raw_model.forward_train(img.float(), **data_gpu)

        if not isinstance(outputs, dict):
            print("Warning: Model output is not a dictionary of losses")
            return results

        task_names = list(TASK_GROUPS.keys())
        task_grads_grouped = {}
        group_keys = [gk for gk in param_groups.keys()]

        for i, task in enumerate(task_names):
            retain = (i < len(task_names) - 1)
            model.zero_grad(set_to_none=True)
            grad_dict = compute_task_gradient_grouped(model, outputs, task, param_groups, retain_graph=retain)
            if grad_dict is not None:
                task_grads_grouped[task] = {gk: g.cpu() for gk, g in grad_dict.items()}
                results['gradient_norms'][task] = torch.norm(grad_dict['_all']).item()

    # Overall pairwise cosine + active-overlap
    active_overlap_results = {}
    for t1, t2 in TASK_PAIRS:
        pk = f"{t1}_vs_{t2}"
        g1 = task_grads_grouped.get(t1, {}).get('_all')
        g2 = task_grads_grouped.get(t2, {}).get('_all')
        if g1 is not None and g2 is not None:
            c = compute_cosine_similarity(g1, g2)
            results['pairwise_cosine'][pk] = c
            results['conflict_flags'][pk] = c < 0 if not np.isnan(c) else False
            active_overlap_results[pk] = compute_active_overlap_cosine(g1, g2)
    results['active_overlap'] = active_overlap_results

    # Overall interference & decomposition
    if analysis_mode in ('full', 'direction'):
        interference = {}
        decomposition = {}
        for t1, t2 in TASK_PAIRS:
            g1 = task_grads_grouped.get(t1, {}).get('_all')
            g2 = task_grads_grouped.get(t2, {}).get('_all')
            if g1 is not None and g2 is not None:
                interference[f"{t1}_on_{t2}"] = compute_interference(g1, g2)
                interference[f"{t2}_on_{t1}"] = compute_interference(g2, g1)
                c1, f1 = compute_decomposition(g1, g2)
                decomposition[f"{t1}_wrt_{t2}"] = (c1, f1)
                c2, f2 = compute_decomposition(g2, g1)
                decomposition[f"{t2}_wrt_{t1}"] = (c2, f2)
        results['interference'] = interference
        results['decomposition'] = decomposition

    # Per-group metrics (includes active-overlap)
    if analysis_mode in ('full', 'direction'):
        pgc, pgn, pgf, pgi, pgd, pgao = compute_per_group_metrics(task_grads_grouped, group_keys)
        results['per_group_cosine'] = pgc
        results['per_group_norms'] = pgn
        results['per_group_conflict'] = pgf
        results['per_group_interference'] = pgi
        results['per_group_decomposition'] = pgd
        results['per_group_active_overlap'] = pgao

    # Store raw gradients for PCA
    if store_gradients:
        results['task_gradients_all'] = {t: task_grads_grouped[t]['_all'] for t in task_grads_grouped}
        results['task_gradients_grouped'] = {
            t: {gk: g for gk, g in task_grads_grouped[t].items() if gk != '_all'}
            for t in task_grads_grouped
        }

    model.zero_grad(set_to_none=True)
    return results


# ============================================================================
# Activation Gradient Analysis
# ============================================================================

class ActivationHookManager:
    """Register forward hooks on specific modules to capture activation tensors."""

    def __init__(self):
        self.activations = {}
        self._handles = []

    def register(self, name: str, module: nn.Module):
        def hook_fn(mod, inp, out, name=name):
            if isinstance(out, torch.Tensor):
                out.retain_grad()
                self.activations[name] = out
            elif isinstance(out, (list, tuple)) and len(out) > 0:
                for i, o in enumerate(out):
                    if isinstance(o, torch.Tensor):
                        o.retain_grad()
                        self.activations[f"{name}_scale{i}"] = o
        handle = module.register_forward_hook(hook_fn)
        self._handles.append(handle)

    def register_input(self, name: str, module: nn.Module):
        def hook_fn(mod, inp, out, name=name):
            if isinstance(inp, tuple) and len(inp) > 0:
                tensor = inp[0]
                if isinstance(tensor, torch.Tensor):
                    tensor.retain_grad()
                    self.activations[name] = tensor
        handle = module.register_forward_hook(hook_fn)
        self._handles.append(handle)

    def register_sliced(self, base_name: str, module: nn.Module,
                        query_select: List[str], anchor_section: Dict[str, List[int]],
                        capture_input: bool = False):
        def hook_fn(mod, inp, out, base_name=base_name, qs=query_select,
                    sections=anchor_section, cap_in=capture_input):
            if isinstance(out, torch.Tensor):
                out.retain_grad()
                self.activations[f"{base_name}_out"] = out
                for q in qs:
                    if q in sections:
                        start, end = sections[q]
                        sliced = out[:, start:end]
                        sliced.retain_grad()
                        self.activations[f"{base_name}_out_{q}"] = sliced

            if cap_in and isinstance(inp, tuple) and len(inp) > 0:
                in_tensor = inp[0]
                if isinstance(in_tensor, torch.Tensor):
                    in_tensor.retain_grad()
                    self.activations[f"{base_name}_in"] = in_tensor
                    for q in qs:
                        if q in sections:
                            start, end = sections[q]
                            sliced = in_tensor[:, start:end]
                            sliced.retain_grad()
                            self.activations[f"{base_name}_in_{q}"] = sliced

        handle = module.register_forward_hook(hook_fn)
        self._handles.append(handle)

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def _get_decoder(model: nn.Module):
    """Get the unified decoder from a (possibly wrapped) model."""
    if hasattr(model, 'module'):
        model = model.module
    head = model.head
    if hasattr(head, 'onedecoder_head'):
        return head.onedecoder_head
    if hasattr(head, 'sparse_head'):
        return head.sparse_head
    if hasattr(head, 'operation_order'):
        return head
    return None


def _setup_activation_hooks(model: nn.Module, hook_manager: ActivationHookManager):
    """Set up activation hooks for all analysis points."""
    raw_model = model.module if hasattr(model, 'module') else model

    # 1. FPN neck
    if hasattr(raw_model, 'img_neck'):
        hook_manager.register('fpn', raw_model.img_neck)

    # 2-4. Decoder internals
    decoder = _get_decoder(model)
    if decoder is None:
        return

    for task_prefix in ['det', 'map', 'ego', 'plan']:
        deform_attr = f'{task_prefix}_deformable'
        if hasattr(decoder, deform_attr):
            deform_list = getattr(decoder, deform_attr)
            for i, deform_mod in enumerate(deform_list):
                hook_manager.register(f'{task_prefix}_deformable_{i}', deform_mod)

    if hasattr(decoder, 'operation_order') and hasattr(decoder, 'layers'):
        dec_mapping = compute_decoder_layer_mapping(decoder.operation_order)
        query_select = getattr(decoder, 'query_select', [])

        for i, op in enumerate(decoder.operation_order):
            if decoder.layers[i] is None:
                continue
            dec_idx = dec_mapping.get(i, -1)

            if op == 'inter_gnn':
                _register_dynamic_sliced_hook(
                    hook_manager, f'inter_gnn_dec{dec_idx}',
                    decoder.layers[i], decoder, query_select,
                    capture_input=True,
                )
            elif op in ('gnn', 'temp_gnn'):
                _register_dynamic_sliced_hook(
                    hook_manager, f'{op}_dec{dec_idx}',
                    decoder.layers[i], decoder, query_select,
                    capture_input=False,
                )


def _register_dynamic_sliced_hook(
    hook_manager: ActivationHookManager,
    base_name: str,
    layer_module: nn.Module,
    decoder: nn.Module,
    query_select: List[str],
    capture_input: bool = False,
):
    def hook_fn(mod, inp, out):
        section = getattr(decoder, 'num_anchor_section', None)
        if section is None:
            if isinstance(out, torch.Tensor):
                out.retain_grad()
                hook_manager.activations[f"{base_name}_out"] = out
            return

        if isinstance(out, torch.Tensor):
            out.retain_grad()
            hook_manager.activations[f"{base_name}_out"] = out
            for q in query_select:
                if q in section:
                    start, end = section[q]
                    sliced = out[:, start:end]
                    sliced.retain_grad()
                    hook_manager.activations[f"{base_name}_out_{q}"] = sliced

        if capture_input and isinstance(inp, tuple) and len(inp) > 0:
            in_tensor = inp[0]
            if isinstance(in_tensor, torch.Tensor):
                in_tensor.retain_grad()
                hook_manager.activations[f"{base_name}_in"] = in_tensor
                for q in query_select:
                    if q in section:
                        start, end = section[q]
                        sliced = in_tensor[:, start:end]
                        sliced.retain_grad()
                        hook_manager.activations[f"{base_name}_in_{q}"] = sliced

    handle = layer_module.register_forward_hook(hook_fn)
    hook_manager._handles.append(handle)


def compute_activation_gradients(
    model: nn.Module,
    loss_dict: Dict[str, torch.Tensor],
    activation_tensors: Dict[str, torch.Tensor],
    task_name: str,
    retain_graph: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    """Compute gradients of a task's loss w.r.t. captured activation tensors."""
    task_loss = _sum_task_loss(loss_dict, task_name)
    if task_loss is None:
        return None

    valid_tensors = {}
    for name, tensor in activation_tensors.items():
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
            valid_tensors[name] = tensor

    if not valid_tensors:
        return None

    for tensor in valid_tensors.values():
        if tensor.grad is not None:
            tensor.grad = None

    try:
        task_loss.backward(retain_graph=retain_graph)
    except RuntimeError as e:
        print(f"Warning: activation gradient failed for {task_name}: {e}")
        return None

    result = {}
    for name, tensor in valid_tensors.items():
        if tensor.grad is not None:
            result[name] = tensor.grad.detach().flatten().cpu()
        else:
            result[name] = torch.zeros(tensor.numel(), dtype=torch.float32)

    return result


def analyze_activation_gradients_single_batch(
    model, data, gpu_id, fp16=False, selective_eval=True,
):
    """Run a single batch with activation hooks and compute per-task activation gradients."""
    results = {
        'activation_cosine': {},
        'activation_active_overlap': {},
        'activation_norms': {},
    }

    data_gpu = scatter_data_to_device(data, gpu_id)
    raw_model = model.module if hasattr(model, 'module') else model
    model.train()

    hook_manager = ActivationHookManager()
    _setup_activation_hooks(model, hook_manager)

    _ctx = _selective_eval(model) if selective_eval else contextlib.nullcontext()
    try:
        with _ctx:
            img = data_gpu.pop('img')
            outputs = raw_model.forward_train(img.float(), **data_gpu)

            if not isinstance(outputs, dict):
                return results

            if not hook_manager.activations:
                print("Warning: No activations captured by hooks")
                return results

            task_names = list(TASK_GROUPS.keys())
            task_act_grads = {}

            for i, task in enumerate(task_names):
                retain = (i < len(task_names) - 1)
                model.zero_grad(set_to_none=True)
                act_grads = compute_activation_gradients(
                    model, outputs, hook_manager.activations, task, retain_graph=retain,
                )
                if act_grads is not None:
                    task_act_grads[task] = act_grads

        all_hook_names = set()
        for task_grads in task_act_grads.values():
            all_hook_names.update(task_grads.keys())

        for hp in sorted(all_hook_names):
            cosines = {}
            active_overlaps = {}
            norms = {}

            for task in task_act_grads:
                g = task_act_grads[task].get(hp)
                norms[task] = torch.norm(g).item() if g is not None else 0.0

            for t1, t2 in TASK_PAIRS:
                pk = f"{t1}_vs_{t2}"
                g1 = task_act_grads.get(t1, {}).get(hp)
                g2 = task_act_grads.get(t2, {}).get(hp)
                if g1 is not None and g2 is not None:
                    cosines[pk] = compute_cosine_similarity(g1, g2)
                    active_overlaps[pk] = compute_active_overlap_cosine(g1, g2)

            if cosines:
                results['activation_cosine'][hp] = cosines
                results['activation_active_overlap'][hp] = active_overlaps
                results['activation_norms'][hp] = norms

    finally:
        hook_manager.remove_hooks()
        hook_manager.clear()
        model.zero_grad(set_to_none=True)

    return results


# ============================================================================
# Aggregation
# ============================================================================

def _agg_scalar_dict(all_results, key):
    agg = defaultdict(list)
    for r in all_results:
        d = r.get(key, {})
        for sk, val in d.items():
            if isinstance(val, (int, float)) and not np.isnan(val):
                agg[sk].append(val)
    return dict(agg)


def _stats_from_values(values):
    if not values:
        return None
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'values': values,
    }


def aggregate_results(all_results, analysis_mode='basic'):
    stats = {
        'pairwise_cosine': {},
        'gradient_norms': {},
        'conflict_frequency': {},
        'num_batches': len(all_results),
    }

    cos_agg = defaultdict(list)
    norm_agg = defaultdict(list)
    flag_agg = defaultdict(list)
    for r in all_results:
        for pk, c in r.get('pairwise_cosine', {}).items():
            if not np.isnan(c):
                cos_agg[pk].append(c)
        for t, n in r.get('gradient_norms', {}).items():
            norm_agg[t].append(n)
        for pk, f in r.get('conflict_flags', {}).items():
            flag_agg[pk].append(f)

    for pk, vals in cos_agg.items():
        stats['pairwise_cosine'][pk] = _stats_from_values(vals)
    for t, vals in norm_agg.items():
        stats['gradient_norms'][t] = _stats_from_values(vals)
    for pk, vals in flag_agg.items():
        ratio = sum(vals) / len(vals)
        stats['conflict_frequency'][pk] = {'ratio': ratio, 'count': sum(vals), 'total': len(vals)}

    # Overall active-overlap
    ao_overlap_cos_agg = defaultdict(list)
    ao_overlap_ratio_agg = defaultdict(list)
    ao_interp_agg = defaultdict(lambda: defaultdict(int))
    for r in all_results:
        for pk, ao in r.get('active_overlap', {}).items():
            oc = ao.get('overlap_cosine', float('nan'))
            if not np.isnan(oc):
                ao_overlap_cos_agg[pk].append(oc)
            oratio = ao.get('overlap_ratio', 0.0)
            ao_overlap_ratio_agg[pk].append(oratio)
            interp = ao.get('interpretation', 'N/A')
            ao_interp_agg[pk][interp] += 1
    stats['active_overlap'] = {}
    for pk in ao_overlap_cos_agg:
        stats['active_overlap'][pk] = {
            'overlap_cosine': _stats_from_values(ao_overlap_cos_agg[pk]),
            'overlap_ratio': _stats_from_values(ao_overlap_ratio_agg[pk]),
            'interpretation_counts': dict(ao_interp_agg[pk]),
        }

    if analysis_mode in ('full', 'direction'):
        # Interference
        intf_agg = _agg_scalar_dict(all_results, 'interference')
        stats['interference'] = {k: _stats_from_values(v) for k, v in intf_agg.items() if v}

        # Decomposition
        decomp_coop = defaultdict(list)
        decomp_conf = defaultdict(list)
        for r in all_results:
            for dk, (c, f) in r.get('decomposition', {}).items():
                if not np.isnan(c):
                    decomp_coop[dk].append(c)
                    decomp_conf[dk].append(f)
        stats['decomposition'] = {}
        for dk in decomp_coop:
            stats['decomposition'][dk] = {
                'cooperative': _stats_from_values(decomp_coop[dk]),
                'conflicting': _stats_from_values(decomp_conf[dk]),
            }

        # Per-group cosine, norms, conflict, interference, decomposition
        pg_cos_agg = defaultdict(lambda: defaultdict(list))
        pg_norm_agg = defaultdict(lambda: defaultdict(list))
        pg_flag_agg = defaultdict(lambda: defaultdict(list))
        pg_intf_agg = defaultdict(lambda: defaultdict(list))
        pg_coop_agg = defaultdict(lambda: defaultdict(list))
        pg_conf_agg = defaultdict(lambda: defaultdict(list))

        for r in all_results:
            for gk, cosines in r.get('per_group_cosine', {}).items():
                for pk, c in cosines.items():
                    if not np.isnan(c):
                        pg_cos_agg[gk][pk].append(c)
            for gk, norms in r.get('per_group_norms', {}).items():
                for t, n in norms.items():
                    pg_norm_agg[gk][t].append(n)
            for gk, flags in r.get('per_group_conflict', {}).items():
                for pk, f in flags.items():
                    pg_flag_agg[gk][pk].append(f)
            for gk, intfs in r.get('per_group_interference', {}).items():
                for ik, v in intfs.items():
                    if not np.isnan(v):
                        pg_intf_agg[gk][ik].append(v)
            for gk, decomps in r.get('per_group_decomposition', {}).items():
                for dk, (c, f) in decomps.items():
                    if not np.isnan(c):
                        pg_coop_agg[gk][dk].append(c)
                        pg_conf_agg[gk][dk].append(f)

        stats['per_group_cosine'] = {
            gk: {pk: _stats_from_values(vals) for pk, vals in pks.items()}
            for gk, pks in pg_cos_agg.items()
        }
        stats['per_group_norms'] = {
            gk: {t: _stats_from_values(vals) for t, vals in ts.items()}
            for gk, ts in pg_norm_agg.items()
        }
        stats['per_group_conflict_frequency'] = {}
        for gk, pks in pg_flag_agg.items():
            stats['per_group_conflict_frequency'][gk] = {}
            for pk, vals in pks.items():
                ratio = sum(vals) / len(vals)
                stats['per_group_conflict_frequency'][gk][pk] = {'ratio': ratio, 'count': sum(vals), 'total': len(vals)}
        stats['per_group_interference'] = {
            gk: {ik: _stats_from_values(vals) for ik, vals in iks.items()}
            for gk, iks in pg_intf_agg.items()
        }
        stats['per_group_decomposition'] = {}
        for gk in pg_coop_agg:
            stats['per_group_decomposition'][gk] = {}
            for dk in pg_coop_agg[gk]:
                stats['per_group_decomposition'][gk][dk] = {
                    'cooperative': _stats_from_values(pg_coop_agg[gk][dk]),
                    'conflicting': _stats_from_values(pg_conf_agg[gk][dk]),
                }

        # Per-group active-overlap
        pg_ao_cos_agg = defaultdict(lambda: defaultdict(list))
        pg_ao_ratio_agg = defaultdict(lambda: defaultdict(list))
        pg_ao_interp_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for r in all_results:
            for gk, aos in r.get('per_group_active_overlap', {}).items():
                for pk, ao in aos.items():
                    oc = ao.get('overlap_cosine', float('nan'))
                    if not np.isnan(oc):
                        pg_ao_cos_agg[gk][pk].append(oc)
                    pg_ao_ratio_agg[gk][pk].append(ao.get('overlap_ratio', 0.0))
                    pg_ao_interp_agg[gk][pk][ao.get('interpretation', 'N/A')] += 1
        stats['per_group_active_overlap'] = {}
        for gk in pg_ao_cos_agg:
            stats['per_group_active_overlap'][gk] = {}
            for pk in pg_ao_cos_agg[gk]:
                stats['per_group_active_overlap'][gk][pk] = {
                    'overlap_cosine': _stats_from_values(pg_ao_cos_agg[gk][pk]),
                    'overlap_ratio': _stats_from_values(pg_ao_ratio_agg[gk][pk]),
                    'interpretation_counts': dict(pg_ao_interp_agg[gk][pk]),
                }

    # Gradient directions (for PCA)
    if analysis_mode == 'direction':
        grad_dirs = defaultdict(list)
        for r in all_results:
            for t, g in r.get('task_gradients_all', {}).items():
                grad_dirs[t].append(g)
        stats['gradient_directions'] = dict(grad_dirs)

        grad_dirs_grouped = defaultdict(lambda: defaultdict(list))
        for r in all_results:
            for t, groups in r.get('task_gradients_grouped', {}).items():
                for gk, g in groups.items():
                    grad_dirs_grouped[t][gk].append(g)
        stats['gradient_directions_grouped'] = {
            t: dict(groups) for t, groups in grad_dirs_grouped.items()
        }

    return stats


# ============================================================================
# Save & Print
# ============================================================================

def save_results(stats, output_dir, analysis_mode='basic'):
    os.makedirs(output_dir, exist_ok=True)

    def strip_values(d):
        if isinstance(d, dict):
            return {k: strip_values(v) for k, v in d.items() if k != 'values'}
        return d

    with open(os.path.join(output_dir, 'gradient_conflict_stats.json'), 'w') as f:
        json.dump(strip_values(stats), f, indent=2, default=str)

    with open(os.path.join(output_dir, 'gradient_conflict_details.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Task Pair', 'Mean Cosine', 'Std Cosine', 'Min', 'Max', 'Conflict Ratio',
                     'Overlap Cosine', 'Overlap Ratio', 'Top Interpretation'])
        for pk, cs in stats.get('pairwise_cosine', {}).items():
            cf = stats.get('conflict_frequency', {}).get(pk, {})
            ao = stats.get('active_overlap', {}).get(pk, {})
            ao_cos = ao.get('overlap_cosine', {})
            ao_ratio = ao.get('overlap_ratio', {})
            ao_interp = ao.get('interpretation_counts', {})
            top_interp = max(ao_interp, key=ao_interp.get) if ao_interp else 'N/A'
            w.writerow([pk, f"{cs['mean']:.4f}", f"{cs['std']:.4f}",
                        f"{cs['min']:.4f}", f"{cs['max']:.4f}", f"{cf.get('ratio', 0):.4f}",
                        f"{ao_cos.get('mean', float('nan')):.4f}",
                        f"{ao_ratio.get('mean', 0):.4f}", top_interp])

    with open(os.path.join(output_dir, 'gradient_norms.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Task', 'Mean Norm', 'Std Norm', 'Min', 'Max'])
        for t, ns in stats.get('gradient_norms', {}).items():
            w.writerow([t, f"{ns['mean']:.4f}", f"{ns['std']:.4f}", f"{ns['min']:.4f}", f"{ns['max']:.4f}"])

    if analysis_mode in ('full', 'direction') and 'per_group_cosine' in stats:
        layer_dir = os.path.join(output_dir, 'per_layer')
        os.makedirs(layer_dir, exist_ok=True)
        with open(os.path.join(layer_dir, 'per_layer_details.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Group', 'Task Pair', 'Mean Cosine', 'Std Cosine', 'Conflict Ratio',
                         'Overlap Cosine', 'Overlap Ratio', 'Top Interpretation'])
            for gk, pks in sorted(stats['per_group_cosine'].items()):
                for pk, cs in sorted(pks.items()):
                    cf = stats.get('per_group_conflict_frequency', {}).get(gk, {}).get(pk, {})
                    ao = stats.get('per_group_active_overlap', {}).get(gk, {}).get(pk, {})
                    ao_cos = ao.get('overlap_cosine', {})
                    ao_ratio = ao.get('overlap_ratio', {})
                    ao_interp = ao.get('interpretation_counts', {})
                    top_interp = max(ao_interp, key=ao_interp.get) if ao_interp else 'N/A'
                    w.writerow([gk, pk, f"{cs['mean']:.4f}", f"{cs['std']:.4f}",
                                f"{cf.get('ratio', 0):.4f}",
                                f"{ao_cos.get('mean', float('nan')):.4f}",
                                f"{ao_ratio.get('mean', 0):.4f}", top_interp])

    print(f"Results saved to {output_dir}")


def print_summary(stats, analysis_mode='basic', focus_task='plan'):
    print("\n" + "=" * 70)
    print("GRADIENT CONFLICT ANALYSIS SUMMARY (nuScenes)")
    print("=" * 70)
    print(f"\nAnalyzed {stats['num_batches']} batches")

    print("\n--- Pairwise Cosine Similarity (Mean +/- Std) ---")
    print(f"  {'Pair':20s}  {'Raw Cosine':>16s}  {'Overlap Cosine':>16s}  {'Overlap%':>8s}  {'Interp':>18s}")
    print(f"  {'-'*20}  {'-'*16}  {'-'*16}  {'-'*8}  {'-'*18}")
    for pk, cs in sorted(stats.get('pairwise_cosine', {}).items()):
        cf = stats.get('conflict_frequency', {}).get(pk, {})
        ratio = cf.get('ratio', 0)
        marker = " << CONFLICT" if ratio > 0.3 else ""
        ao = stats.get('active_overlap', {}).get(pk, {})
        ao_cos = ao.get('overlap_cosine', {})
        ao_ratio = ao.get('overlap_ratio', {})
        ao_interp = ao.get('interpretation_counts', {})
        top_interp = max(ao_interp, key=ao_interp.get) if ao_interp else 'N/A'
        ao_cos_str = f"{ao_cos['mean']:+.4f}" if ao_cos and 'mean' in ao_cos else "N/A"
        ao_ratio_str = f"{ao_ratio['mean']*100:.1f}%" if ao_ratio and 'mean' in ao_ratio else "N/A"
        print(f"  {pk:20s}  {cs['mean']:+.4f} +/- {cs['std']:.4f}"
              f"  {ao_cos_str:>16s}  {ao_ratio_str:>8s}  {top_interp:>18s}{marker}")

    print("\n--- Gradient Norms (Mean +/- Std) ---")
    for t, ns in sorted(stats.get('gradient_norms', {}).items()):
        print(f"  {t:10s}: {ns['mean']:.4f} +/- {ns['std']:.4f}")

    norms = stats.get('gradient_norms', {})
    if focus_task in norms:
        focus_norm = norms[focus_task]['mean']
        for t, ns in norms.items():
            if t != focus_task and focus_norm > 0:
                ratio = ns['mean'] / focus_norm
                if ratio > 5:
                    print(f"  WARNING: {t} norm is {ratio:.1f}x larger than {focus_task}")

    if analysis_mode in ('full', 'direction'):
        intf = stats.get('interference', {})
        if intf:
            print(f"\n--- Interference on '{focus_task}' (higher = worse) ---")
            for t in ['det', 'map', 'motion', 'ego']:
                key = f"{t}_on_{focus_task}"
                if key in intf:
                    print(f"  {t:10s} -> {focus_task}: {intf[key]['mean']:.4f} +/- {intf[key]['std']:.4f}")

        pg_cos = stats.get('per_group_cosine', {})
        if pg_cos:
            print(f"\n--- Top 10 Conflicting (group, pair) ---")
            all_entries = []
            for gk, pks in pg_cos.items():
                for pk, cs in pks.items():
                    all_entries.append((gk, pk, cs['mean']))
            all_entries.sort(key=lambda x: x[2])
            for gk, pk, mean_cos in all_entries[:10]:
                cf = stats.get('per_group_conflict_frequency', {}).get(gk, {}).get(pk, {})
                ratio = cf.get('ratio', 0)
                print(f"  {gk:25s} | {pk:20s} | cos={mean_cos:+.4f} | conflict={ratio * 100:.1f}%")

    print("\n--- Conflict Assessment ---")
    high = [(k, v['ratio']) for k, v in stats.get('conflict_frequency', {}).items() if v['ratio'] > 0.3]
    if high:
        print("  High conflict pairs (>30% negative cosine):")
        for pair, ratio in sorted(high, key=lambda x: -x[1]):
            print(f"    - {pair}: {ratio * 100:.1f}%")
    else:
        print("  No significant gradient conflicts detected.")

    ao_stats = stats.get('active_overlap', {})
    if ao_stats:
        hidden = [(pk, ao) for pk, ao in ao_stats.items()
                  if ao.get('interpretation_counts', {}).get('hidden_conflict', 0) > 0]
        if hidden:
            print("\n  DIAGNOSTIC: Hidden conflicts detected (raw cosine ~0 but overlap cosine < 0):")
            for pk, ao in hidden:
                hc = ao['interpretation_counts']['hidden_conflict']
                total = sum(ao['interpretation_counts'].values())
                print(f"    - {pk}: hidden_conflict in {hc}/{total} batches "
                      f"(overlap_cos={ao['overlap_cosine']['mean']:+.4f})")
        disjoint = [(pk, ao) for pk, ao in ao_stats.items()
                    if ao.get('interpretation_counts', {}).get('disjoint', 0) > 0]
        if disjoint:
            print("\n  DIAGNOSTIC: Disjoint gradient support (tasks update different coordinates):")
            for pk, ao in disjoint:
                dc = ao['interpretation_counts']['disjoint']
                total = sum(ao['interpretation_counts'].values())
                print(f"    - {pk}: disjoint in {dc}/{total} batches "
                      f"(overlap_ratio={ao['overlap_ratio']['mean']*100:.1f}%)")

    print("=" * 70 + "\n")


# ============================================================================
# Visualizations — imported from bench2drive version (identical logic)
# ============================================================================

# Import visualization functions from the bench2drive version to avoid duplication.
# All visualization functions are dataset-agnostic (they only operate on stats dicts).
try:
    from projects.gradient_conflict_anal import (
        create_basic_visualizations,
        create_per_layer_visualizations,
        create_advanced_visualizations,
        create_active_overlap_visualizations,
        create_subspace_visualizations,
        create_activation_visualizations,
        create_direction_visualizations,
        create_epoch_dynamics_visualizations,
        create_baseline_visualizations,
    )
    _VIZ_AVAILABLE = True
except ImportError:
    try:
        # Fallback: import from notebooks/analyze_gradient_conflict.py
        import importlib.util
        _viz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyze_gradient_conflict.py')
        _spec = importlib.util.spec_from_file_location('analyze_gradient_conflict', _viz_path)
        _viz_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_viz_mod)
        create_basic_visualizations = _viz_mod.create_basic_visualizations
        create_per_layer_visualizations = _viz_mod.create_per_layer_visualizations
        create_advanced_visualizations = _viz_mod.create_advanced_visualizations
        create_active_overlap_visualizations = _viz_mod.create_active_overlap_visualizations
        create_subspace_visualizations = _viz_mod.create_subspace_visualizations
        create_activation_visualizations = _viz_mod.create_activation_visualizations
        create_direction_visualizations = _viz_mod.create_direction_visualizations
        create_epoch_dynamics_visualizations = _viz_mod.create_epoch_dynamics_visualizations
        create_baseline_visualizations = _viz_mod.create_baseline_visualizations
        _VIZ_AVAILABLE = True
    except Exception:
        _VIZ_AVAILABLE = False


# ============================================================================
# Random Baseline & Statistical Significance
# ============================================================================

def generate_random_baseline(dim: int, num_pairs: int = 10000, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    cosines = []
    for _ in range(num_pairs):
        v1 = rng.randn(dim)
        v2 = rng.randn(dim)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-10 and n2 > 1e-10:
            cosines.append(np.dot(v1, v2) / (n1 * n2))
    return np.array(cosines)


def compute_statistical_tests(observed_cosines: np.ndarray, random_cosines: np.ndarray) -> dict:
    obs = observed_cosines[~np.isnan(observed_cosines)]
    rand = random_cosines[~np.isnan(random_cosines)]

    if len(obs) < 2 or len(rand) < 2:
        return {'error': 'insufficient data'}

    t_stat, t_pval = sp_stats.ttest_ind(obs, rand, equal_var=False)
    ks_stat, ks_pval = sp_stats.ks_2samp(obs, rand)

    pooled_std = np.sqrt((np.var(obs) + np.var(rand)) / 2)
    cohens_d = (np.mean(obs) - np.mean(rand)) / pooled_std if pooled_std > 1e-10 else 0.0

    return {
        't_test': {'statistic': float(t_stat), 'p_value': float(t_pval)},
        'ks_test': {'statistic': float(ks_stat), 'p_value': float(ks_pval)},
        'effect_size_cohens_d': float(cohens_d),
        'observed': {
            'mean': float(np.mean(obs)), 'std': float(np.std(obs)),
            'median': float(np.median(obs)), 'n': len(obs),
        },
        'random_baseline': {
            'mean': float(np.mean(rand)), 'std': float(np.std(rand)),
            'median': float(np.median(rand)), 'n': len(rand),
            'theoretical_std': 1.0 / np.sqrt(len(rand)),
        },
    }


def detect_bimodality(values: np.ndarray) -> dict:
    vals = values[~np.isnan(values)]
    if len(vals) < 10:
        return {'error': 'insufficient data'}

    excess_kurtosis = float(sp_stats.kurtosis(vals, fisher=True))
    skewness = float(sp_stats.skew(vals))
    neg_frac = float(np.mean(vals < 0))
    pos_frac = float(np.mean(vals > 0))
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))
    spread_ratio = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else float('inf')

    dip_stat = _hartigans_dip(vals)

    is_bimodal_kurtosis = excess_kurtosis < -1.0
    is_bimodal_balance = 0.25 < neg_frac < 0.75
    bimodality_score = 0.0
    if is_bimodal_kurtosis:
        bimodality_score += 0.4
    if is_bimodal_balance and spread_ratio > 10:
        bimodality_score += 0.3
    if dip_stat > 0.05:
        bimodality_score += 0.3

    return {
        'excess_kurtosis': excess_kurtosis,
        'skewness': skewness,
        'dip_statistic': dip_stat,
        'negative_fraction': neg_frac,
        'positive_fraction': pos_frac,
        'mean': mean_val,
        'std': std_val,
        'spread_ratio': spread_ratio,
        'is_bimodal_kurtosis': is_bimodal_kurtosis,
        'bimodality_score': bimodality_score,
        'interpretation': (
            'BIMODAL (strong conflict + cooperation alternating)'
            if bimodality_score >= 0.6
            else 'POSSIBLY BIMODAL (mixed signals)'
            if bimodality_score >= 0.3
            else 'UNIMODAL (consistent direction, near-orthogonal noise)'
        ),
    }


def _hartigans_dip(data: np.ndarray) -> float:
    sorted_data = np.sort(data)
    n = len(sorted_data)
    if n < 4:
        return 0.0

    ecdf = np.arange(1, n + 1) / n

    max_diff = 0.0
    for i in range(n):
        for j in range(i + 1, min(i + max(10, n // 10), n)):
            span = sorted_data[j] - sorted_data[i]
            if span < 1e-15:
                continue
            for k in range(i, j + 1):
                expected = (sorted_data[k] - sorted_data[i]) / span
                expected_cdf = ecdf[i] + expected * (ecdf[j] - ecdf[i])
                diff = abs(ecdf[k] - expected_cdf)
                if diff > max_diff:
                    max_diff = diff
    return float(max_diff)


def run_random_baseline_analysis(stats: dict, output_dir: str) -> dict:
    baseline_results = {
        'statistical_tests': {},
        'bimodality': {},
    }

    total_params = 0
    pg_norms = stats.get('per_group_norms', {})
    if pg_norms:
        total_params = len(pg_norms) * 1000
    else:
        total_params = 100000

    cos_data = stats.get('pairwise_cosine', {})

    print("\nGenerating random baseline for statistical comparison...")
    random_cosines = generate_random_baseline(total_params, num_pairs=10000)
    baseline_results['random_baseline_dim'] = total_params
    baseline_results['random_baseline_cosines'] = {
        'mean': float(np.mean(random_cosines)),
        'std': float(np.std(random_cosines)),
    }

    for pk, cs in cos_data.items():
        vals = cs.get('values', [])
        if not vals:
            continue
        obs = np.array(vals)
        test_result = compute_statistical_tests(obs, random_cosines)
        baseline_results['statistical_tests'][pk] = test_result
        bimod = detect_bimodality(obs)
        baseline_results['bimodality'][pk] = bimod

    os.makedirs(output_dir, exist_ok=True)
    baseline_dir = os.path.join(output_dir, 'baseline_comparison')
    os.makedirs(baseline_dir, exist_ok=True)

    with open(os.path.join(baseline_dir, 'statistical_tests.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("RANDOM BASELINE COMPARISON & BIMODALITY ANALYSIS")
    print("=" * 70)
    print(f"\nRandom baseline: dim={total_params}, mean={np.mean(random_cosines):.6f}, std={np.std(random_cosines):.6f}")

    for pk in sorted(baseline_results['statistical_tests'].keys()):
        st = baseline_results['statistical_tests'][pk]
        bm = baseline_results['bimodality'].get(pk, {})
        print(f"\n  {pk}:")
        if 'error' not in st:
            sig = "***" if st['t_test']['p_value'] < 0.001 else "**" if st['t_test']['p_value'] < 0.01 else "*" if st['t_test']['p_value'] < 0.05 else "n.s."
            print(f"    Observed: mean={st['observed']['mean']:+.6f}, std={st['observed']['std']:.6f} (n={st['observed']['n']})")
            print(f"    t-test vs random: t={st['t_test']['statistic']:.3f}, p={st['t_test']['p_value']:.4f} [{sig}]")
            print(f"    KS-test vs random: D={st['ks_test']['statistic']:.3f}, p={st['ks_test']['p_value']:.4f}")
            print(f"    Effect size (Cohen's d): {st['effect_size_cohens_d']:.4f}")
        if 'error' not in bm:
            print(f"    Bimodality: kurtosis={bm['excess_kurtosis']:.3f}, dip={bm['dip_statistic']:.4f}")
            print(f"    Distribution: {bm['interpretation']}")
            print(f"    Neg/Pos split: {bm['negative_fraction']*100:.1f}% / {bm['positive_fraction']*100:.1f}%")

    print("=" * 70)

    return baseline_results


# ============================================================================
# Multi-Checkpoint Epoch-wise Dynamics
# ============================================================================

def _checkpoint_gpu_worker(
    gpu_id: int,
    config_path: str,
    checkpoint_path: str,
    num_batches: int,
    batch_size: int,
    shared_layers: List[str],
    last_n_layers: Optional[int],
    fp16: bool,
    seed: int,
    batch_offset: int,
    result_path: str,
):
    device = f"cuda:{gpu_id}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.chdir(PROJECT_ROOT)

    from mmcv import Config
    from mmcv.runner import load_checkpoint, wrap_fp16_model

    cfg = Config.fromfile(config_path)
    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = min(4, batch_size)

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = model.to(device)
    model = MMDataParallel(model, device_ids=[gpu_id])

    param_groups, group_meta = get_shared_parameters_grouped(model, shared_layers, last_n_layers)

    dataset = build_dataset(cfg.data.train)
    dataloader = build_dataloader_mmdet(
        dataset, samples_per_gpu=batch_size, workers_per_gpu=min(4, batch_size),
        dist=False, shuffle=True,
    )

    data_iter = iter(dataloader)
    for _ in range(batch_offset):
        try:
            next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            next(data_iter)

    all_results = []
    for batch_idx in range(num_batches):
        try:
            data = next(data_iter)
        except StopIteration:
            break
        try:
            result = analyze_single_batch(
                model, data, param_groups, group_meta, gpu_id,
                fp16=fp16, analysis_mode='full',
                store_gradients=False, selective_eval=True,
            )
            result.pop('task_gradients_all', None)
            result.pop('task_gradients_grouped', None)
            all_results.append(result)
        except Exception as e:
            print(f"[GPU {gpu_id}] batch {batch_idx} failed: {e}")
            continue
        if (batch_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()

    with open(result_path, 'wb') as f:
        pickle.dump({'results': all_results, 'group_meta': dict(group_meta)}, f)
    print(f"[GPU {gpu_id}] Done: {len(all_results)} batches")


def analyze_checkpoints(
    config_path: str,
    checkpoint_paths: List[str],
    checkpoint_labels: List[str],
    num_batches: int,
    batch_size: int,
    device: str,
    shared_layers: List[str],
    fp16: bool = False,
    output_dir: str = 'gradient_dynamics',
    seed: int = 42,
    last_n_layers: Optional[int] = None,
    num_gpus: int = 1,
):
    os.chdir(PROJECT_ROOT)

    checkpoint_stats = []

    for ckpt_idx, (ckpt_path, label) in enumerate(zip(checkpoint_paths, checkpoint_labels)):
        print(f"\n{'='*60}")
        print(f"Analyzing checkpoint [{ckpt_idx+1}/{len(checkpoint_paths)}]: {label}")
        print(f"  Path: {ckpt_path}")
        print(f"  GPUs: {num_gpus}, batch_size: {batch_size}, batches: {num_batches}")
        print(f"{'='*60}")

        if num_gpus > 1:
            import torch.multiprocessing as mp
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass

            batches_per_gpu = num_batches // num_gpus
            remainder = num_batches % num_gpus

            tmp_dir = tempfile.mkdtemp(prefix=f'grad_ckpt_{ckpt_idx}_')
            result_paths = [os.path.join(tmp_dir, f'gpu_{i}.pkl') for i in range(num_gpus)]

            processes = []
            batch_offset = 0
            for gpu_id in range(num_gpus):
                n = batches_per_gpu + (1 if gpu_id < remainder else 0)
                p = mp.Process(
                    target=_checkpoint_gpu_worker,
                    args=(
                        gpu_id, config_path, ckpt_path, n, batch_size,
                        shared_layers, last_n_layers, fp16,
                        seed + gpu_id * 1000, batch_offset, result_paths[gpu_id],
                    ),
                )
                processes.append(p)
                batch_offset += n

            for p in processes:
                p.start()
            for p in processes:
                p.join()

            all_results = []
            group_meta = None
            for rp in result_paths:
                if os.path.exists(rp):
                    with open(rp, 'rb') as f:
                        data = pickle.load(f)
                    all_results.extend(data['results'])
                    if group_meta is None:
                        group_meta = data['group_meta']
                    os.remove(rp)
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

        else:
            from mmcv import Config
            from mmcv.runner import load_checkpoint, wrap_fp16_model

            torch.manual_seed(seed)
            np.random.seed(seed)

            cfg = Config.fromfile(config_path)
            cfg.data.samples_per_gpu = batch_size
            cfg.data.workers_per_gpu = min(4, batch_size)

            gpu_id = int(device.split(':')[1]) if ':' in device else 0

            if ckpt_idx == 0:
                model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
                model.init_weights()

                fp16_cfg = cfg.get('fp16', None)
                if fp16_cfg is not None:
                    wrap_fp16_model(model)

                model = model.to(device)
                model = MMDataParallel(model, device_ids=[gpu_id])
                dataset = build_dataset(cfg.data.train)
                dataloader = build_dataloader_mmdet(
                    dataset, samples_per_gpu=batch_size, workers_per_gpu=min(4, batch_size),
                    dist=False, shuffle=True,
                )

            load_checkpoint(model.module, ckpt_path, map_location='cpu')
            param_groups, group_meta = get_shared_parameters_grouped(model, shared_layers, last_n_layers)

            torch.manual_seed(seed)
            np.random.seed(seed)

            all_results = []
            data_iter = iter(dataloader)
            for batch_idx in range(num_batches):
                try:
                    data = next(data_iter)
                except StopIteration:
                    break
                print(f"  Batch {batch_idx+1}/{num_batches}...", end='\r')
                try:
                    result = analyze_single_batch(
                        model, data, param_groups, group_meta, gpu_id,
                        fp16=fp16, analysis_mode='full',
                        store_gradients=False, selective_eval=True,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"\n  Warning: batch {batch_idx} failed: {e}")
                    continue
                if (batch_idx + 1) % 5 == 0:
                    torch.cuda.empty_cache()

        if all_results:
            ckpt_stat = aggregate_results(all_results, analysis_mode='full')
            ckpt_stat['checkpoint_label'] = label
            ckpt_stat['checkpoint_path'] = ckpt_path
            checkpoint_stats.append(ckpt_stat)
            print(f"\n  Processed {len(all_results)} batches for {label}")

    dynamics_dir = os.path.join(output_dir, 'epoch_dynamics')
    os.makedirs(dynamics_dir, exist_ok=True)

    save_data = []
    for cs in checkpoint_stats:
        entry = {
            'label': cs['checkpoint_label'],
            'pairwise_cosine': {pk: {'mean': v['mean'], 'std': v['std']}
                                 for pk, v in cs.get('pairwise_cosine', {}).items()},
            'conflict_frequency': cs.get('conflict_frequency', {}),
            'gradient_norms': {t: {'mean': v['mean'], 'std': v['std']}
                               for t, v in cs.get('gradient_norms', {}).items()},
        }
        save_data.append(entry)

    with open(os.path.join(dynamics_dir, 'epoch_dynamics.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    if _VIZ_AVAILABLE:
        create_epoch_dynamics_visualizations(checkpoint_stats, dynamics_dir)

    return checkpoint_stats


# ============================================================================
# Recommendations
# ============================================================================

def generate_recommendations(stats, output_dir, focus_task='plan', baseline_results=None):
    lines = []
    lines.append("=" * 70)
    lines.append("GRADIENT CONFLICT ANALYSIS - RECOMMENDATIONS (nuScenes)")
    lines.append("=" * 70)
    lines.append("")

    if baseline_results and 'statistical_tests' in baseline_results:
        lines.append("0. STATISTICAL VALIDATION vs RANDOM BASELINE")
        lines.append("-" * 40)
        any_significant = False
        any_bimodal = False
        for pk in sorted(baseline_results['statistical_tests'].keys()):
            st = baseline_results['statistical_tests'][pk]
            bm = baseline_results.get('bimodality', {}).get(pk, {})
            if 'error' in st:
                continue
            p_val = st['t_test']['p_value']
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            if p_val < 0.05:
                any_significant = True
            lines.append(f"  {pk}: t-test p={p_val:.4f} [{sig}], Cohen's d={st['effect_size_cohens_d']:.4f}")
            if 'error' not in bm:
                interp = bm['interpretation']
                lines.append(f"    Distribution: {interp} (kurtosis={bm['excess_kurtosis']:.3f})")
                if bm['bimodality_score'] >= 0.3:
                    any_bimodal = True
        lines.append("")
        if not any_significant:
            lines.append("  CRITICAL FINDING: No task pair shows statistically significant")
            lines.append("  difference from random baseline.")
        elif any_bimodal:
            lines.append("  FINDING: Some task pairs show bimodal distribution.")
        else:
            lines.append("  FINDING: Statistically significant but unimodal.")
        lines.append("")

    lines.append("1. OVERALL CONFLICT ASSESSMENT")
    lines.append("-" * 40)
    cf = stats.get('conflict_frequency', {})
    high_conflict = [(k, v['ratio']) for k, v in cf.items() if v['ratio'] > 0.3]
    if high_conflict:
        for pair, ratio in sorted(high_conflict, key=lambda x: -x[1]):
            lines.append(f"  HIGH CONFLICT: {pair} (conflict rate: {ratio * 100:.1f}%)")
    else:
        lines.append("  No significant overall gradient conflicts detected (all < 30%).")
    lines.append("")

    lines.append("2. GRADIENT MAGNITUDE IMBALANCE")
    lines.append("-" * 40)
    norms = stats.get('gradient_norms', {})
    if focus_task in norms:
        fn = norms[focus_task]['mean']
        for t, ns in sorted(norms.items(), key=lambda x: -x[1]['mean']):
            ratio = ns['mean'] / fn if fn > 1e-10 else float('inf')
            marker = " << DOMINATES" if ratio > 5 else ""
            lines.append(f"  {t:10s}: norm={ns['mean']:.4f}  (ratio to {focus_task}: {ratio:.1f}x){marker}")
        lines.append("")
        any_imbalance = any(ns['mean'] / fn > 5 for t, ns in norms.items() if t != focus_task and fn > 1e-10)
        if any_imbalance:
            lines.append(f"  RECOMMENDATION: Consider gradient norm balancing (GradNorm, IMTL)")
    lines.append("")

    lines.append(f"3. INTERFERENCE ON {focus_task.upper()}")
    lines.append("-" * 40)
    intf = stats.get('interference', {})
    if intf:
        for t in ['det', 'map', 'motion', 'ego']:
            key = f"{t}_on_{focus_task}"
            if key in intf:
                v = intf[key]['mean']
                severity = "HIGH" if v > 1.0 else "MODERATE" if v > 0.3 else "LOW"
                lines.append(f"  {t:10s} -> {focus_task}: {v:.4f} ({severity})")
    lines.append("")

    ao_stats = stats.get('active_overlap', {})
    if ao_stats:
        lines.append("3.5 ACTIVE-OVERLAP DIAGNOSTIC")
        lines.append("-" * 40)
        for pk in sorted(ao_stats.keys()):
            ao = ao_stats[pk]
            ao_cos = ao.get('overlap_cosine', {})
            ao_ratio = ao.get('overlap_ratio', {})
            interp = ao.get('interpretation_counts', {})
            top_interp = max(interp, key=interp.get) if interp else 'N/A'
            ao_cos_str = f"{ao_cos['mean']:+.4f}" if ao_cos and 'mean' in ao_cos else 'N/A'
            ao_ratio_str = f"{ao_ratio['mean']*100:.1f}%" if ao_ratio and 'mean' in ao_ratio else 'N/A'
            lines.append(f"  {pk:20s}: overlap_cos={ao_cos_str}, overlap_ratio={ao_ratio_str}, "
                         f"dominant={top_interp}")
        lines.append("")

    lines.append("4. PER-LAYER CONFLICT HOTSPOTS")
    lines.append("-" * 40)
    pg_cos = stats.get('per_group_cosine', {})
    pg_cf = stats.get('per_group_conflict_frequency', {})
    if pg_cos:
        all_entries = []
        for gk, pks in pg_cos.items():
            for pk, cs in pks.items():
                cf_ratio = pg_cf.get(gk, {}).get(pk, {}).get('ratio', 0)
                all_entries.append((gk, pk, cs['mean'], cf_ratio))
        all_entries.sort(key=lambda x: x[2])
        lines.append("  Top 15 most conflicting (layer, pair):")
        for gk, pk, mean_cos, cf_ratio in all_entries[:15]:
            lines.append(f"    {gk:25s} | {pk:20s} | cos={mean_cos:+.4f} | conflict={cf_ratio * 100:.1f}%")
    lines.append("")

    lines.append("5. SUGGESTED ACTIONS")
    lines.append("-" * 40)
    suggestions = []

    if high_conflict:
        suggestions.append("- Apply PCGrad or CAGrad to resolve directional conflicts.")
        if pg_cos:
            conflict_layers = set()
            for gk, pks in pg_cf.items():
                for pk, v in pks.items():
                    if v['ratio'] > 0.4:
                        conflict_layers.add(gk)
            if conflict_layers:
                suggestions.append(f"- Focus gradient surgery on: {', '.join(sorted(conflict_layers))}")

    any_imbalance = False
    if focus_task in norms:
        fn = norms[focus_task]['mean']
        any_imbalance = any(ns['mean'] / fn > 5 for t, ns in norms.items() if t != focus_task and fn > 1e-10)
    if any_imbalance:
        suggestions.append(f"- Apply GradNorm or dynamic loss weighting to balance {focus_task} signal.")

    if not suggestions:
        suggestions.append("- No critical issues detected. Current training may be adequate.")

    for s in suggestions:
        lines.append(f"  {s}")
    lines.append("")

    text = '\n'.join(lines)
    filepath = os.path.join(output_dir, 'recommendations.txt')
    with open(filepath, 'w') as f:
        f.write(text)
    print(f"Recommendations saved to {filepath}")
    return text


# ============================================================================
# Debug Mode
# ============================================================================

def debug_mode(model, dataloader, gpu_id, fp16=False):
    print("\n" + "=" * 60)
    print("DEBUG MODE - Inspecting model and loss structure (nuScenes)")
    print("=" * 60)

    decoder = get_decoder(model)
    print(f"\nDecoder operation order ({len(decoder.operation_order)} ops):")
    unique_ops = list(dict.fromkeys(decoder.operation_order))
    print(f"  Unique operations: {unique_ops}")

    dec_mapping = compute_decoder_layer_mapping(decoder.operation_order)
    num_decoders = max(dec_mapping.values()) + 1
    print(f"  Number of decoder layers: {num_decoders}")
    for d in range(num_decoders):
        ops_in_dec = [op for i, op in enumerate(decoder.operation_order) if dec_mapping[i] == d]
        print(f"  Dec{d}: {ops_in_dec}")

    data = next(iter(dataloader))
    data_gpu = scatter_data_to_device(data, gpu_id)
    raw_model = model.module if hasattr(model, 'module') else model
    img = data_gpu.pop('img')

    model.train()
    outputs = raw_model.forward_train(img.float(), **data_gpu)

    print(f"\nLoss keys returned by model ({len(outputs)} total):")
    for key, val in sorted(outputs.items()):
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, requires_grad={val.requires_grad}, value={val.item():.4f}")
        else:
            print(f"  {key}: {type(val)}")

    print("\n" + "-" * 40)
    print("Task groups mapping:")
    for task, prefixes in TASK_GROUPS.items():
        matched = [k for k in outputs.keys() if match_loss_key(k, prefixes)]
        print(f"  {task}: {len(matched)} losses matched")
        for m in matched[:5]:
            print(f"    - {m}")
        if len(matched) > 5:
            print(f"    ... and {len(matched) - 5} more")

    print("=" * 60 + "\n")


# ============================================================================
# Multi-GPU Parallel Analysis
# ============================================================================

def _gpu_worker(
    gpu_id: int,
    config_path: str,
    checkpoint_path: str,
    num_batches: int,
    batch_size: int,
    shared_layers: List[str],
    last_n_layers: Optional[int],
    fp16: bool,
    analysis_mode: str,
    store_gradients: bool,
    selective_eval: bool,
    seed: int,
    batch_offset: int,
    result_path: str,
    modular: bool = False,
):
    device = f"cuda:{gpu_id}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.chdir(PROJECT_ROOT)

    from mmcv import Config
    from mmcv.runner import load_checkpoint, wrap_fp16_model

    cfg = Config.fromfile(config_path)
    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = min(4, batch_size)

    print(f"[GPU {gpu_id}] Building model...")
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = model.to(device)
    model = MMDataParallel(model, device_ids=[gpu_id])

    param_groups, group_meta = get_shared_parameters_grouped(
        model, shared_layers, last_n_layers, modular=modular)

    dataset = build_dataset(cfg.data.train)
    dataloader = build_dataloader_mmdet(
        dataset, samples_per_gpu=batch_size, workers_per_gpu=min(4, batch_size),
        dist=False, shuffle=True,
    )

    print(f"[GPU {gpu_id}] Processing {num_batches} batches (offset={batch_offset}, "
          f"batch_size={batch_size}, samples={num_batches * batch_size})...")

    data_iter = iter(dataloader)
    for _ in range(batch_offset):
        try:
            next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            next(data_iter)

    all_results = []
    for batch_idx in range(num_batches):
        try:
            data = next(data_iter)
        except StopIteration:
            break

        print(f"[GPU {gpu_id}] Batch {batch_idx+1}/{num_batches}...", end='\r')
        try:
            result = analyze_single_batch(
                model, data, param_groups, group_meta, gpu_id,
                fp16=fp16, analysis_mode=analysis_mode,
                store_gradients=store_gradients,
                selective_eval=selective_eval,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n[GPU {gpu_id}] Warning: batch {batch_idx} failed: {e}")
            continue

        if (batch_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()

    print(f"\n[GPU {gpu_id}] Done. Processed {len(all_results)} batches.")

    for r in all_results:
        r.pop('task_gradients_all', None)
        r.pop('task_gradients_grouped', None)

    with open(result_path, 'wb') as f:
        pickle.dump({'results': all_results, 'group_meta': dict(group_meta)}, f)


def run_multi_gpu_analysis(args):
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    num_gpus = args.num_gpus
    total_batches = args.num_batches
    batches_per_gpu = total_batches // num_gpus
    remainder = total_batches % num_gpus

    batch_size = args.batch_size
    store_gradients = (args.analysis_mode == 'direction')

    print(f"\n{'='*60}")
    print(f"MULTI-GPU GRADIENT ANALYSIS (nuScenes)")
    print(f"  GPUs: {num_gpus}")
    print(f"  Batch size per GPU: {batch_size}")
    print(f"  Total batches: {total_batches} ({batches_per_gpu} per GPU + {remainder} remainder)")
    print(f"  Total samples: {total_batches * batch_size}")
    print(f"  Analysis mode: {args.analysis_mode}")
    print(f"{'='*60}\n")

    tmp_dir = tempfile.mkdtemp(prefix='grad_analysis_nusc_')
    result_paths = [os.path.join(tmp_dir, f'gpu_{i}.pkl') for i in range(num_gpus)]

    processes = []
    batch_offset = 0
    for gpu_id in range(num_gpus):
        n = batches_per_gpu + (1 if gpu_id < remainder else 0)
        p = mp.Process(
            target=_gpu_worker,
            args=(
                gpu_id, args.config, args.checkpoint, n, batch_size,
                args.shared_layers, args.last_n_layers, args.fp16,
                args.analysis_mode, store_gradients,
                not args.no_selective_eval, args.seed + gpu_id * 1000,
                batch_offset, result_paths[gpu_id],
            ),
            kwargs={'modular': getattr(args, 'modular', False)},
        )
        processes.append(p)
        batch_offset += n

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("\nMerging results from all GPUs...")
    all_results = []
    group_meta = None
    for rp in result_paths:
        if os.path.exists(rp):
            with open(rp, 'rb') as f:
                data = pickle.load(f)
            all_results.extend(data['results'])
            if group_meta is None:
                group_meta = OrderedDict(data['group_meta'])
            os.remove(rp)

    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"Merged {len(all_results)} batches from {num_gpus} GPUs")
    return all_results, group_meta


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze gradient conflicts in HiP-AD (nuScenes)')
    parser.add_argument('--config', type=str,
                        default='projects/configs/hipad_nusc_stage2.py',
                        help='Config file path (default: nuScenes stage2)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint file path (required for single-checkpoint mode)')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batches to analyze (default: 100)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size per GPU. If not specified, reads samples_per_gpu from config.')
    parser.add_argument('--output-dir', type=str, default='gradient_analysis_results_nusc',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (single-GPU mode)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs for parallel analysis.')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use FP16')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                        help='Disable FP16')
    parser.add_argument('--shared-layers', type=str, nargs='+',
                        default=['backbone', 'neck', 'inter_gnn', 'ffn', 'norm', 'fc_before', 'fc_after'],
                        help='Names of shared layers to analyze')
    parser.add_argument('--last-n-layers', type=int, default=None,
                        help='Only analyze last N decoder layers (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to print loss keys and exit')
    parser.add_argument('--analysis-mode', type=str, default='full',
                        choices=['basic', 'full', 'direction'],
                        help='basic: overall only; full: +per-layer+advanced; direction: +PCA')
    parser.add_argument('--focus-task', type=str, default='plan',
                        help='Task to focus planning-centric analysis on')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip visualization, only save numerical results')
    parser.add_argument('--no-selective-eval', action='store_true',
                        help='Disable selective eval')
    parser.add_argument('--random-baseline', action='store_true',
                        help='Run random baseline comparison')
    parser.add_argument('--activation', action='store_true',
                        help='Run activation gradient analysis')
    parser.add_argument('--activation-batches', type=int, default=None,
                        help='Number of batches for activation analysis')
    parser.add_argument('--modular', action='store_true', default=True,
                        help='MGCM-style fine-grained module decomposition')
    parser.add_argument('--no-modular', dest='modular', action='store_false',
                        help='Disable MGCM-style modular decomposition')
    parser.add_argument('--num-splits', type=int, default=1,
                        help='Split collected batches into N equal sets')
    parser.add_argument('--checkpoints', type=str, nargs='+', default=None,
                        help='Multiple checkpoint paths for epoch-wise dynamics analysis')
    parser.add_argument('--checkpoint-labels', type=str, nargs='+', default=None,
                        help='Labels for each checkpoint (e.g., epoch_1 epoch_5 epoch_10)')
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    if not args.checkpoint and not args.checkpoints:
        print("Error: either --checkpoint or --checkpoints is required.")
        sys.exit(1)

    # Resolve batch_size early
    if args.batch_size is None:
        _cfg = Config.fromfile(args.config)
        args.batch_size = getattr(_cfg.data, 'samples_per_gpu', 6)
        print(f"Auto batch_size from config: {args.batch_size}")

    # Multi-checkpoint mode
    if args.checkpoints:
        ckpt_paths = args.checkpoints
        ckpt_labels = args.checkpoint_labels or [f"ckpt_{i}" for i in range(len(ckpt_paths))]
        if len(ckpt_labels) != len(ckpt_paths):
            ckpt_labels = ckpt_labels + [f"ckpt_{i}" for i in range(len(ckpt_labels), len(ckpt_paths))]

        analyze_checkpoints(
            config_path=args.config,
            checkpoint_paths=ckpt_paths,
            checkpoint_labels=ckpt_labels,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            device=args.device,
            shared_layers=args.shared_layers,
            fp16=args.fp16,
            output_dir=args.output_dir,
            seed=args.seed,
            last_n_layers=args.last_n_layers,
            num_gpus=args.num_gpus,
        )
        print(f"\nMulti-checkpoint analysis complete! Results saved to: {args.output_dir}/epoch_dynamics/")
        return

    store_gradients = (args.analysis_mode == 'direction')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.chdir(PROJECT_ROOT)

    cfg = Config.fromfile(args.config)

    if args.batch_size is None:
        args.batch_size = getattr(cfg.data, 'samples_per_gpu', 6)
        print(f"Auto batch_size from config: {args.batch_size}")

    # Multi-GPU path
    if args.num_gpus > 1:
        all_results, group_meta = run_multi_gpu_analysis(args)
        if not all_results:
            print("Error: No batches were processed successfully")
            return
        stats = aggregate_results(all_results, analysis_mode=args.analysis_mode)
    else:
        # Single-GPU path
        cfg.data.samples_per_gpu = args.batch_size
        cfg.data.workers_per_gpu = min(4, args.batch_size)

        print(f"Loading config from: {args.config}")
        print(f"Loading checkpoint from: {args.checkpoint}")
        print(f"Analysis mode: {args.analysis_mode}")
        print(f"FP16: {args.fp16}")
        print(f"Analyzing {args.num_batches} batches x batch_size {args.batch_size} "
              f"= {args.num_batches * args.batch_size} samples")

        model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.init_weights()

        # nuScenes: apply wrap_fp16_model if config has fp16
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
            print("Applied wrap_fp16_model from config fp16 settings")

        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        print("Checkpoint loaded successfully")

        gpu_id = int(args.device.split(':')[1]) if ':' in args.device else 0
        model = model.to(args.device)
        model = MMDataParallel(model, device_ids=[gpu_id])

        param_groups, group_meta = get_shared_parameters_grouped(
            model, args.shared_layers, args.last_n_layers,
            modular=getattr(args, 'modular', False),
        )

        total_groups = len(param_groups)
        total_params = sum(p.numel() for grp in param_groups.values() for p in grp.values())
        print(f"\nFound {total_groups} parameter groups ({total_params:,} parameters):")
        for gk, params in param_groups.items():
            n = sum(p.numel() for p in params.values())
            meta = group_meta.get(gk, {})
            print(f"  {gk:30s}: {n:>8,} params  (dec={meta.get('decoder_idx', 'N/A')}, op={meta.get('op_type', 'N/A')})")

        if total_params == 0:
            print("\nWarning: No shared parameters found! Check --shared-layers argument.")
            return

        # nuScenes: use mmdet's build_dataset and build_dataloader
        dataset = build_dataset(cfg.data.train)
        dataloader = build_dataloader_mmdet(
            dataset, samples_per_gpu=args.batch_size,
            workers_per_gpu=min(4, args.batch_size),
            dist=False, shuffle=True,
        )

        print(f"\nDataset size: {len(dataset)}")

        if args.debug:
            debug_mode(model, dataloader, gpu_id, args.fp16)
            return

        print(f"\nStarting gradient conflict analysis...\n")

        all_results = []
        for batch_idx, data in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break

            print(f"Processing batch {batch_idx + 1}/{args.num_batches}...", end='\r')

            try:
                result = analyze_single_batch(
                    model, data, param_groups, group_meta, gpu_id,
                    fp16=args.fp16, analysis_mode=args.analysis_mode,
                    store_gradients=store_gradients,
                    selective_eval=not args.no_selective_eval,
                )
                all_results.append(result)
            except Exception as e:
                print(f"\nWarning: Failed to process batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

            if (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()

        print(f"\nProcessed {len(all_results)} batches successfully")

        if not all_results:
            print("Error: No batches were processed successfully")
            return

        stats = aggregate_results(all_results, analysis_mode=args.analysis_mode)

    # Split analysis
    num_splits = getattr(args, 'num_splits', 1)
    if num_splits > 1:
        split_size = len(all_results) // num_splits
        splits = []
        for s in range(num_splits):
            start = s * split_size
            end = start + split_size if s < num_splits - 1 else len(all_results)
            splits.append(all_results[start:end])
        print(f"\n{'='*60}")
        print(f"SPLIT ANALYSIS: {num_splits} sets of ~{split_size} batches each")
        print(f"{'='*60}")
    else:
        splits = [all_results]

    for split_idx, split_results in enumerate(splits):
        if num_splits > 1:
            split_label = f"split{split_idx+1}"
            split_output_dir = os.path.join(args.output_dir, split_label)
            print(f"\n{'='*60}")
            print(f"[{split_label}] Aggregating batches {split_idx*split_size+1}~"
                  f"{split_idx*split_size+len(split_results)} "
                  f"({len(split_results)} batches)")
            print(f"{'='*60}")
        else:
            split_label = None
            split_output_dir = args.output_dir

        split_stats = aggregate_results(split_results, analysis_mode=args.analysis_mode)

        print_summary(split_stats, analysis_mode=args.analysis_mode, focus_task=args.focus_task)
        save_results(split_stats, split_output_dir, analysis_mode=args.analysis_mode)

        if not args.no_plots and _VIZ_AVAILABLE:
            try:
                create_basic_visualizations(split_stats, split_output_dir)
            except Exception as e:
                print(f"Warning: Failed to create basic visualizations: {e}")

            if args.analysis_mode in ('full', 'direction'):
                try:
                    create_per_layer_visualizations(split_stats, group_meta, split_output_dir)
                except Exception as e:
                    print(f"Warning: Failed to create per-layer visualizations: {e}")

                try:
                    create_advanced_visualizations(split_stats, group_meta, split_output_dir, focus_task=args.focus_task)
                except Exception as e:
                    print(f"Warning: Failed to create advanced visualizations: {e}")

                try:
                    create_active_overlap_visualizations(split_stats, group_meta, split_output_dir)
                except Exception as e:
                    print(f"Warning: Failed to create active-overlap visualizations: {e}")

            if args.analysis_mode == 'direction':
                try:
                    create_direction_visualizations(split_stats, split_output_dir, group_meta=group_meta)
                except Exception as e:
                    print(f"Warning: Failed to create direction visualizations: {e}")

        # Random baseline comparison
        baseline_results = None
        if args.random_baseline:
            try:
                baseline_results = run_random_baseline_analysis(split_stats, split_output_dir)
                if not args.no_plots and _VIZ_AVAILABLE:
                    create_baseline_visualizations(split_stats, baseline_results, split_output_dir)
            except Exception as e:
                print(f"Warning: Failed to run random baseline analysis: {e}")
                import traceback
                traceback.print_exc()

        # Generate recommendations
        if args.analysis_mode in ('full', 'direction'):
            try:
                rec = generate_recommendations(
                    split_stats, split_output_dir, focus_task=args.focus_task,
                    baseline_results=baseline_results,
                )
                print("\n" + rec)
            except Exception as e:
                print(f"Warning: Failed to generate recommendations: {e}")

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
