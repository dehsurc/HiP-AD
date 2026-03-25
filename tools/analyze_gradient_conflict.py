#!/usr/bin/env python
"""
Gradient Conflict Analysis Script for HiP-AD Multi-Task Learning

This script analyzes gradient conflicts between different tasks (planning, detection, map, motion)
in the shared decoder parameters. It computes pairwise cosine similarity between task gradients
to diagnose potential gradient conflicts before applying PCGrad or similar methods.

Analysis modes:
  basic     - Overall pairwise cosine similarity across all shared parameters (original)
  full      - Per-layer analysis + advanced metrics (interference, decomposition)
  direction - PCA-based gradient direction visualization + all of 'full'

Usage:
    python tools/analyze_gradient_conflict.py \
        --config projects/configs/hipad_b2d_stage2.py \
        --checkpoint work_dirs/hipad_b2d_stage2/latest.pth \
        --num-batches 50 \
        --batch-size 1 \
        --analysis-mode full \
        --output-dir gradient_analysis_results
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
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, MMDataParallel
from mmdet.models import build_detector

# Import HiP-AD custom modules
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.datasets.builder import custom_build_dataset
from projects.mmdet3d_plugin.models.blocks import DeformableFeatureAggregation


# ============================================================================
# Constants
# ============================================================================

TASK_GROUPS = {
    'det': ['det_loss_cls', 'det_loss_box', 'det_loss_cns', 'det_loss_yns'],
    'map': ['map_loss_cls', 'map_loss_line'],
    'motion': ['motion_loss_cls', 'motion_loss_reg'],
    'plan': ['plan_loss_temp_cls', 'plan_loss_temp_reg',
             'plan_loss_spat_cls', 'plan_loss_spat_reg',
             'plan_loss_speed_cls', 'plan_loss_speed_reg'],
}

TASK_PAIRS = [
    ('plan', 'det'),
    ('plan', 'map'),
    ('plan', 'motion'),
    ('det', 'map'),
    ('det', 'motion'),
    ('map', 'motion'),
]

TASK_COLORS = {
    'det': '#e74c3c',
    'map': '#2ecc71',
    'motion': '#3498db',
    'plan': '#f39c12',
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


def compute_interference(grad_source: torch.Tensor, grad_victim: torch.Tensor) -> float:
    """Interference of source on victim: ||g_src|| * max(0, -cos(g_src, g_vic))."""
    cos = compute_cosine_similarity(grad_source, grad_victim)
    if np.isnan(cos):
        return float('nan')
    return torch.norm(grad_source).item() * max(0.0, -cos)


def compute_decomposition(grad_a: torch.Tensor, grad_b: torch.Tensor) -> Tuple[float, float]:
    """Decompose grad_a into cooperative and conflicting components w.r.t. grad_b.

    Returns (cooperative_magnitude, conflicting_magnitude).
    Cooperative: projection of a onto b when positive.
    Conflicting: projection of a onto b when negative (magnitude of opposing component).
    """
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


def move_to_device(data: dict, device: str) -> dict:
    result = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, list):
            result[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        elif isinstance(value, dict):
            result[key] = move_to_device(value, device)
        else:
            result[key] = value
    return result


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
    """Map each operation index to its decoder layer index (0-based).

    Each decoder layer ends with a 'refine' operation.
    """
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

def get_shared_parameters_grouped(
    model: nn.Module,
    shared_layer_names: List[str],
    last_n_layers: Optional[int] = None,
) -> Tuple[OrderedDict, OrderedDict]:
    """Extract shared parameters grouped by (decoder_layer, operation_type).

    Returns:
        param_groups: OrderedDict[group_key -> OrderedDict[param_name -> Parameter]]
        group_meta:   OrderedDict[group_key -> dict] with keys:
                        decoder_idx, op_type, global_layer_idx (or -1 for fc)
    """
    decoder = get_decoder(model)
    operation_order = decoder.operation_order
    dec_mapping = compute_decoder_layer_mapping(operation_order)

    if last_n_layers is not None:
        refine_indices = [i for i, op in enumerate(operation_order) if op == 'refine']
        start_idx = refine_indices[-last_n_layers] if last_n_layers <= len(refine_indices) else 0
    else:
        start_idx = 0

    # Track how many times each op type appears within a decoder layer to disambiguate
    # e.g. two 'norm' ops in the same decoder layer -> dec0_norm_0, dec0_norm_1
    op_count_per_dec: Dict[Tuple[int, str], int] = defaultdict(int)

    param_groups = OrderedDict()
    group_meta = OrderedDict()

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
    """Compute per-group gradient vectors for a task. Single autograd.grad call.

    Returns dict mapping group_key -> flattened gradient tensor (on same device).
    Also includes '_all' key for full concatenated gradient.
    Returns None if no matching losses.
    """
    task_loss = _sum_task_loss(loss_dict, task_name)
    if task_loss is None:
        return None

    # Build ordered list of all parameters across groups, tracking group boundaries
    all_params = []
    group_boundaries = []  # (group_key, start_idx, end_idx)
    for gk, params in param_groups.items():
        start = len(all_params)
        all_params.extend(params.values())
        group_boundaries.append((gk, start, len(all_params)))

    if not all_params:
        return None

    try:
        grads = torch.autograd.grad(
            outputs=task_loss,
            inputs=all_params,
            retain_graph=retain_graph,
            allow_unused=True,
            create_graph=False,
        )
    except RuntimeError as e:
        print(f"Warning: gradient computation failed for {task_name}: {e}")
        return None

    # Replace None grads with zeros
    device = all_params[0].device
    dtype = all_params[0].dtype
    grad_flat = []
    for idx, g in enumerate(grads):
        if g is not None:
            grad_flat.append(g.detach().flatten())
        else:
            grad_flat.append(torch.zeros(all_params[idx].numel(), device=device, dtype=dtype))

    full_grad = torch.cat(grad_flat)

    # Split into groups
    result = {}
    cum = 0
    for gk, params in param_groups.items():
        n = sum(p.numel() for p in params.values())
        result[gk] = full_grad[cum:cum + n]
        cum += n
    result['_all'] = full_grad

    return result


def compute_task_gradient(model, loss_dict, task_name, shared_params, retain_graph=True):
    """Backward-compatible: returns single flat gradient tensor."""
    # Build a single-group wrapper
    wrapper = OrderedDict([('_flat', shared_params)])
    result = compute_task_gradient_grouped(model, loss_dict, task_name, wrapper, retain_graph)
    if result is None:
        return None
    return result['_all']


# ============================================================================
# Per-Group Metrics
# ============================================================================

def compute_per_group_metrics(task_grads_grouped, group_keys):
    """Compute cosine, norms, interference, decomposition for all groups and task pairs.

    Args:
        task_grads_grouped: dict[task_name -> dict[group_key -> tensor]]
        group_keys: list of group keys to analyze
    """
    tasks = list(task_grads_grouped.keys())
    per_group_cosine = {}
    per_group_norms = {}
    per_group_conflict = {}
    per_group_interference = {}
    per_group_decomposition = {}

    for gk in group_keys:
        cosines = {}
        norms = {}
        conflicts = {}
        interferences = {}
        decompositions = {}

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

        per_group_cosine[gk] = cosines
        per_group_norms[gk] = norms
        per_group_conflict[gk] = conflicts
        per_group_interference[gk] = interferences
        per_group_decomposition[gk] = decompositions

    return per_group_cosine, per_group_norms, per_group_conflict, per_group_interference, per_group_decomposition


# ============================================================================
# Single Batch Analysis
# ============================================================================

def analyze_single_batch(
    model, data, param_groups, group_meta, device,
    fp16=False, analysis_mode='full', store_gradients=False,
    selective_eval=True,
):
    results = {
        'pairwise_cosine': {},
        'gradient_norms': {},
        'conflict_flags': {},
    }

    data = move_to_device(data, device)
    model.train()

    _ctx = _selective_eval(model) if selective_eval else contextlib.nullcontext()
    with _ctx:
        with torch.cuda.amp.autocast(enabled=fp16):
            img = data.pop('img')
            outputs = model(img=img, **data)

        if not isinstance(outputs, dict):
            print("Warning: Model output is not a dictionary of losses")
            return results

        task_names = list(TASK_GROUPS.keys())
        task_grads_grouped = {}
        group_keys = [gk for gk in param_groups.keys()]

        for i, task in enumerate(task_names):
            retain = (i < len(task_names) - 1)
            grad_dict = compute_task_gradient_grouped(model, outputs, task, param_groups, retain_graph=retain)
            if grad_dict is not None:
                # Move to CPU immediately to save GPU memory
                task_grads_grouped[task] = {gk: g.cpu() for gk, g in grad_dict.items()}
                results['gradient_norms'][task] = torch.norm(grad_dict['_all']).item()

    # Overall pairwise cosine
    for t1, t2 in TASK_PAIRS:
        pk = f"{t1}_vs_{t2}"
        g1 = task_grads_grouped.get(t1, {}).get('_all')
        g2 = task_grads_grouped.get(t2, {}).get('_all')
        if g1 is not None and g2 is not None:
            c = compute_cosine_similarity(g1, g2)
            results['pairwise_cosine'][pk] = c
            results['conflict_flags'][pk] = c < 0 if not np.isnan(c) else False

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

    # Per-group metrics
    if analysis_mode in ('full', 'direction'):
        pgc, pgn, pgf, pgi, pgd = compute_per_group_metrics(task_grads_grouped, group_keys)
        results['per_group_cosine'] = pgc
        results['per_group_norms'] = pgn
        results['per_group_conflict'] = pgf
        results['per_group_interference'] = pgi
        results['per_group_decomposition'] = pgd

    # Store raw gradients for PCA
    if store_gradients:
        results['task_gradients_all'] = {t: task_grads_grouped[t]['_all'] for t in task_grads_grouped}
        # Store per-group gradients for flow arrows visualization
        results['task_gradients_grouped'] = {
            t: {gk: g for gk, g in task_grads_grouped[t].items() if gk != '_all'}
            for t in task_grads_grouped
        }

    model.zero_grad(set_to_none=True)
    return results


# ============================================================================
# Aggregation
# ============================================================================

def _agg_scalar_dict(all_results, key):
    """Aggregate a key that maps to {sub_key: float} across batches."""
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

    # Overall cosine
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

    # Gradient directions (for PCA)
    if analysis_mode == 'direction':
        grad_dirs = defaultdict(list)
        for r in all_results:
            for t, g in r.get('task_gradients_all', {}).items():
                grad_dirs[t].append(g)
        stats['gradient_directions'] = dict(grad_dirs)

        # Per-group gradient directions (for flow arrows)
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

    # Compact JSON (strip raw values lists)
    def strip_values(d):
        if isinstance(d, dict):
            return {k: strip_values(v) for k, v in d.items() if k != 'values'}
        return d

    with open(os.path.join(output_dir, 'gradient_conflict_stats.json'), 'w') as f:
        json.dump(strip_values(stats), f, indent=2, default=str)

    # Overall CSV
    with open(os.path.join(output_dir, 'gradient_conflict_details.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Task Pair', 'Mean Cosine', 'Std Cosine', 'Min', 'Max', 'Conflict Ratio'])
        for pk, cs in stats.get('pairwise_cosine', {}).items():
            cf = stats.get('conflict_frequency', {}).get(pk, {})
            w.writerow([pk, f"{cs['mean']:.4f}", f"{cs['std']:.4f}",
                        f"{cs['min']:.4f}", f"{cs['max']:.4f}", f"{cf.get('ratio', 0):.4f}"])

    with open(os.path.join(output_dir, 'gradient_norms.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Task', 'Mean Norm', 'Std Norm', 'Min', 'Max'])
        for t, ns in stats.get('gradient_norms', {}).items():
            w.writerow([t, f"{ns['mean']:.4f}", f"{ns['std']:.4f}", f"{ns['min']:.4f}", f"{ns['max']:.4f}"])

    # Per-layer CSV
    if analysis_mode in ('full', 'direction') and 'per_group_cosine' in stats:
        layer_dir = os.path.join(output_dir, 'per_layer')
        os.makedirs(layer_dir, exist_ok=True)
        with open(os.path.join(layer_dir, 'per_layer_details.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Group', 'Task Pair', 'Mean Cosine', 'Std Cosine', 'Conflict Ratio'])
            for gk, pks in sorted(stats['per_group_cosine'].items()):
                for pk, cs in sorted(pks.items()):
                    cf = stats.get('per_group_conflict_frequency', {}).get(gk, {}).get(pk, {})
                    w.writerow([gk, pk, f"{cs['mean']:.4f}", f"{cs['std']:.4f}", f"{cf.get('ratio', 0):.4f}"])

    print(f"Results saved to {output_dir}")


def print_summary(stats, analysis_mode='basic', focus_task='plan'):
    print("\n" + "=" * 70)
    print("GRADIENT CONFLICT ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nAnalyzed {stats['num_batches']} batches")

    print("\n--- Pairwise Cosine Similarity (Mean +/- Std) ---")
    for pk, cs in sorted(stats.get('pairwise_cosine', {}).items()):
        cf = stats.get('conflict_frequency', {}).get(pk, {})
        ratio = cf.get('ratio', 0)
        marker = " << CONFLICT" if ratio > 0.3 else ""
        print(f"  {pk:20s}: {cs['mean']:+.4f} +/- {cs['std']:.4f}  "
              f"(conflict rate: {ratio * 100:.1f}%){marker}")

    print("\n--- Gradient Norms (Mean +/- Std) ---")
    for t, ns in sorted(stats.get('gradient_norms', {}).items()):
        print(f"  {t:10s}: {ns['mean']:.4f} +/- {ns['std']:.4f}")

    # Norm imbalance warning
    norms = stats.get('gradient_norms', {})
    if focus_task in norms:
        focus_norm = norms[focus_task]['mean']
        for t, ns in norms.items():
            if t != focus_task and focus_norm > 0:
                ratio = ns['mean'] / focus_norm
                if ratio > 5:
                    print(f"  WARNING: {t} norm is {ratio:.1f}x larger than {focus_task}")

    if analysis_mode in ('full', 'direction'):
        # Interference summary
        intf = stats.get('interference', {})
        if intf:
            print(f"\n--- Interference on '{focus_task}' (higher = worse) ---")
            for t in ['det', 'map', 'motion']:
                key = f"{t}_on_{focus_task}"
                if key in intf:
                    print(f"  {t:10s} -> {focus_task}: {intf[key]['mean']:.4f} +/- {intf[key]['std']:.4f}")

        # Per-layer top conflicts
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

    print("=" * 70 + "\n")


# ============================================================================
# Visualizations — Basic (Original 4 plots)
# ============================================================================

def create_basic_visualizations(stats, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not installed. Skipping visualizations.")
        return

    os.makedirs(output_dir, exist_ok=True)
    task_names = ['plan', 'det', 'map', 'motion']

    # 1. Heatmap
    n = len(task_names)
    mat = np.ones((n, n))
    for i, t1 in enumerate(task_names):
        for j, t2 in enumerate(task_names):
            if i != j:
                pk = f"{t1}_vs_{t2}" if f"{t1}_vs_{t2}" in stats['pairwise_cosine'] else f"{t2}_vs_{t1}"
                if pk in stats['pairwise_cosine']:
                    mat[i, j] = stats['pairwise_cosine'][pk]['mean']
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                xticklabels=task_names, yticklabels=task_names, vmin=-1, vmax=1)
    plt.title('Mean Cosine Similarity between Task Gradients\n(Negative = Conflict)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_heatmap.png'), dpi=150)
    plt.close()

    # 2. Conflict frequency bar chart
    pairs = list(stats.get('conflict_frequency', {}).keys())
    if pairs:
        ratios = [stats['conflict_frequency'][p]['ratio'] for p in pairs]
        colors = ['#e74c3c' if r > 0.5 else '#f39c12' if r > 0.2 else '#2ecc71' for r in ratios]
        plt.figure(figsize=(10, 5))
        plt.bar(pairs, ratios, color=colors)
        plt.axhline(y=0.5, color='r', linestyle='--', label='50% threshold')
        plt.xlabel('Task Pair')
        plt.ylabel('Conflict Frequency (cosine < 0)')
        plt.title('Gradient Conflict Frequency by Task Pair')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'conflict_frequency.png'), dpi=150)
        plt.close()

    # 3. Box plot
    data_plot, labels = [], []
    for pk in stats.get('pairwise_cosine', {}):
        vals = stats['pairwise_cosine'][pk].get('values', [])
        if vals:
            data_plot.append(vals)
            labels.append(pk)
    if data_plot:
        plt.figure(figsize=(12, 5))
        plt.boxplot(data_plot, labels=labels, patch_artist=True)
        plt.axhline(y=0, color='r', linestyle='--', label='Conflict threshold')
        plt.xlabel('Task Pair')
        plt.ylabel('Cosine Similarity')
        plt.title('Distribution of Gradient Cosine Similarities')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cosine_distribution.png'), dpi=150)
        plt.close()

    # 4. Gradient norms
    tasks = list(stats.get('gradient_norms', {}).keys())
    if tasks:
        means = [stats['gradient_norms'][t]['mean'] for t in tasks]
        stds = [stats['gradient_norms'][t]['std'] for t in tasks]
        plt.figure(figsize=(8, 5))
        plt.bar(tasks, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
        plt.xlabel('Task')
        plt.ylabel('Gradient Norm')
        plt.title('Mean Gradient Norm by Task')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_norms.png'), dpi=150)
        plt.close()

    print(f"Basic visualizations saved to {output_dir}")


# ============================================================================
# Visualizations — Per-Layer (Plots 4-8)
# ============================================================================

def create_per_layer_visualizations(stats, group_meta, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not installed.")
        return

    layer_dir = os.path.join(output_dir, 'per_layer')
    os.makedirs(layer_dir, exist_ok=True)

    pg_cos = stats.get('per_group_cosine', {})
    pg_norms = stats.get('per_group_norms', {})
    pg_cf = stats.get('per_group_conflict_frequency', {})
    if not pg_cos:
        return

    # Sort group keys by decoder_idx then global_layer_idx
    sorted_gks = sorted(pg_cos.keys(), key=lambda gk: (
        group_meta.get(gk, {}).get('decoder_idx', 99),
        group_meta.get(gk, {}).get('global_layer_idx', 99),
    ))
    pair_keys = [f"{t1}_vs_{t2}" for t1, t2 in TASK_PAIRS]

    # --- Plot 4: Per-Operation Conflict Heatmap ---
    mat = np.full((len(sorted_gks), len(pair_keys)), np.nan)
    for i, gk in enumerate(sorted_gks):
        for j, pk in enumerate(pair_keys):
            cs = pg_cos.get(gk, {}).get(pk)
            if cs:
                mat[i, j] = cs['mean']

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_gks) * 0.35)))
    sns.heatmap(mat, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                xticklabels=pair_keys, yticklabels=sorted_gks,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title('Per-Operation Cosine Similarity (Negative = Conflict)')
    plt.tight_layout()
    plt.savefig(os.path.join(layer_dir, 'per_operation_conflict_heatmap.png'), dpi=150)
    plt.close()

    # --- Plot 5: Per-Decoder-Layer Conflict (aggregated) ---
    dec_indices = sorted(set(
        group_meta[gk]['decoder_idx'] for gk in sorted_gks if group_meta.get(gk, {}).get('decoder_idx', -1) >= 0
    ))
    # Also add fc_before/fc_after as a virtual decoder layer
    has_fc = any(gk in ('fc_before', 'fc_after') for gk in sorted_gks)

    dec_labels = [f"dec{d}" for d in dec_indices]
    if has_fc:
        dec_labels.append("fc_shared")

    mat_dec = np.full((len(dec_labels), len(pair_keys)), np.nan)
    mat_cf = np.full((len(dec_labels), len(pair_keys)), np.nan)

    for di, dec_idx in enumerate(dec_indices):
        dec_gks = [gk for gk in sorted_gks if group_meta.get(gk, {}).get('decoder_idx') == dec_idx]
        for pj, pk in enumerate(pair_keys):
            vals = []
            cf_vals = []
            for gk in dec_gks:
                cs = pg_cos.get(gk, {}).get(pk)
                if cs:
                    vals.extend(cs.get('values', [cs['mean']]))
                cf = pg_cf.get(gk, {}).get(pk)
                if cf:
                    cf_vals.append(cf['ratio'])
            if vals:
                mat_dec[di, pj] = float(np.mean(vals))
            if cf_vals:
                mat_cf[di, pj] = float(np.mean(cf_vals))

    if has_fc:
        fc_gks = [gk for gk in sorted_gks if gk in ('fc_before', 'fc_after')]
        fi = len(dec_labels) - 1
        for pj, pk in enumerate(pair_keys):
            vals = []
            cf_vals = []
            for gk in fc_gks:
                cs = pg_cos.get(gk, {}).get(pk)
                if cs:
                    vals.extend(cs.get('values', [cs['mean']]))
                cf = pg_cf.get(gk, {}).get(pk)
                if cf:
                    cf_vals.append(cf['ratio'])
            if vals:
                mat_dec[fi, pj] = float(np.mean(vals))
            if cf_vals:
                mat_cf[fi, pj] = float(np.mean(cf_vals))

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(dec_labels) * 0.5)))
    sns.heatmap(mat_dec, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                xticklabels=pair_keys, yticklabels=dec_labels,
                vmin=-1, vmax=1, ax=axes[0], linewidths=0.5)
    axes[0].set_title('Per-Decoder-Layer Mean Cosine')

    sns.heatmap(mat_cf, annot=True, fmt='.1%', cmap='Reds',
                xticklabels=pair_keys, yticklabels=dec_labels,
                vmin=0, vmax=1, ax=axes[1], linewidths=0.5)
    axes[1].set_title('Per-Decoder-Layer Conflict Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(layer_dir, 'per_decoder_layer_conflict.png'), dpi=150)
    plt.close()

    # --- Plot 6: Operation Type Conflict Profile ---
    op_types = sorted(set(group_meta[gk]['op_type'] for gk in sorted_gks))
    mat_op = np.full((len(op_types), len(pair_keys)), np.nan)
    for oi, op in enumerate(op_types):
        op_gks = [gk for gk in sorted_gks if group_meta.get(gk, {}).get('op_type') == op]
        for pj, pk in enumerate(pair_keys):
            vals = []
            for gk in op_gks:
                cs = pg_cos.get(gk, {}).get(pk)
                if cs:
                    vals.extend(cs.get('values', [cs['mean']]))
            if vals:
                mat_op[oi, pj] = float(np.mean(vals))

    plt.figure(figsize=(10, max(4, len(op_types) * 0.6)))
    sns.heatmap(mat_op, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                xticklabels=pair_keys, yticklabels=op_types,
                vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Operation Type Conflict Profile (Aggregated across decoder layers)')
    plt.tight_layout()
    plt.savefig(os.path.join(layer_dir, 'operation_type_conflict.png'), dpi=150)
    plt.close()

    # --- Plot 7: Per-Layer Gradient Norm Dominance ---
    if pg_norms:
        task_names = ['det', 'map', 'motion', 'plan']
        available_tasks = set()
        for gk in sorted_gks:
            available_tasks.update(pg_norms.get(gk, {}).keys())
        task_names = [t for t in task_names if t in available_tasks]

        x = np.arange(len(sorted_gks))
        width = 0.8 / len(task_names)

        fig, ax = plt.subplots(figsize=(max(12, len(sorted_gks) * 0.6), 6))
        for ti, task in enumerate(task_names):
            means = []
            for gk in sorted_gks:
                ns = pg_norms.get(gk, {}).get(task)
                means.append(ns['mean'] if ns else 0)
            ax.bar(x + ti * width, means, width, label=task, color=TASK_COLORS.get(task, 'gray'), alpha=0.8)

        ax.set_xticks(x + width * len(task_names) / 2)
        ax.set_xticklabels(sorted_gks, rotation=90, ha='center', fontsize=7)
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Per-Layer Gradient Norm Dominance')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(layer_dir, 'gradient_norm_dominance.png'), dpi=150)
        plt.close()

    # --- Plot 8: Gradient Norm Ratio Heatmap ---
    if pg_norms:
        mat_ratio = np.full((len(sorted_gks), len(pair_keys)), np.nan)
        for i, gk in enumerate(sorted_gks):
            for j, pk in enumerate(pair_keys):
                t1, t2 = pk.split('_vs_')
                n1 = pg_norms.get(gk, {}).get(t1)
                n2 = pg_norms.get(gk, {}).get(t2)
                if n1 and n2 and n1['mean'] > 1e-10 and n2['mean'] > 1e-10:
                    mat_ratio[i, j] = np.log2(n1['mean'] / n2['mean'])

        plt.figure(figsize=(10, max(6, len(sorted_gks) * 0.35)))
        sns.heatmap(mat_ratio, annot=True, fmt='.1f', cmap='coolwarm', center=0,
                    xticklabels=pair_keys, yticklabels=sorted_gks, linewidths=0.5)
        plt.title('Gradient Norm Ratio log2(task1/task2) per Layer\n(Positive = task1 dominates)')
        plt.tight_layout()
        plt.savefig(os.path.join(layer_dir, 'gradient_norm_ratio.png'), dpi=150)
        plt.close()

    print(f"Per-layer visualizations saved to {layer_dir}")


# ============================================================================
# Visualizations — Advanced (Plots 9-12)
# ============================================================================

def create_advanced_visualizations(stats, group_meta, output_dir, focus_task='plan'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not installed.")
        return

    adv_dir = os.path.join(output_dir, 'advanced')
    os.makedirs(adv_dir, exist_ok=True)

    task_names = ['det', 'map', 'motion', 'plan']

    # --- Plot 9: Interference Magnitude (asymmetric heatmap) ---
    intf = stats.get('interference', {})
    if intf:
        mat = np.zeros((len(task_names), len(task_names)))
        for i, src in enumerate(task_names):
            for j, vic in enumerate(task_names):
                if i != j:
                    key = f"{src}_on_{vic}"
                    if key in intf:
                        mat[i, j] = intf[key]['mean']

        plt.figure(figsize=(8, 6))
        sns.heatmap(mat, annot=True, fmt='.3f', cmap='Reds',
                    xticklabels=[f"{t}\n(victim)" for t in task_names],
                    yticklabels=[f"{t}\n(source)" for t in task_names],
                    linewidths=0.5)
        plt.title('Gradient Interference Magnitude\n||g_src|| * max(0, -cos(g_src, g_vic))')
        plt.tight_layout()
        plt.savefig(os.path.join(adv_dir, 'interference_magnitude.png'), dpi=150)
        plt.close()

    # --- Plot 10: Cooperative vs Conflicting Decomposition per decoder layer ---
    pg_decomp = stats.get('per_group_decomposition', {})
    pg_cos = stats.get('per_group_cosine', {})
    if pg_decomp:
        sorted_gks = sorted(pg_cos.keys(), key=lambda gk: (
            group_meta.get(gk, {}).get('decoder_idx', 99),
            group_meta.get(gk, {}).get('global_layer_idx', 99),
        ))

        # Aggregate by decoder layer for focus_task pairs
        dec_indices = sorted(set(
            group_meta[gk]['decoder_idx'] for gk in sorted_gks if group_meta.get(gk, {}).get('decoder_idx', -1) >= 0
        ))

        aux_tasks = [t for t in task_names if t != focus_task]
        fig, axes = plt.subplots(1, len(aux_tasks), figsize=(6 * len(aux_tasks), 5))
        if len(aux_tasks) == 1:
            axes = [axes]

        for ai, aux in enumerate(aux_tasks):
            dk = f"{aux}_wrt_{focus_task}"
            coop_vals = []
            conf_vals = []
            dec_labels = []
            for d in dec_indices:
                dec_gks = [gk for gk in sorted_gks if group_meta.get(gk, {}).get('decoder_idx') == d]
                coop_sum, conf_sum, cnt = 0, 0, 0
                for gk in dec_gks:
                    decomp = pg_decomp.get(gk, {}).get(dk)
                    if decomp:
                        coop_sum += decomp['cooperative']['mean']
                        conf_sum += decomp['conflicting']['mean']
                        cnt += 1
                if cnt > 0:
                    coop_vals.append(coop_sum / cnt)
                    conf_vals.append(conf_sum / cnt)
                    dec_labels.append(f"dec{d}")

            x = np.arange(len(dec_labels))
            axes[ai].bar(x, coop_vals, 0.6, label='Cooperative', color='#2ecc71', alpha=0.8)
            axes[ai].bar(x, [-c for c in conf_vals], 0.6, label='Conflicting', color='#e74c3c', alpha=0.8)
            axes[ai].set_xticks(x)
            axes[ai].set_xticklabels(dec_labels)
            axes[ai].axhline(0, color='k', linewidth=0.5)
            axes[ai].set_title(f'{aux} gradient w.r.t. {focus_task}')
            axes[ai].set_ylabel('Component Magnitude')
            axes[ai].legend()

        plt.suptitle(f'Cooperative vs Conflicting Gradient Components (focus: {focus_task})', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(adv_dir, 'cooperative_vs_conflicting.png'), dpi=150)
        plt.close()

    # --- Plot 11: Cumulative Gradient Flow ---
    if pg_cos:
        sorted_gks = sorted(pg_cos.keys(), key=lambda gk: (
            group_meta.get(gk, {}).get('decoder_idx', 99),
            group_meta.get(gk, {}).get('global_layer_idx', 99),
        ))
        dec_indices = sorted(set(
            group_meta[gk]['decoder_idx'] for gk in sorted_gks if group_meta.get(gk, {}).get('decoder_idx', -1) >= 0
        ))

        fig, ax = plt.subplots(figsize=(10, 5))
        for pk in [f"{t1}_vs_{t2}" for t1, t2 in TASK_PAIRS]:
            vals = []
            for d in dec_indices:
                dec_gks = [gk for gk in sorted_gks if group_meta.get(gk, {}).get('decoder_idx') == d]
                batch_vals = []
                for gk in dec_gks:
                    cs = pg_cos.get(gk, {}).get(pk)
                    if cs:
                        batch_vals.extend(cs.get('values', [cs['mean']]))
                vals.append(float(np.mean(batch_vals)) if batch_vals else np.nan)

            ax.plot(dec_indices, vals, 'o-', label=pk, linewidth=2, markersize=6)

        ax.axhline(0, color='r', linestyle='--', linewidth=0.8, label='Conflict threshold')
        ax.set_xlabel('Decoder Layer')
        ax.set_ylabel('Mean Cosine Similarity')
        ax.set_title('Gradient Conflict Across Decoder Depth')
        ax.legend(fontsize=8, ncol=2)
        ax.set_xticks(dec_indices)
        ax.set_xticklabels([f"dec{d}" for d in dec_indices])
        plt.tight_layout()
        plt.savefig(os.path.join(adv_dir, 'cumulative_gradient_flow.png'), dpi=150)
        plt.close()

    # --- Plot 12: Planning-Centric Dashboard ---
    if pg_cos and intf:
        aux_tasks = [t for t in task_names if t != focus_task]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        dec_indices_valid = sorted(set(
            group_meta[gk]['decoder_idx'] for gk in pg_cos.keys()
            if group_meta.get(gk, {}).get('decoder_idx', -1) >= 0
        ))

        # Subplot 1: Cosine per decoder layer (plan vs aux)
        ax = axes[0, 0]
        for aux in aux_tasks:
            pk = f"{focus_task}_vs_{aux}" if f"{focus_task}_vs_{aux}" in list(pg_cos.values())[0] else f"{aux}_vs_{focus_task}"
            vals = []
            for d in dec_indices_valid:
                dec_gks = [gk for gk in pg_cos if group_meta.get(gk, {}).get('decoder_idx') == d]
                bv = []
                for gk in dec_gks:
                    cs = pg_cos.get(gk, {}).get(pk)
                    if cs:
                        bv.append(cs['mean'])
                vals.append(float(np.mean(bv)) if bv else np.nan)
            ax.plot(dec_indices_valid, vals, 'o-', label=f"{focus_task} vs {aux}",
                    color=TASK_COLORS.get(aux, 'gray'), linewidth=2)
        ax.axhline(0, color='r', linestyle='--', linewidth=0.8)
        ax.set_title(f'Cosine Similarity: {focus_task} vs auxiliary tasks')
        ax.set_xlabel('Decoder Layer')
        ax.set_ylabel('Mean Cosine')
        ax.legend()

        # Subplot 2: Interference per decoder layer
        ax = axes[0, 1]
        pg_intf = stats.get('per_group_interference', {})
        for aux in aux_tasks:
            ik = f"{aux}_on_{focus_task}"
            vals = []
            for d in dec_indices_valid:
                dec_gks = [gk for gk in pg_intf if group_meta.get(gk, {}).get('decoder_idx') == d]
                bv = []
                for gk in dec_gks:
                    iv = pg_intf.get(gk, {}).get(ik)
                    if iv:
                        bv.append(iv['mean'])
                vals.append(float(np.mean(bv)) if bv else 0)
            ax.plot(dec_indices_valid, vals, 's-', label=f"{aux} -> {focus_task}",
                    color=TASK_COLORS.get(aux, 'gray'), linewidth=2)
        ax.set_title(f'Interference Magnitude on {focus_task}')
        ax.set_xlabel('Decoder Layer')
        ax.set_ylabel('Interference')
        ax.legend()

        # Subplot 3: Helpfulness score = cos * ||g_aux||
        ax = axes[1, 0]
        overall_cos = stats.get('pairwise_cosine', {})
        overall_norms = stats.get('gradient_norms', {})
        for aux in aux_tasks:
            pk = f"{focus_task}_vs_{aux}" if f"{focus_task}_vs_{aux}" in overall_cos else f"{aux}_vs_{focus_task}"
            cs = overall_cos.get(pk, {})
            ns = overall_norms.get(aux, {})
            if cs and ns:
                score = cs['mean'] * ns['mean']
                ax.bar(aux, score, color=TASK_COLORS.get(aux, 'gray'), alpha=0.8)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title(f'Helpfulness Score: cos(g_aux, g_{focus_task}) * ||g_aux||')
        ax.set_ylabel('Score (positive = helpful)')

        # Subplot 4: Summary text
        ax = axes[1, 1]
        ax.axis('off')
        lines = [f"=== {focus_task.upper()}-CENTRIC SUMMARY ===\n"]
        for aux in aux_tasks:
            pk = f"{focus_task}_vs_{aux}" if f"{focus_task}_vs_{aux}" in overall_cos else f"{aux}_vs_{focus_task}"
            cs = overall_cos.get(pk, {})
            cf = stats.get('conflict_frequency', {}).get(pk, {})
            intf_key = f"{aux}_on_{focus_task}"
            iv = intf.get(intf_key, {})
            lines.append(f"{aux}:")
            lines.append(f"  cosine = {cs.get('mean', 'N/A'):+.4f}" if isinstance(cs.get('mean'), float) else f"  cosine = N/A")
            lines.append(f"  conflict rate = {cf.get('ratio', 0) * 100:.1f}%")
            lines.append(f"  interference = {iv.get('mean', 0):.4f}" if isinstance(iv.get('mean'), float) else f"  interference = N/A")
            lines.append("")
        ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')

        plt.suptitle(f'Planning-Centric Gradient Analysis Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(adv_dir, 'planning_centric_dashboard.png'), dpi=150)
        plt.close()

    print(f"Advanced visualizations saved to {adv_dir}")


# ============================================================================
# Visualizations — Direction (Plots 1-3)
# ============================================================================

def create_direction_visualizations(stats, output_dir, group_meta=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        from sklearn.decomposition import PCA
    except ImportError:
        print("Warning: matplotlib or sklearn not installed. Skipping direction visualizations.")
        return

    dir_dir = os.path.join(output_dir, 'direction')
    os.makedirs(dir_dir, exist_ok=True)

    grad_dirs = stats.get('gradient_directions', {})
    if not grad_dirs:
        print("Warning: No gradient directions stored. Use --analysis-mode direction")
        return

    task_names = list(grad_dirs.keys())
    if len(task_names) < 2:
        return

    # Stack all gradients for PCA
    all_grads = []
    all_labels = []
    for task in task_names:
        for g in grad_dirs[task]:
            all_grads.append(g.numpy())
            all_labels.append(task)

    X = np.stack(all_grads)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # --- Plot 1: PCA Gradient Compass ---
    fig, ax = plt.subplots(figsize=(10, 8))

    for task in task_names:
        mask = [l == task for l in all_labels]
        pts = X_2d[mask]
        color = TASK_COLORS.get(task, 'gray')

        # Scatter
        ax.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.3, s=30, label=f'{task} (per batch)')

        # Mean arrow
        mean_pt = pts.mean(axis=0)
        ax.annotate('', xy=mean_pt, xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        ax.text(mean_pt[0], mean_pt[1], f' {task}', fontsize=11, fontweight='bold', color=color)

        # 2-sigma confidence ellipse
        if len(pts) > 2:
            cov = np.cov(pts, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
            w, h = 2 * 2 * np.sqrt(eigenvalues)  # 2-sigma
            ellipse = Ellipse(xy=mean_pt, width=w, height=h, angle=angle,
                              edgecolor=color, facecolor=color, alpha=0.1, linewidth=1.5)
            ax.add_patch(ellipse)

    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    ax.set_title('Task Gradient Direction Compass (PCA 2D Projection)\nArrows = mean direction, Ellipses = 2-sigma')
    ax.legend()
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_dir, 'pca_gradient_compass.png'), dpi=150)
    plt.close()

    # --- Plot 2: Cosine Stability Time Series ---
    num_batches = len(grad_dirs[task_names[0]])
    fig, ax = plt.subplots(figsize=(12, 5))

    for t1, t2 in TASK_PAIRS:
        if t1 not in grad_dirs or t2 not in grad_dirs:
            continue
        cosines = []
        for bi in range(num_batches):
            g1 = grad_dirs[t1][bi]
            g2 = grad_dirs[t2][bi]
            cosines.append(compute_cosine_similarity(g1, g2))

        cosines = np.array(cosines)
        pk = f"{t1}_vs_{t2}"
        ax.plot(range(num_batches), cosines, alpha=0.5, linewidth=1, label=pk)

        # Running mean (window=5)
        if len(cosines) > 5:
            kernel = np.ones(5) / 5
            smoothed = np.convolve(cosines, kernel, mode='valid')
            ax.plot(range(2, 2 + len(smoothed)), smoothed, linewidth=2.5)

    ax.axhline(0, color='r', linestyle='--', linewidth=0.8, label='Conflict threshold')
    ax.set_xlabel('Batch Index')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Gradient Cosine Stability Across Batches')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_dir, 'cosine_stability_timeseries.png'), dpi=150)
    plt.close()

    # --- Plot 3: Angular Histogram ---
    n_pairs = len(TASK_PAIRS)
    cols = min(3, n_pairs)
    rows = math.ceil(n_pairs / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), subplot_kw={'projection': 'polar'})
    axes_flat = np.array(axes).flatten() if n_pairs > 1 else [axes]

    for idx, (t1, t2) in enumerate(TASK_PAIRS):
        if t1 not in grad_dirs or t2 not in grad_dirs:
            continue
        angles = []
        for bi in range(num_batches):
            g1 = grad_dirs[t1][bi]
            g2 = grad_dirs[t2][bi]
            cos = compute_cosine_similarity(g1, g2)
            if not np.isnan(cos):
                angles.append(np.arccos(np.clip(cos, -1, 1)))

        if not angles:
            continue

        ax = axes_flat[idx]
        angles = np.array(angles)
        bins = np.linspace(0, np.pi, 19)
        counts, _ = np.histogram(angles, bins=bins)

        # Plot on half-circle (0 to pi)
        theta = (bins[:-1] + bins[1:]) / 2
        width = bins[1] - bins[0]
        colors = ['#2ecc71' if t < np.pi / 2 else '#e74c3c' for t in theta]
        ax.bar(theta, counts, width=width, color=colors, alpha=0.7)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_title(f'{t1} vs {t2}', fontsize=10, pad=15)
        ax.axvline(np.pi / 2, color='gray', linestyle='--', linewidth=0.8)

    # Hide unused axes
    for idx in range(len(TASK_PAIRS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle('Gradient Angle Distribution (green < 90 = cooperative, red > 90 = conflict)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_dir, 'angular_histogram.png'), dpi=150)
    plt.close()

    # --- Plot 4 & 5: Gradient Arrows 2D (raw + unit normalized) ---
    try:
        plot_gradient_arrows_2d(grad_dirs, dir_dir, use_unit_norm=False)
        plot_gradient_arrows_2d(grad_dirs, dir_dir, use_unit_norm=True)
    except Exception as e:
        print(f"Warning: Failed to create 2D arrow plots: {e}")

    # --- Plot 6: Gradient Arrows 3D ---
    try:
        plot_gradient_arrows_3d(grad_dirs, dir_dir)
    except Exception as e:
        print(f"Warning: Failed to create 3D arrow plot: {e}")

    # --- Plot 7: Pairwise Gradient Arrows ---
    try:
        plot_pairwise_gradient_arrows(grad_dirs, dir_dir)
    except Exception as e:
        print(f"Warning: Failed to create pairwise arrow plot: {e}")

    # --- Plot 8: Gradient Flow Arrows (per decoder layer) ---
    grad_dirs_grouped = stats.get('gradient_directions_grouped', {})
    if grad_dirs_grouped and group_meta:
        try:
            plot_gradient_flow_arrows(grad_dirs_grouped, group_meta, dir_dir)
        except Exception as e:
            print(f"Warning: Failed to create gradient flow arrows: {e}")

    print(f"Direction visualizations saved to {dir_dir}")


# ============================================================================
# Visualizations — Gradient Arrow Plots (Plots 4-8 in direction/)
# ============================================================================

def _prepare_pca_data(grad_dirs, task_names):
    """Stack all task gradients and return (X, labels, task_names)."""
    all_grads = []
    all_labels = []
    for task in task_names:
        for g in grad_dirs[task]:
            all_grads.append(g.numpy() if hasattr(g, 'numpy') else g)
            all_labels.append(task)
    X = np.stack(all_grads)
    return X, all_labels


def plot_gradient_arrows_2d(grad_dirs, output_dir, use_unit_norm=False):
    """PCA 2D arrow plot — each batch as thin arrow, mean as thick arrow."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    task_names = list(grad_dirs.keys())
    if len(task_names) < 2:
        return

    X, all_labels = _prepare_pca_data(grad_dirs, task_names)

    if use_unit_norm:
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
        X = X / norms

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))

    for task in task_names:
        mask = np.array([l == task for l in all_labels])
        pts = X_2d[mask]
        color = TASK_COLORS.get(task, 'gray')

        # Individual batch arrows (thin, transparent)
        for pt in pts:
            ax.annotate('', xy=pt, xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.8, alpha=0.15))

        # Mean arrow (thick, opaque)
        mean_pt = pts.mean(axis=0)
        ax.annotate('', xy=mean_pt, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=3.0, alpha=0.9))
        ax.text(mean_pt[0], mean_pt[1], f'  {task}', fontsize=12, fontweight='bold', color=color,
                ha='left', va='bottom')

    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)
    ax.plot(0, 0, 'ko', markersize=5, zorder=5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    suffix = ' (Unit Normalized)' if use_unit_norm else ' (Raw Gradient)'
    ax.set_title(f'Gradient Direction Arrows — PCA 2D{suffix}\n'
                 f'Thin: per-batch, Thick: mean direction')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(
        [plt.Line2D([0], [0], color=TASK_COLORS.get(t, 'gray'), lw=3) for t in task_names],
        task_names, loc='best'
    )
    plt.tight_layout()
    fname = 'gradient_arrows_2d_unit.png' if use_unit_norm else 'gradient_arrows_2d.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()


def plot_gradient_arrows_3d(grad_dirs, output_dir):
    """PCA 3D arrow plot with three viewpoints (front, side, top)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from sklearn.decomposition import PCA

    task_names = list(grad_dirs.keys())
    if len(task_names) < 2:
        return

    X, all_labels = _prepare_pca_data(grad_dirs, task_names)

    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)

    views = [
        ('Front', 25, -45),
        ('Side', 25, 45),
        ('Top', 80, -45),
    ]

    fig = plt.figure(figsize=(18, 6))

    for vi, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, vi + 1, projection='3d')

        for task in task_names:
            mask = np.array([l == task for l in all_labels])
            pts = X_3d[mask]
            color = TASK_COLORS.get(task, 'gray')

            # Individual batch arrows (thin, transparent)
            for pt in pts:
                ax.quiver(0, 0, 0, pt[0], pt[1], pt[2],
                          arrow_length_ratio=0.1, color=color, alpha=0.1, linewidth=0.5)

            # Mean arrow (thick, opaque)
            mean_pt = pts.mean(axis=0)
            ax.quiver(0, 0, 0, mean_pt[0], mean_pt[1], mean_pt[2],
                      arrow_length_ratio=0.12, color=color, alpha=0.9, linewidth=2.5)
            ax.text(mean_pt[0], mean_pt[1], mean_pt[2], f' {task}',
                    fontsize=9, fontweight='bold', color=color)

        ax.scatter([0], [0], [0], c='black', s=30, zorder=5)
        evr = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)', fontsize=8)
        ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)', fontsize=8)
        ax.set_zlabel(f'PC3 ({evr[2]*100:.1f}%)', fontsize=8)
        ax.set_title(f'{view_name} View', fontsize=10)
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(f'Gradient Direction Arrows — PCA 3D\n'
                 f'Total explained variance: {sum(pca.explained_variance_ratio_[:3])*100:.1f}%',
                 fontsize=13)
    fig.legend(
        [plt.Line2D([0], [0], color=TASK_COLORS.get(t, 'gray'), lw=3) for t in task_names],
        task_names, loc='lower center', ncol=len(task_names), fontsize=10
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(os.path.join(output_dir, 'gradient_arrows_3d.png'), dpi=150)
    plt.close()


def plot_pairwise_gradient_arrows(grad_dirs, output_dir):
    """Pairwise 2D arrow plot — no PCA, direct cosine angle between task pairs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    task_names = list(grad_dirs.keys())
    num_batches = len(grad_dirs[task_names[0]])

    n_pairs = len(TASK_PAIRS)
    cols = min(3, n_pairs)
    rows = math.ceil(n_pairs / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes_flat = np.array(axes).flatten() if n_pairs > 1 else [axes]

    for idx, (t1, t2) in enumerate(TASK_PAIRS):
        if t1 not in grad_dirs or t2 not in grad_dirs:
            continue

        ax = axes_flat[idx]

        # Draw unit circle
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', linewidth=0.5, alpha=0.3)

        # Reference arrow for t1 (fixed on x-axis)
        color1 = TASK_COLORS.get(t1, 'gray')
        ax.annotate('', xy=(1, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color1, lw=2.5))
        ax.text(1.05, 0, f'{t1}', fontsize=10, fontweight='bold', color=color1,
                ha='left', va='center')

        # t2 arrows at angle = arccos(cosine)
        color2 = TASK_COLORS.get(t2, 'gray')
        angles = []
        cosines = []
        for bi in range(num_batches):
            g1 = grad_dirs[t1][bi]
            g2 = grad_dirs[t2][bi]
            cos = compute_cosine_similarity(g1, g2)
            if np.isnan(cos):
                continue
            cosines.append(cos)
            angle = np.arccos(np.clip(cos, -1, 1))
            angles.append(angle)

            # Arrow color: green if cooperative, red if conflicting
            arrow_color = '#2ecc71' if cos > 0 else '#e74c3c'
            ax.annotate('', xy=(np.cos(angle), np.sin(angle)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, lw=0.6, alpha=0.25))

        # Mean arrow for t2
        if angles:
            mean_angle = np.mean(angles)
            mean_cos = np.mean(cosines)
            mean_color = '#2ecc71' if mean_cos > 0 else '#e74c3c'
            mx, my = np.cos(mean_angle), np.sin(mean_angle)
            ax.annotate('', xy=(mx, my), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=mean_color, lw=3.0, alpha=0.9))
            ax.text(mx, my, f'  {t2}\n  (cos={mean_cos:.3f})', fontsize=9, fontweight='bold',
                    color=color2, ha='left', va='bottom')

        # 90-degree conflict line
        ax.plot([0, 0], [0, 1.1], 'r--', linewidth=0.8, alpha=0.5)
        ax.text(0.02, 1.05, '90°', fontsize=8, color='red', alpha=0.7)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', linewidth=0.3)
        ax.axvline(0, color='gray', linewidth=0.3)
        ax.plot(0, 0, 'ko', markersize=4, zorder=5)
        ax.set_title(f'{t1} vs {t2}', fontsize=11)
        ax.set_xlabel('cos(θ) direction')
        ax.set_ylabel('sin(θ) direction')

    # Hide unused axes
    for idx in range(len(TASK_PAIRS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('Pairwise Gradient Angle Arrows\n'
                 'Green = cooperative (cos > 0), Red = conflicting (cos < 0)',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(output_dir, 'pairwise_gradient_arrows.png'), dpi=150)
    plt.close()


def plot_gradient_flow_arrows(grad_dirs_grouped, group_meta, output_dir):
    """Per-decoder-layer gradient direction arrows using shared PCA space."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    task_names = list(grad_dirs_grouped.keys())
    if len(task_names) < 2:
        return

    # Collect decoder layer indices from group_meta
    dec_groups = defaultdict(list)
    for gk, meta in group_meta.items():
        dec_idx = meta.get('decoder_idx', -1)
        if dec_idx >= 0:
            dec_groups[dec_idx].append(gk)
    dec_indices = sorted(dec_groups.keys())

    if not dec_indices:
        return

    # For each decoder layer, concatenate all group gradients belonging to that layer
    # Build per-task, per-decoder-layer gradient vectors
    # task -> dec_idx -> list of gradient vectors (one per batch)
    task_dec_grads = defaultdict(lambda: defaultdict(list))
    num_batches = None
    for task in task_names:
        for dec_idx in dec_indices:
            gks = dec_groups[dec_idx]
            # Determine batch count from first available group
            for gk in gks:
                if gk in grad_dirs_grouped[task]:
                    if num_batches is None:
                        num_batches = len(grad_dirs_grouped[task][gk])
                    break

            if num_batches is None:
                continue

            for bi in range(num_batches):
                parts = []
                for gk in gks:
                    if gk in grad_dirs_grouped[task]:
                        g = grad_dirs_grouped[task][gk][bi]
                        parts.append(g.numpy() if hasattr(g, 'numpy') else g)
                if parts:
                    task_dec_grads[task][dec_idx].append(np.concatenate(parts))

    if num_batches is None:
        return

    # Fit PCA once across all layers and tasks
    all_vecs = []
    for task in task_names:
        for dec_idx in dec_indices:
            for v in task_dec_grads[task][dec_idx]:
                all_vecs.append(v)

    if not all_vecs:
        return

    # Pad vectors to same length (different layers may have different param counts)
    max_dim = max(v.shape[0] for v in all_vecs)
    all_vecs_padded = []
    for v in all_vecs:
        if v.shape[0] < max_dim:
            all_vecs_padded.append(np.pad(v, (0, max_dim - v.shape[0])))
        else:
            all_vecs_padded.append(v)

    X_all = np.stack(all_vecs_padded)
    pca = PCA(n_components=2)
    pca.fit(X_all)

    # Plot: X-axis = decoder layer, arrows at each layer position
    fig, ax = plt.subplots(figsize=(max(10, len(dec_indices) * 2.5), 8))

    # For each decoder layer and task, project mean and individual gradients
    for task in task_names:
        color = TASK_COLORS.get(task, 'gray')
        mean_xs = []
        mean_ys = []
        layer_positions = []

        for dec_idx in dec_indices:
            vecs = task_dec_grads[task].get(dec_idx, [])
            if not vecs:
                continue

            # Pad to max_dim
            vecs_padded = []
            for v in vecs:
                if v.shape[0] < max_dim:
                    vecs_padded.append(np.pad(v, (0, max_dim - v.shape[0])))
                else:
                    vecs_padded.append(v)

            projected = pca.transform(np.stack(vecs_padded))

            # Normalize projected vectors for display
            for pt in projected:
                norm = np.linalg.norm(pt)
                if norm > 1e-10:
                    direction = pt / norm
                else:
                    direction = pt
                # Scale arrows for visibility
                scale = 0.3
                ax.annotate('', xy=(dec_idx + direction[0] * scale, direction[1] * scale),
                            xytext=(dec_idx, 0),
                            arrowprops=dict(arrowstyle='->', color=color, lw=0.5, alpha=0.12))

            # Mean direction
            mean_proj = projected.mean(axis=0)
            norm = np.linalg.norm(mean_proj)
            if norm > 1e-10:
                mean_dir = mean_proj / norm
            else:
                mean_dir = mean_proj
            scale = 0.4
            ax.annotate('', xy=(dec_idx + mean_dir[0] * scale, mean_dir[1] * scale),
                        xytext=(dec_idx, 0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.5, alpha=0.9))

            mean_xs.append(dec_idx + mean_dir[0] * scale)
            mean_ys.append(mean_dir[1] * scale)
            layer_positions.append(dec_idx)

        # Connect mean directions with a trace line
        if len(mean_xs) > 1:
            ax.plot(mean_xs, mean_ys, '--', color=color, alpha=0.4, linewidth=1.0)

    # Decoder layer markers
    for dec_idx in dec_indices:
        ax.axvline(dec_idx, color='gray', linewidth=0.3, alpha=0.5)
        ax.plot(dec_idx, 0, 'ko', markersize=4, zorder=5)

    ax.set_xticks(dec_indices)
    ax.set_xticklabels([f'dec{d}' for d in dec_indices])
    ax.set_xlabel('Decoder Layer')
    ax.set_ylabel('PCA Projected Direction')
    evr = pca.explained_variance_ratio_
    ax.set_title(f'Gradient Direction Flow Across Decoder Layers\n'
                 f'PC1: {evr[0]*100:.1f}%, PC2: {evr[1]*100:.1f}% explained variance')
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.legend(
        [plt.Line2D([0], [0], color=TASK_COLORS.get(t, 'gray'), lw=3) for t in task_names],
        task_names, loc='best'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_flow_arrows.png'), dpi=150)
    plt.close()


# ============================================================================
# Random Baseline & Statistical Significance
# ============================================================================

def generate_random_baseline(dim: int, num_pairs: int = 10000, seed: int = 42) -> np.ndarray:
    """Generate cosine similarities between random vector pairs in R^dim.

    In high-dimensional spaces, random vectors are nearly orthogonal.
    Expected: mean ≈ 0, std ≈ 1/sqrt(dim).
    """
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
    """Compare observed gradient cosines against random baseline.

    Returns dict with:
      - t_test: (statistic, p_value) — are means significantly different?
      - ks_test: (statistic, p_value) — are distributions significantly different?
      - effect_size: Cohen's d
      - observed_stats: mean, std, median
      - random_stats: mean, std, median
    """
    obs = observed_cosines[~np.isnan(observed_cosines)]
    rand = random_cosines[~np.isnan(random_cosines)]

    if len(obs) < 2 or len(rand) < 2:
        return {'error': 'insufficient data'}

    t_stat, t_pval = sp_stats.ttest_ind(obs, rand, equal_var=False)
    ks_stat, ks_pval = sp_stats.ks_2samp(obs, rand)

    # Cohen's d
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
    """Detect whether a distribution is bimodal using Hartigan's dip test approximation
    and kurtosis analysis.

    Returns:
      - kurtosis: negative kurtosis suggests bimodality (platykurtic)
      - dip_statistic: Hartigan's dip test statistic (higher = more bimodal)
      - is_bimodal_kurtosis: True if excess kurtosis < -1
      - negative_fraction: fraction of values < 0
      - positive_fraction: fraction of values > 0
      - spread_ratio: std / |mean| — high ratio with mean ≈ 0 suggests bimodal or high variance
    """
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

    # Hartigan's dip test (simplified approximation using sorted data)
    dip_stat = _hartigans_dip(vals)

    # Heuristic bimodality assessment
    # Bimodal if: excess kurtosis strongly negative AND both tails have significant mass
    is_bimodal_kurtosis = excess_kurtosis < -1.0
    is_bimodal_balance = 0.25 < neg_frac < 0.75  # balanced tails
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
    """Simplified Hartigan's dip test statistic.

    Measures departure from unimodality. Returns dip statistic in [0, 0.5].
    Higher values indicate stronger evidence against unimodality.
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    if n < 4:
        return 0.0

    # Compute empirical CDF
    ecdf = np.arange(1, n + 1) / n

    # Compute greatest convex minorant (GCM) and least concave majorant (LCM)
    # Simplified: compare empirical CDF with best-fit uniform on each interval
    max_diff = 0.0
    for i in range(n):
        for j in range(i + 1, min(i + max(10, n // 10), n)):
            # Linear interpolation between points i and j
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
    """Run full random baseline comparison and bimodality analysis.

    Uses gradient dimensionality from stats to generate appropriate random baseline.
    """
    baseline_results = {
        'statistical_tests': {},
        'bimodality': {},
    }

    # Estimate gradient dimensionality from norms (approximate)
    # Use the per-group parameter counts if available
    total_params = 0
    pg_norms = stats.get('per_group_norms', {})
    if pg_norms:
        total_params = len(pg_norms) * 1000  # rough estimate
    else:
        total_params = 100000  # fallback

    # Try to get actual dimension from raw values
    cos_data = stats.get('pairwise_cosine', {})
    first_pair_vals = None
    for pk, cs in cos_data.items():
        vals = cs.get('values', [])
        if vals:
            first_pair_vals = np.array(vals)
            break

    # Generate random baseline
    print("\nGenerating random baseline for statistical comparison...")
    random_cosines = generate_random_baseline(total_params, num_pairs=10000)
    baseline_results['random_baseline_dim'] = total_params
    baseline_results['random_baseline_cosines'] = {
        'mean': float(np.mean(random_cosines)),
        'std': float(np.std(random_cosines)),
    }

    # Statistical tests for each task pair
    for pk, cs in cos_data.items():
        vals = cs.get('values', [])
        if not vals:
            continue
        obs = np.array(vals)

        # Statistical significance vs random baseline
        test_result = compute_statistical_tests(obs, random_cosines)
        baseline_results['statistical_tests'][pk] = test_result

        # Bimodality analysis
        bimod = detect_bimodality(obs)
        baseline_results['bimodality'][pk] = bimod

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    baseline_dir = os.path.join(output_dir, 'baseline_comparison')
    os.makedirs(baseline_dir, exist_ok=True)

    with open(os.path.join(baseline_dir, 'statistical_tests.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2, default=str)

    # Print summary
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


def create_baseline_visualizations(stats: dict, baseline_results: dict, output_dir: str):
    """Create visualizations comparing observed gradient cosines with random baseline."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("Warning: matplotlib not installed. Skipping baseline visualizations.")
        return

    baseline_dir = os.path.join(output_dir, 'baseline_comparison')
    os.makedirs(baseline_dir, exist_ok=True)

    cos_data = stats.get('pairwise_cosine', {})
    pairs = sorted(cos_data.keys())

    if not pairs:
        return

    # Generate random baseline for overlay
    dim = baseline_results.get('random_baseline_dim', 100000)
    random_cosines = generate_random_baseline(dim, num_pairs=10000)

    # --- Figure 1: KDE + Histogram per task pair with random overlay ---
    n_pairs = len(pairs)
    cols = min(3, n_pairs)
    rows = math.ceil(n_pairs / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes_flat = np.array(axes).flatten() if n_pairs > 1 else [axes]

    for idx, pk in enumerate(pairs):
        ax = axes_flat[idx]
        vals = np.array(cos_data[pk].get('values', []))
        if len(vals) == 0:
            continue

        # Histogram + KDE for observed
        ax.hist(vals, bins=30, density=True, alpha=0.5, color='steelblue', label='Observed', edgecolor='white')

        # KDE
        if len(vals) > 3:
            kde = sp_stats.gaussian_kde(vals)
            x_range = np.linspace(min(vals.min(), -0.2), max(vals.max(), 0.2), 200)
            ax.plot(x_range, kde(x_range), color='steelblue', linewidth=2, label='Observed KDE')

        # Random baseline KDE
        rand_kde = sp_stats.gaussian_kde(random_cosines)
        x_range_rand = np.linspace(-0.15, 0.15, 200)
        ax.plot(x_range_rand, rand_kde(x_range_rand), color='red', linewidth=2, linestyle='--', label='Random baseline')

        # Bimodality info
        bm = baseline_results.get('bimodality', {}).get(pk, {})
        st = baseline_results.get('statistical_tests', {}).get(pk, {})

        title = pk
        if 'error' not in bm:
            title += f"\n{bm['interpretation']}"
        if 'error' not in st:
            p_val = st['t_test']['p_value']
            sig = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
            title += f" | t-test: {sig}"

        ax.set_title(title, fontsize=9)
        ax.axvline(0, color='gray', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)

    for idx in range(n_pairs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('Cosine Similarity Distribution vs Random Baseline\n'
                 '(Bimodal = strong conflict + cooperation alternating; Unimodal = noise-level)',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(baseline_dir, 'cosine_distribution_vs_baseline.png'), dpi=150)
    plt.close()

    # --- Figure 2: Summary comparison bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Mean cosine with CI
    means = [cos_data[pk]['mean'] for pk in pairs]
    stds = [cos_data[pk]['std'] for pk in pairs]
    x = np.arange(len(pairs))
    axes[0].bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7, label='Observed')
    axes[0].axhline(np.mean(random_cosines), color='red', linestyle='--', label=f'Random mean ({np.mean(random_cosines):.4f})')
    axes[0].axhspan(np.mean(random_cosines) - np.std(random_cosines),
                     np.mean(random_cosines) + np.std(random_cosines),
                     alpha=0.1, color='red', label='Random ±1σ')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Mean Cosine Similarity')
    axes[0].set_title('Observed vs Random Baseline')
    axes[0].legend(fontsize=8)

    # Panel 2: p-values from t-test
    p_vals = []
    for pk in pairs:
        st = baseline_results.get('statistical_tests', {}).get(pk, {})
        p_vals.append(st.get('t_test', {}).get('p_value', 1.0))
    colors = ['#e74c3c' if p < 0.05 else '#f39c12' if p < 0.1 else '#2ecc71' for p in p_vals]
    axes[1].bar(x, [-np.log10(max(p, 1e-10)) for p in p_vals], color=colors, alpha=0.7)
    axes[1].axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    axes[1].axhline(-np.log10(0.01), color='darkred', linestyle=':', label='p=0.01')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('-log10(p-value)')
    axes[1].set_title('Statistical Significance (t-test)')
    axes[1].legend(fontsize=8)

    # Panel 3: Bimodality scores
    bm_scores = []
    for pk in pairs:
        bm = baseline_results.get('bimodality', {}).get(pk, {})
        bm_scores.append(bm.get('bimodality_score', 0))
    bm_colors = ['#e74c3c' if s >= 0.6 else '#f39c12' if s >= 0.3 else '#2ecc71' for s in bm_scores]
    axes[2].bar(x, bm_scores, color=bm_colors, alpha=0.7)
    axes[2].axhline(0.6, color='red', linestyle='--', label='Bimodal threshold')
    axes[2].axhline(0.3, color='orange', linestyle=':', label='Possibly bimodal')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
    axes[2].set_ylabel('Bimodality Score')
    axes[2].set_title('Distribution Shape Analysis')
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=8)

    plt.suptitle('Gradient Conflict Statistical Validation Summary', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(baseline_dir, 'statistical_summary.png'), dpi=150)
    plt.close()

    print(f"Baseline comparison visualizations saved to {baseline_dir}")


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
    """Worker for multi-GPU checkpoint analysis. One GPU processes a batch subset."""
    device = f"cuda:{gpu_id}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.chdir(PROJECT_ROOT)

    from mmcv import Config
    from mmcv.runner import load_checkpoint

    cfg = Config.fromfile(config_path)
    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = min(4, batch_size)

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = model.to(device)
    model = MMDataParallel(model, device_ids=[gpu_id])

    param_groups, group_meta = get_shared_parameters_grouped(model, shared_layers, last_n_layers)

    dataset = custom_build_dataset(cfg.data.train)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=min(4, batch_size),
        collate_fn=partial(collate, samples_per_gpu=batch_size), drop_last=True,
    )

    # Skip to batch_offset
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
                model, data, param_groups, group_meta, device,
                fp16=fp16, analysis_mode='full',
                store_gradients=False, selective_eval=True,
            )
            # Strip non-serializable data
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
    """Analyze gradient dynamics across multiple training checkpoints.

    Supports multi-GPU: each checkpoint's batches are split across GPUs.
    """
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

            # Merge results from all GPUs
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
            # Single-GPU path
            from mmcv import Config
            from mmcv.runner import load_checkpoint

            torch.manual_seed(seed)
            np.random.seed(seed)

            cfg = Config.fromfile(config_path)
            cfg.data.samples_per_gpu = batch_size
            cfg.data.workers_per_gpu = min(4, batch_size)

            if ckpt_idx == 0:
                model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
                model.init_weights()
                device_id = int(device.split(':')[1]) if ':' in device else 0
                model = model.to(device)
                model = MMDataParallel(model, device_ids=[device_id])
                dataset = custom_build_dataset(cfg.data.train)
                dataloader = DataLoader(
                    dataset, batch_size=batch_size, shuffle=True,
                    num_workers=min(4, batch_size),
                    collate_fn=partial(collate, samples_per_gpu=batch_size), drop_last=True,
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
                        model, data, param_groups, group_meta, device,
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

    # Save and visualize dynamics
    dynamics_dir = os.path.join(output_dir, 'epoch_dynamics')
    os.makedirs(dynamics_dir, exist_ok=True)

    # Save raw stats
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

    create_epoch_dynamics_visualizations(checkpoint_stats, dynamics_dir)

    return checkpoint_stats


def create_epoch_dynamics_visualizations(checkpoint_stats: list, output_dir: str):
    """Visualize gradient dynamics across training checkpoints."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed.")
        return

    if len(checkpoint_stats) < 2:
        print("Need at least 2 checkpoints for dynamics visualization.")
        return

    labels = [cs['checkpoint_label'] for cs in checkpoint_stats]
    x = np.arange(len(labels))

    # --- Figure 1: Cosine similarity evolution ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Mean cosine per task pair
    ax = axes[0]
    pair_keys = sorted(checkpoint_stats[0].get('pairwise_cosine', {}).keys())
    for pk in pair_keys:
        means = []
        for cs in checkpoint_stats:
            v = cs.get('pairwise_cosine', {}).get(pk, {})
            means.append(v.get('mean', np.nan) if isinstance(v, dict) else np.nan)
        ax.plot(x, means, 'o-', label=pk, linewidth=2, markersize=6)

    ax.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Gradient Cosine Evolution Across Training')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: Conflict frequency evolution
    ax = axes[1]
    for pk in pair_keys:
        ratios = []
        for cs in checkpoint_stats:
            cf = cs.get('conflict_frequency', {}).get(pk, {})
            ratios.append(cf.get('ratio', np.nan) if isinstance(cf, dict) else np.nan)
        ax.plot(x, ratios, 's-', label=pk, linewidth=2, markersize=6)

    ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='Random baseline (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Conflict Frequency')
    ax.set_title('Conflict Frequency Evolution')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Gradient Dynamics Across Training Checkpoints', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(output_dir, 'cosine_evolution.png'), dpi=150)
    plt.close()

    # --- Figure 2: Gradient norm evolution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    task_names = sorted(checkpoint_stats[0].get('gradient_norms', {}).keys())
    for task in task_names:
        means = []
        stds = []
        for cs in checkpoint_stats:
            ns = cs.get('gradient_norms', {}).get(task, {})
            means.append(ns.get('mean', 0) if isinstance(ns, dict) else 0)
            stds.append(ns.get('std', 0) if isinstance(ns, dict) else 0)
        color = TASK_COLORS.get(task, 'gray')
        ax.plot(x, means, 'o-', label=task, color=color, linewidth=2, markersize=6)
        ax.fill_between(x, np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.15, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Task Gradient Magnitude Evolution\n(Shaded = ±1σ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_norm_evolution.png'), dpi=150)
    plt.close()

    # --- Figure 3: Magnitude ratio evolution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'map' in task_names:
        for task in task_names:
            if task == 'map':
                continue
            ratios = []
            for cs in checkpoint_stats:
                n_task = cs.get('gradient_norms', {}).get(task, {}).get('mean', 1)
                n_map = cs.get('gradient_norms', {}).get('map', {}).get('mean', 1)
                ratios.append(n_task / max(n_map, 1e-10))
            color = TASK_COLORS.get(task, 'gray')
            ax.plot(x, ratios, 'o-', label=f'{task}/map', color=color, linewidth=2, markersize=6)

        ax.axhline(1, color='gray', linestyle=':', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Norm Ratio (task / map)')
        ax.set_title('Gradient Magnitude Imbalance vs Map Task')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'magnitude_ratio_evolution.png'), dpi=150)
    plt.close()

    print(f"Epoch dynamics visualizations saved to {output_dir}")


# ============================================================================
# Recommendations
# ============================================================================

def generate_recommendations(stats, output_dir, focus_task='plan', baseline_results=None):
    lines = []
    lines.append("=" * 70)
    lines.append("GRADIENT CONFLICT ANALYSIS - RECOMMENDATIONS")
    lines.append("=" * 70)
    lines.append("")

    # 0. Statistical validation (if baseline comparison was run)
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
            lines.append("  difference from random baseline. Gradient conflicts may be")
            lines.append("  indistinguishable from high-dimensional noise.")
            lines.append("  → Consider reframing research from 'conflict resolution' to")
            lines.append("    'gradient dynamics optimization' (magnitude balancing, etc.)")
        elif any_bimodal:
            lines.append("  FINDING: Some task pairs show bimodal distribution, indicating")
            lines.append("  real alternation between conflict and cooperation across batches.")
            lines.append("  → Gradient surgery (PCGrad/CAGrad) is justified for these pairs.")
        else:
            lines.append("  FINDING: Statistically significant but unimodal — consistent")
            lines.append("  directional relationship exists. Not random noise.")
        lines.append("")

    # 1. Overall conflict assessment
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

    # 2. Norm imbalance
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
            lines.append(f"  to prevent {focus_task} gradient from being dominated.")
    lines.append("")

    # 3. Interference analysis
    lines.append(f"3. INTERFERENCE ON {focus_task.upper()}")
    lines.append("-" * 40)
    intf = stats.get('interference', {})
    if intf:
        for t in ['det', 'map', 'motion']:
            key = f"{t}_on_{focus_task}"
            if key in intf:
                v = intf[key]['mean']
                severity = "HIGH" if v > 1.0 else "MODERATE" if v > 0.3 else "LOW"
                lines.append(f"  {t:10s} -> {focus_task}: {v:.4f} ({severity})")
    lines.append("")

    # 4. Per-layer hotspots
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

    # 5. Actionable suggestions
    lines.append("5. SUGGESTED ACTIONS")
    lines.append("-" * 40)
    suggestions = []

    if high_conflict:
        suggestions.append("- Apply PCGrad or CAGrad to resolve directional conflicts.")
        # Find if conflicts concentrate in specific layers
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
        suggestions.append(f"- Consider upweighting {focus_task} loss or downweighting dominant task losses.")

    if not suggestions:
        suggestions.append("- No critical issues detected. Current training may be adequate.")
        suggestions.append("- Monitor planning metrics to confirm.")

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

def debug_mode(model, dataloader, device, fp16=False):
    print("\n" + "=" * 60)
    print("DEBUG MODE - Inspecting model and loss structure")
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
    data = move_to_device(data, device)
    img = data.pop('img')

    model.train()
    with torch.cuda.amp.autocast(enabled=fp16):
        outputs = model(img=img, **data)

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
):
    """Worker function for multi-GPU gradient analysis.

    Each worker runs on a separate GPU, processes a subset of batches,
    and saves results to a temporary file.
    """
    device = f"cuda:{gpu_id}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.chdir(PROJECT_ROOT)

    from mmcv import Config
    from mmcv.runner import load_checkpoint

    cfg = Config.fromfile(config_path)
    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = min(4, batch_size)

    print(f"[GPU {gpu_id}] Building model...")
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = model.to(device)
    model = MMDataParallel(model, device_ids=[gpu_id])

    param_groups, group_meta = get_shared_parameters_grouped(model, shared_layers, last_n_layers)

    dataset = custom_build_dataset(cfg.data.train)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=min(4, batch_size),
        collate_fn=partial(collate, samples_per_gpu=batch_size), drop_last=True,
    )

    print(f"[GPU {gpu_id}] Processing {num_batches} batches (offset={batch_offset}, "
          f"batch_size={batch_size}, samples={num_batches * batch_size})...")

    # Skip to batch_offset
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
                model, data, param_groups, group_meta, device,
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

    # Save results (avoid CUDA tensors in pickle — already moved to CPU in analyze_single_batch)
    # But strip stored gradients to avoid serialization issues
    for r in all_results:
        r.pop('task_gradients_all', None)
        r.pop('task_gradients_grouped', None)

    with open(result_path, 'wb') as f:
        pickle.dump({'results': all_results, 'group_meta': dict(group_meta)}, f)


def run_multi_gpu_analysis(args):
    """Dispatch gradient analysis across multiple GPUs using multiprocessing."""
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    num_gpus = args.num_gpus
    total_batches = args.num_batches
    batches_per_gpu = total_batches // num_gpus
    remainder = total_batches % num_gpus

    batch_size = args.batch_size
    store_gradients = (args.analysis_mode == 'direction')

    print(f"\n{'='*60}")
    print(f"MULTI-GPU GRADIENT ANALYSIS")
    print(f"  GPUs: {num_gpus}")
    print(f"  Batch size per GPU: {batch_size}")
    print(f"  Total batches: {total_batches} ({batches_per_gpu} per GPU + {remainder} remainder)")
    print(f"  Total samples: {total_batches * batch_size}")
    print(f"  Analysis mode: {args.analysis_mode}")
    print(f"{'='*60}\n")

    # Create temp files for results
    tmp_dir = tempfile.mkdtemp(prefix='grad_analysis_')
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
        )
        processes.append(p)
        batch_offset += n

    # Launch all workers
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # Merge results
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

    # Cleanup temp dir
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
    parser = argparse.ArgumentParser(description='Analyze gradient conflicts in HiP-AD')
    parser.add_argument('--config', type=str,
                        default='projects/configs/hipad_b2d_stage2.py',
                        help='Config file path')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint file path (required for single-checkpoint mode)')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batches to analyze (default: 100 for statistical power)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size per GPU. If not specified, reads samples_per_gpu from config '
                             '(hipad_b2d_stage2: 6). Use smaller value if OOM.')
    parser.add_argument('--output-dir', type=str, default='gradient_analysis_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (single-GPU mode)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs for parallel analysis. Uses GPU 0..N-1. '
                             'Overrides --device when > 1.')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use FP16 (default: True, matching training config)')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                        help='Disable FP16')
    parser.add_argument('--shared-layers', type=str, nargs='+',
                        default=['gnn', 'temp_gnn', 'inter_gnn', 'ffn', 'norm', 'fc_before', 'fc_after'],
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
                        help='Disable selective eval (keep original train mode behavior)')
    parser.add_argument('--random-baseline', action='store_true',
                        help='Run random baseline comparison with statistical tests and bimodality analysis')
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

    # Validate: need either --checkpoint or --checkpoints
    if not args.checkpoint and not args.checkpoints:
        print("Error: either --checkpoint or --checkpoints is required.")
        sys.exit(1)

    # Resolve batch_size early for multi-checkpoint mode
    if args.batch_size is None:
        _cfg = Config.fromfile(args.config)
        args.batch_size = getattr(_cfg.data, 'samples_per_gpu', 6)
        print(f"Auto batch_size from config: {args.batch_size}")

    # Multi-checkpoint mode: analyze epoch-wise dynamics
    if args.checkpoints:
        ckpt_paths = args.checkpoints
        ckpt_labels = args.checkpoint_labels or [f"ckpt_{i}" for i in range(len(ckpt_paths))]
        if len(ckpt_labels) != len(ckpt_paths):
            print(f"Warning: {len(ckpt_labels)} labels for {len(ckpt_paths)} checkpoints. Padding.")
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

    # Resolve batch_size: use config's samples_per_gpu if not specified
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
        # For visualization/recommendations we still need group_meta
    else:
        # Single-GPU path
        cfg.data.samples_per_gpu = args.batch_size
        cfg.data.workers_per_gpu = min(4, args.batch_size)

        print(f"Loading config from: {args.config}")
        print(f"Loading checkpoint from: {args.checkpoint}")
        print(f"Analysis mode: {args.analysis_mode}")
        print(f"FP16: {args.fp16}")
        print(f"Analyzing {args.num_batches} batches × batch_size {args.batch_size} "
              f"= {args.num_batches * args.batch_size} samples")

        model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.init_weights()
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        print("Checkpoint loaded successfully")

        device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
        model = model.to(args.device)
        model = MMDataParallel(model, device_ids=[device_id])

        # Get grouped shared parameters
        param_groups, group_meta = get_shared_parameters_grouped(
            model, args.shared_layers, args.last_n_layers
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

        dataset = custom_build_dataset(cfg.data.train)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=min(4, args.batch_size),
            collate_fn=partial(collate, samples_per_gpu=args.batch_size), drop_last=True,
        )

        print(f"\nDataset size: {len(dataset)}")

        if args.debug:
            debug_mode(model, dataloader, args.device, args.fp16)
            return

        print(f"\nStarting gradient conflict analysis...\n")

        all_results = []
        for batch_idx, data in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break

            print(f"Processing batch {batch_idx + 1}/{args.num_batches}...", end='\r')

            try:
                result = analyze_single_batch(
                    model, data, param_groups, group_meta, args.device,
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

    print_summary(stats, analysis_mode=args.analysis_mode, focus_task=args.focus_task)
    save_results(stats, args.output_dir, analysis_mode=args.analysis_mode)

    if not args.no_plots:
        try:
            create_basic_visualizations(stats, args.output_dir)
        except Exception as e:
            print(f"Warning: Failed to create basic visualizations: {e}")

        if args.analysis_mode in ('full', 'direction'):
            try:
                create_per_layer_visualizations(stats, group_meta, args.output_dir)
            except Exception as e:
                print(f"Warning: Failed to create per-layer visualizations: {e}")

            try:
                create_advanced_visualizations(stats, group_meta, args.output_dir, focus_task=args.focus_task)
            except Exception as e:
                print(f"Warning: Failed to create advanced visualizations: {e}")

        if args.analysis_mode == 'direction':
            try:
                create_direction_visualizations(stats, args.output_dir, group_meta=group_meta)
            except Exception as e:
                print(f"Warning: Failed to create direction visualizations: {e}")

    # Random baseline comparison & bimodality analysis
    baseline_results = None
    if args.random_baseline:
        try:
            baseline_results = run_random_baseline_analysis(stats, args.output_dir)
            if not args.no_plots:
                create_baseline_visualizations(stats, baseline_results, args.output_dir)
        except Exception as e:
            print(f"Warning: Failed to run random baseline analysis: {e}")
            import traceback
            traceback.print_exc()

    # Generate recommendations
    if args.analysis_mode in ('full', 'direction'):
        try:
            rec = generate_recommendations(
                stats, args.output_dir, focus_task=args.focus_task,
                baseline_results=baseline_results,
            )
            print("\n" + rec)
        except Exception as e:
            print(f"Warning: Failed to generate recommendations: {e}")

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
