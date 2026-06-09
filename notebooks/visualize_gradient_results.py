#!/usr/bin/env python
"""
Visualize pre-computed gradient conflict analysis results.

Usage:
    python notebooks/visualize_gradient_results.py \
        --results-dir notebooks/gradient_analysis_results \
        --epochs ep1 ep2 ep3

Reads saved JSON/CSV from each epoch directory and generates all plots
without re-running the expensive gradient computation.
"""

import os
import re
import sys
import json
import math
import argparse
from collections import defaultdict

import numpy as np

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


# ---------------------------------------------------------------------------
# Matplotlib-only heatmap (no seaborn dependency)
# ---------------------------------------------------------------------------

def _heatmap(ax, data, xticklabels, yticklabels, cmap='RdBu',
             vmin=None, vmax=None, center=None, annot=True, fmt='.2f',
             colorbar=True, colorbar_label=''):
    """Draw an annotated heatmap using only matplotlib."""
    if center is not None and vmin is not None and vmax is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    elif center is not None:
        abs_max = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 0.01)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=center, vmax=abs_max)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(data, cmap=cmap, aspect='auto', norm=norm)

    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(yticklabels, fontsize=7)

    if annot:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                # Choose text color based on background brightness
                rgba = im.cmap(im.norm(val))
                brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = 'white' if brightness < 0.5 else 'black'
                text = f'{val:{fmt}}' if fmt != '.1%' else f'{val * 100:.1f}%'
                ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)

    # Grid lines
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    if colorbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        ax.figure.colorbar(im, cax=cax, label=colorbar_label)

    return im


# ---------------------------------------------------------------------------
# Constants (nuScenes version — no det_kd)
# ---------------------------------------------------------------------------

TASK_PAIRS = [
    ('plan', 'det'),
    ('plan', 'map'),
    ('plan', 'motion'),
    ('plan', 'ego'),
    ('ego', 'det'),
    ('ego', 'map'),
    ('ego', 'motion'),
    ('det', 'map'),
    ('det', 'motion'),
    ('map', 'motion'),
]

TASK_COLORS = {
    'det': '#e74c3c',
    'map': '#2ecc71',
    'motion': '#3498db',
    'ego': '#9b59b6',
    'plan': '#f39c12',
}


# ---------------------------------------------------------------------------
# Reconstruct group_meta from group key names
# ---------------------------------------------------------------------------

def reconstruct_group_meta(group_keys):
    """Infer decoder_idx, op_type, global_layer_idx from group key naming."""
    meta = {}
    global_idx = 0
    for gk in sorted(group_keys, key=_group_sort_key):
        m = re.match(r'dec(\d+)_(.+)', gk)
        if m:
            dec_idx = int(m.group(1))
            op_type = m.group(2)
        elif gk.startswith('backbone_'):
            dec_idx = -1
            op_type = gk
        elif gk == 'neck':
            dec_idx = -1
            op_type = 'neck'
        elif gk in ('fc_before', 'fc_after'):
            dec_idx = -1
            op_type = gk
        else:
            dec_idx = -1
            op_type = gk
        meta[gk] = {
            'decoder_idx': dec_idx,
            'op_type': op_type,
            'global_layer_idx': global_idx,
        }
        global_idx += 1
    return meta


def _group_sort_key(gk):
    """Sort: backbone < neck < dec0..dec5 < fc"""
    if gk.startswith('backbone_stem'):
        return (0, 0, gk)
    if gk.startswith('backbone_layer'):
        m = re.search(r'(\d+)', gk)
        return (0, int(m.group(1)) if m else 99, gk)
    if gk == 'neck':
        return (1, 0, gk)
    m = re.match(r'dec(\d+)_(.*)', gk)
    if m:
        dec = int(m.group(1))
        sub = m.group(2)
        # order within decoder layer
        sub_order = {'norm_0': 0, 'ffn_0_prenorm': 1, 'ffn_0_w1': 2, 'ffn_0_w2': 3,
                     'ffn_0_other': 4, 'norm_1': 5}
        return (2, dec, sub_order.get(sub, 9))
    if gk == 'fc_before':
        return (3, 0, gk)
    if gk == 'fc_after':
        return (3, 1, gk)
    return (9, 0, gk)


# ---------------------------------------------------------------------------
# Load saved results
# ---------------------------------------------------------------------------

def load_stats(epoch_dir):
    """Load gradient_conflict_stats.json."""
    path = os.path.join(epoch_dir, 'gradient_conflict_stats.json')
    with open(path) as f:
        return json.load(f)


def load_baseline(epoch_dir):
    """Load baseline comparison results."""
    path = os.path.join(epoch_dir, 'baseline_comparison', 'statistical_tests.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Random baseline helper (cheap, no GPU needed)
# ---------------------------------------------------------------------------

def generate_random_baseline(dim, num_pairs=10000, seed=42):
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


# ---------------------------------------------------------------------------
# Visualization functions
# (Adapted from analyze_gradient_conflict.py — work with saved JSON data
#  which has no 'values' field, only mean/std/min/max/median)
# ---------------------------------------------------------------------------

def create_basic_visualizations(stats, output_dir):
    pass  # matplotlib already imported at module level

    os.makedirs(output_dir, exist_ok=True)
    task_names = ['plan', 'det', 'map', 'motion', 'ego']

    # 1. Cosine similarity heatmap
    n = len(task_names)
    mat = np.ones((n, n))
    for i, t1 in enumerate(task_names):
        for j, t2 in enumerate(task_names):
            if i != j:
                pk = f"{t1}_vs_{t2}" if f"{t1}_vs_{t2}" in stats['pairwise_cosine'] else f"{t2}_vs_{t1}"
                if pk in stats['pairwise_cosine']:
                    mat[i, j] = stats['pairwise_cosine'][pk]['mean']
    fig, ax = plt.subplots(figsize=(8, 6))
    _heatmap(ax, mat, task_names, task_names, cmap='RdBu', center=0, vmin=-0.3, vmax=0.3, fmt='.3f')
    ax.set_title('Mean Cosine Similarity between Task Gradients\n(Negative = Conflict)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_heatmap.png'), dpi=600)
    plt.close()

    # 2. Conflict frequency bar chart
    pairs = list(stats.get('conflict_frequency', {}).keys())
    # Filter out pairs with zero data (e.g., ego-related)
    pairs = [p for p in pairs if stats['conflict_frequency'][p].get('total', 0) > 0]
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
        plt.savefig(os.path.join(output_dir, 'conflict_frequency.png'), dpi=600)
        plt.close()

    # 3. Gradient norms bar chart
    tasks = list(stats.get('gradient_norms', {}).keys())
    if tasks:
        means = [stats['gradient_norms'][t]['mean'] for t in tasks]
        stds = [stats['gradient_norms'][t]['std'] for t in tasks]
        task_colors = [TASK_COLORS.get(t, 'gray') for t in tasks]
        plt.figure(figsize=(8, 5))
        plt.bar(tasks, means, yerr=stds, capsize=5, color=task_colors, alpha=0.8)
        plt.xlabel('Task')
        plt.ylabel('Gradient Norm')
        plt.title('Mean Gradient Norm by Task')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_norms.png'), dpi=600)
        plt.close()

    print(f"  Basic visualizations saved to {output_dir}")


def create_per_layer_visualizations(stats, group_meta, output_dir):
    pass  # matplotlib already imported at module level

    layer_dir = os.path.join(output_dir, 'per_layer')
    os.makedirs(layer_dir, exist_ok=True)

    pg_cos = stats.get('per_group_cosine', {})
    pg_norms = stats.get('per_group_norms', {})
    pg_cf = stats.get('per_group_conflict_frequency', {})
    if not pg_cos:
        print("  No per_group_cosine data, skipping per-layer visualizations.")
        return

    sorted_gks = sorted(pg_cos.keys(), key=_group_sort_key)
    # Only use pairs that exist in the data
    all_pair_keys = [f"{t1}_vs_{t2}" for t1, t2 in TASK_PAIRS]
    pair_keys = [pk for pk in all_pair_keys if any(pk in pg_cos.get(gk, {}) for gk in sorted_gks)]

    # --- Per-Operation Conflict Heatmap ---
    mat = np.full((len(sorted_gks), len(pair_keys)), np.nan)
    for i, gk in enumerate(sorted_gks):
        for j, pk in enumerate(pair_keys):
            cs = pg_cos.get(gk, {}).get(pk)
            if cs:
                mat[i, j] = cs['mean']

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_gks) * 0.35)))
    _heatmap(ax, mat, pair_keys, sorted_gks, cmap='RdBu', center=0, vmin=-0.3, vmax=0.3, fmt='.2f')
    ax.set_title('Per-Operation Cosine Similarity (Negative = Conflict)')
    plt.tight_layout()
    plt.savefig(os.path.join(layer_dir, 'per_operation_conflict_heatmap.png'), dpi=600)
    plt.close()

    # --- Per-Decoder-Layer Conflict (aggregated) ---
    dec_indices = sorted(set(
        group_meta[gk]['decoder_idx'] for gk in sorted_gks
        if group_meta.get(gk, {}).get('decoder_idx', -1) >= 0
    ))
    has_fc = any(gk in ('fc_before', 'fc_after') for gk in sorted_gks)
    has_backbone = any(gk.startswith('backbone_') for gk in sorted_gks)
    has_neck = 'neck' in sorted_gks

    dec_labels = []
    if has_backbone:
        dec_labels.append('backbone')
    if has_neck:
        dec_labels.append('neck')
    dec_labels.extend([f"dec{d}" for d in dec_indices])
    if has_fc:
        dec_labels.append("fc_shared")

    mat_dec = np.full((len(dec_labels), len(pair_keys)), np.nan)
    mat_cf = np.full((len(dec_labels), len(pair_keys)), np.nan)

    def _fill_agg(label_idx, gks_for_agg):
        for pj, pk in enumerate(pair_keys):
            vals, cf_vals = [], []
            for gk in gks_for_agg:
                cs = pg_cos.get(gk, {}).get(pk)
                if cs:
                    vals.append(cs['mean'])
                cf = pg_cf.get(gk, {}).get(pk)
                if cf:
                    cf_vals.append(cf['ratio'])
            if vals:
                mat_dec[label_idx, pj] = float(np.mean(vals))
            if cf_vals:
                mat_cf[label_idx, pj] = float(np.mean(cf_vals))

    label_offset = 0
    if has_backbone:
        bb_gks = [gk for gk in sorted_gks if gk.startswith('backbone_')]
        _fill_agg(label_offset, bb_gks)
        label_offset += 1
    if has_neck:
        _fill_agg(label_offset, ['neck'])
        label_offset += 1
    for di, dec_idx in enumerate(dec_indices):
        dec_gks = [gk for gk in sorted_gks if group_meta.get(gk, {}).get('decoder_idx') == dec_idx]
        _fill_agg(label_offset + di, dec_gks)
    if has_fc:
        fc_gks = [gk for gk in sorted_gks if gk in ('fc_before', 'fc_after')]
        _fill_agg(len(dec_labels) - 1, fc_gks)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(dec_labels) * 0.5)))
    _heatmap(axes[0], mat_dec, pair_keys, dec_labels, cmap='RdBu', center=0, vmin=-0.3, vmax=0.3, fmt='.2f')
    axes[0].set_title('Per-Decoder-Layer Mean Cosine')

    _heatmap(axes[1], mat_cf, pair_keys, dec_labels, cmap='Reds', vmin=0, vmax=1, fmt='.1%')
    axes[1].set_title('Per-Decoder-Layer Conflict Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(layer_dir, 'per_decoder_layer_conflict.png'), dpi=600)
    plt.close()

    # --- Operation Type Conflict Profile ---
    op_types = sorted(set(group_meta[gk]['op_type'] for gk in sorted_gks))
    mat_op = np.full((len(op_types), len(pair_keys)), np.nan)
    for oi, op in enumerate(op_types):
        op_gks = [gk for gk in sorted_gks if group_meta.get(gk, {}).get('op_type') == op]
        for pj, pk in enumerate(pair_keys):
            vals = []
            for gk in op_gks:
                cs = pg_cos.get(gk, {}).get(pk)
                if cs:
                    vals.append(cs['mean'])
            if vals:
                mat_op[oi, pj] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(10, max(4, len(op_types) * 0.6)))
    _heatmap(ax, mat_op, pair_keys, op_types, cmap='RdBu', center=0, vmin=-0.3, vmax=0.3, fmt='.3f')
    ax.set_title('Operation Type Conflict Profile (Aggregated across decoder layers)')
    plt.tight_layout()
    plt.savefig(os.path.join(layer_dir, 'operation_type_conflict.png'), dpi=600)
    plt.close()

    # --- Per-Layer Gradient Norm Dominance ---
    if pg_norms:
        task_names = ['det', 'map', 'motion', 'ego', 'plan']
        available_tasks = set()
        for gk in sorted_gks:
            available_tasks.update(pg_norms.get(gk, {}).keys())
        task_names = [t for t in task_names if t in available_tasks]

        x = np.arange(len(sorted_gks))
        width = 0.8 / max(len(task_names), 1)

        fig, ax = plt.subplots(figsize=(max(12, len(sorted_gks) * 0.6), 6))
        for ti, task in enumerate(task_names):
            means = []
            for gk in sorted_gks:
                ns = pg_norms.get(gk, {}).get(task)
                means.append(ns['mean'] if ns else 0)
            ax.bar(x + ti * width, means, width, label=task,
                   color=TASK_COLORS.get(task, 'gray'), alpha=0.8)

        ax.set_xticks(x + width * len(task_names) / 2)
        ax.set_xticklabels(sorted_gks, rotation=90, ha='center', fontsize=7)
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Per-Layer Gradient Norm Dominance')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(layer_dir, 'gradient_norm_dominance.png'), dpi=600)
        plt.close()

    # --- Gradient Norm Ratio Heatmap ---
    if pg_norms:
        mat_ratio = np.full((len(sorted_gks), len(pair_keys)), np.nan)
        for i, gk in enumerate(sorted_gks):
            for j, pk in enumerate(pair_keys):
                t1, t2 = pk.split('_vs_')
                n1 = pg_norms.get(gk, {}).get(t1)
                n2 = pg_norms.get(gk, {}).get(t2)
                if n1 and n2 and n1['mean'] > 1e-10 and n2['mean'] > 1e-10:
                    mat_ratio[i, j] = np.log2(n1['mean'] / n2['mean'])

        fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_gks) * 0.35)))
        _heatmap(ax, mat_ratio, pair_keys, sorted_gks, cmap='coolwarm', center=0, fmt='.1f')
        ax.set_title('Gradient Norm Ratio log2(task1/task2) per Layer\n(Positive = task1 dominates)')
        plt.tight_layout()
        plt.savefig(os.path.join(layer_dir, 'gradient_norm_ratio.png'), dpi=600)
        plt.close()

    print(f"  Per-layer visualizations saved to {layer_dir}")


def create_advanced_visualizations(stats, group_meta, output_dir, focus_task='plan'):
    pass  # matplotlib already imported at module level

    adv_dir = os.path.join(output_dir, 'advanced')
    os.makedirs(adv_dir, exist_ok=True)

    task_names = ['det', 'map', 'motion', 'ego', 'plan']

    # --- Interference Magnitude (asymmetric heatmap) ---
    intf = stats.get('interference', {})
    if intf:
        mat = np.zeros((len(task_names), len(task_names)))
        for i, src in enumerate(task_names):
            for j, vic in enumerate(task_names):
                if i != j:
                    key = f"{src}_on_{vic}"
                    if key in intf:
                        mat[i, j] = intf[key]['mean']

        fig, ax = plt.subplots(figsize=(8, 6))
        _heatmap(ax, mat, [f"{t}\n(victim)" for t in task_names],
                 [f"{t}\n(source)" for t in task_names],
                 cmap='Reds', vmin=0, fmt='.3f')
        plt.title('Gradient Interference Magnitude\n||g_src|| * max(0, -cos(g_src, g_vic))')
        plt.tight_layout()
        plt.savefig(os.path.join(adv_dir, 'interference_magnitude.png'), dpi=600)
        plt.close()

    # --- Cooperative vs Conflicting Decomposition per decoder layer ---
    pg_decomp = stats.get('per_group_decomposition', {})
    pg_cos = stats.get('per_group_cosine', {})
    if pg_decomp and pg_cos:
        sorted_gks = sorted(pg_cos.keys(), key=_group_sort_key)
        dec_indices = sorted(set(
            group_meta[gk]['decoder_idx'] for gk in sorted_gks
            if group_meta.get(gk, {}).get('decoder_idx', -1) >= 0
        ))

        aux_tasks = [t for t in task_names if t != focus_task]
        fig, axes = plt.subplots(1, len(aux_tasks), figsize=(6 * len(aux_tasks), 5))
        if len(aux_tasks) == 1:
            axes = [axes]

        for ai, aux in enumerate(aux_tasks):
            dk = f"{aux}_wrt_{focus_task}"
            coop_vals, conf_vals, dec_labels = [], [], []
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

            if dec_labels:
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
        plt.savefig(os.path.join(adv_dir, 'cooperative_vs_conflicting.png'), dpi=600)
        plt.close()

    # --- Cumulative Gradient Flow ---
    if pg_cos:
        sorted_gks = sorted(pg_cos.keys(), key=_group_sort_key)
        dec_indices = sorted(set(
            group_meta[gk]['decoder_idx'] for gk in sorted_gks
            if group_meta.get(gk, {}).get('decoder_idx', -1) >= 0
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
                        batch_vals.append(cs['mean'])
                vals.append(float(np.mean(batch_vals)) if batch_vals else np.nan)

            if any(not np.isnan(v) for v in vals):
                ax.plot(dec_indices, vals, 'o-', label=pk, linewidth=2, markersize=6)

        ax.axhline(0, color='r', linestyle='--', linewidth=0.8, label='Conflict threshold')
        ax.set_xlabel('Decoder Layer')
        ax.set_ylabel('Mean Cosine Similarity')
        ax.set_title('Gradient Conflict Across Decoder Depth')
        ax.legend(fontsize=8, ncol=2)
        ax.set_xticks(dec_indices)
        ax.set_xticklabels([f"dec{d}" for d in dec_indices])
        plt.tight_layout()
        plt.savefig(os.path.join(adv_dir, 'cumulative_gradient_flow.png'), dpi=600)
        plt.close()

    # --- Planning-Centric Dashboard ---
    if pg_cos and intf:
        aux_tasks = [t for t in task_names if t != focus_task]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        dec_indices_valid = sorted(set(
            group_meta[gk]['decoder_idx'] for gk in pg_cos.keys()
            if group_meta.get(gk, {}).get('decoder_idx', -1) >= 0
        ))

        # Subplot 1: Cosine per decoder layer (plan vs aux)
        ax = axes[0, 0]
        first_gk_pairs = list(list(pg_cos.values())[0].keys()) if pg_cos else []
        for aux in aux_tasks:
            pk = f"{focus_task}_vs_{aux}" if f"{focus_task}_vs_{aux}" in first_gk_pairs else f"{aux}_vs_{focus_task}"
            vals = []
            for d in dec_indices_valid:
                dec_gks = [gk for gk in pg_cos if group_meta.get(gk, {}).get('decoder_idx') == d]
                bv = [pg_cos.get(gk, {}).get(pk, {}).get('mean') for gk in dec_gks
                      if pg_cos.get(gk, {}).get(pk)]
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
                bv = [pg_intf.get(gk, {}).get(ik, {}).get('mean') for gk in dec_gks
                      if pg_intf.get(gk, {}).get(ik)]
                vals.append(float(np.mean(bv)) if bv else 0)
            ax.plot(dec_indices_valid, vals, 's-', label=f"{aux} -> {focus_task}",
                    color=TASK_COLORS.get(aux, 'gray'), linewidth=2)
        ax.set_title(f'Interference Magnitude on {focus_task}')
        ax.set_xlabel('Decoder Layer')
        ax.set_ylabel('Interference')
        ax.legend()

        # Subplot 3: Helpfulness score
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
            lines.append(f"  cosine = {cs.get('mean', 'N/A'):+.4f}" if isinstance(cs.get('mean'), (int, float)) else f"  cosine = N/A")
            lines.append(f"  conflict rate = {cf.get('ratio', 0) * 100:.1f}%")
            lines.append(f"  interference = {iv.get('mean', 0):.4f}" if isinstance(iv.get('mean'), (int, float)) else f"  interference = N/A")
            lines.append("")
        ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')

        plt.suptitle(f'Planning-Centric Gradient Analysis Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(adv_dir, 'planning_centric_dashboard.png'), dpi=600)
        plt.close()

    print(f"  Advanced visualizations saved to {adv_dir}")


def create_active_overlap_visualizations(stats, group_meta, output_dir):
    pass  # matplotlib already imported at module level

    ao_dir = os.path.join(output_dir, 'active_overlap')
    os.makedirs(ao_dir, exist_ok=True)

    ao_stats = stats.get('active_overlap', {})
    cos_stats = stats.get('pairwise_cosine', {})
    if ao_stats and cos_stats:
        pairs = sorted(ao_stats.keys())
        raw_means = [cos_stats.get(pk, {}).get('mean', 0) for pk in pairs]
        overlap_means = [ao_stats[pk].get('overlap_cosine', {}).get('mean', 0) for pk in pairs]
        overlap_ratios = [ao_stats[pk].get('overlap_ratio', {}).get('mean', 0) for pk in pairs]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        x = np.arange(len(pairs))
        width = 0.35
        axes[0].bar(x - width/2, raw_means, width, label='Raw Cosine', color='steelblue', alpha=0.7)
        axes[0].bar(x + width/2, overlap_means, width, label='Overlap Cosine', color='coral', alpha=0.7)
        axes[0].axhline(0, color='k', linewidth=0.5)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(pairs, rotation=45, ha='right', fontsize=7)
        axes[0].set_ylabel('Cosine Similarity')
        axes[0].set_title('Raw vs Active-Overlap Cosine')
        axes[0].legend(fontsize=8)

        colors = ['#e74c3c' if r < 0.1 else '#f39c12' if r < 0.3 else '#2ecc71' for r in overlap_ratios]
        axes[1].bar(x, overlap_ratios, color=colors, alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(pairs, rotation=45, ha='right', fontsize=7)
        axes[1].set_ylabel('Overlap Ratio')
        axes[1].set_title('Active Coordinate Overlap Ratio\n(Low = disjoint gradient support)')
        axes[1].set_ylim(0, 1)

        interp_types = ['cooperative', 'true_orthogonal', 'disjoint', 'hidden_conflict', 'conflict']
        interp_colors = {'cooperative': '#2ecc71', 'true_orthogonal': '#3498db',
                         'disjoint': '#f39c12', 'hidden_conflict': '#e74c3c', 'conflict': '#c0392b'}
        bottom = np.zeros(len(pairs))
        for it in interp_types:
            vals = []
            for pk in pairs:
                counts = ao_stats[pk].get('interpretation_counts', {})
                total = max(sum(counts.values()), 1)
                vals.append(counts.get(it, 0) / total)
            axes[2].bar(x, vals, bottom=bottom, label=it,
                        color=interp_colors.get(it, 'gray'), alpha=0.8)
            bottom += np.array(vals)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(pairs, rotation=45, ha='right', fontsize=7)
        axes[2].set_ylabel('Fraction')
        axes[2].set_title('Interpretation Distribution')
        axes[2].legend(fontsize=7, loc='upper right')

        plt.suptitle('Active-Overlap Gradient Analysis', fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(os.path.join(ao_dir, 'active_overlap_summary.png'), dpi=600)
        plt.close()

    # --- Per-Group Active-Overlap Heatmap ---
    pg_ao = stats.get('per_group_active_overlap', {})
    pg_cos = stats.get('per_group_cosine', {})
    if pg_ao and pg_cos:
        sorted_gks = sorted(pg_cos.keys(), key=_group_sort_key)
        all_pair_keys = [f"{t1}_vs_{t2}" for t1, t2 in TASK_PAIRS]
        pair_keys = [pk for pk in all_pair_keys if any(pk in pg_cos.get(gk, {}) for gk in sorted_gks)]

        mat_raw = np.full((len(sorted_gks), len(pair_keys)), np.nan)
        mat_overlap = np.full((len(sorted_gks), len(pair_keys)), np.nan)
        mat_ratio = np.full((len(sorted_gks), len(pair_keys)), np.nan)

        for i, gk in enumerate(sorted_gks):
            for j, pk in enumerate(pair_keys):
                cs = pg_cos.get(gk, {}).get(pk)
                if cs:
                    mat_raw[i, j] = cs['mean']
                ao = pg_ao.get(gk, {}).get(pk, {})
                ao_cos = ao.get('overlap_cosine', {})
                ao_ratio_d = ao.get('overlap_ratio', {})
                if ao_cos and 'mean' in ao_cos:
                    mat_overlap[i, j] = ao_cos['mean']
                if ao_ratio_d and 'mean' in ao_ratio_d:
                    mat_ratio[i, j] = ao_ratio_d['mean']

        fig, axes = plt.subplots(1, 3, figsize=(24, max(6, len(sorted_gks) * 0.35)))
        _heatmap(axes[0], mat_raw, pair_keys, sorted_gks, cmap='RdBu', center=0, vmin=-0.3, vmax=0.3, fmt='.2f')
        axes[0].set_title('Raw Cosine Similarity')

        _heatmap(axes[1], mat_overlap, pair_keys, sorted_gks, cmap='RdBu', center=0, vmin=-0.3, vmax=0.3, fmt='.2f')
        axes[1].set_title('Active-Overlap Cosine')

        _heatmap(axes[2], mat_ratio, pair_keys, sorted_gks, cmap='YlOrRd_r', vmin=0, vmax=1, fmt='.2f')
        axes[2].set_title('Overlap Ratio (low = disjoint)')

        plt.suptitle('Per-Operation: Raw vs Overlap Cosine & Overlap Ratio', fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(os.path.join(ao_dir, 'per_operation_active_overlap.png'), dpi=600)
        plt.close()

    print(f"  Active-overlap visualizations saved to {ao_dir}")


def create_baseline_visualizations(stats, baseline_results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    baseline_dir = os.path.join(output_dir, 'baseline_comparison')
    os.makedirs(baseline_dir, exist_ok=True)

    cos_data = stats.get('pairwise_cosine', {})
    pairs = sorted(cos_data.keys())
    if not pairs:
        return

    # Note: 'values' was stripped from JSON. We generate a synthetic normal
    # distribution from mean/std for approximate KDE plots.
    dim = 56000  # fallback
    st_tests = baseline_results.get('statistical_tests', {})
    for pk in pairs:
        st = st_tests.get(pk, {})
        rb = st.get('random_baseline', {})
        if 'theoretical_std' in rb:
            # dim ~ 1/std^2 for random vectors
            dim = int(round(1.0 / rb['theoretical_std'] ** 2))
            break

    random_cosines = generate_random_baseline(dim, num_pairs=10000)

    # --- Summary comparison bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    means = [cos_data[pk]['mean'] for pk in pairs]
    stds = [cos_data[pk]['std'] for pk in pairs]
    x = np.arange(len(pairs))
    axes[0].bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7, label='Observed')
    axes[0].axhline(np.mean(random_cosines), color='red', linestyle='--',
                    label=f'Random mean ({np.mean(random_cosines):.4f})')
    axes[0].axhspan(np.mean(random_cosines) - np.std(random_cosines),
                     np.mean(random_cosines) + np.std(random_cosines),
                     alpha=0.1, color='red', label='Random ±1σ')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Mean Cosine Similarity')
    axes[0].set_title('Observed vs Random Baseline')
    axes[0].legend(fontsize=8)

    # p-values
    p_vals = []
    for pk in pairs:
        st = st_tests.get(pk, {})
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

    # Bimodality scores
    bm_data = baseline_results.get('bimodality', {})
    bm_scores = [bm_data.get(pk, {}).get('bimodality_score', 0) for pk in pairs]
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
    plt.savefig(os.path.join(baseline_dir, 'statistical_summary.png'), dpi=600)
    plt.close()

    print(f"  Baseline visualizations saved to {baseline_dir}")


def create_epoch_dynamics_visualizations(checkpoint_stats, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    if len(checkpoint_stats) < 2:
        print("  Need at least 2 checkpoints for dynamics visualization.")
        return

    labels = [cs.get('label', cs.get('checkpoint_label', f'ckpt{i}'))
              for i, cs in enumerate(checkpoint_stats)]
    x = np.arange(len(labels))

    # --- Figure 1: 2x2 Active-Overlap + Raw Cosine ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    pair_keys = sorted(checkpoint_stats[0].get('pairwise_cosine', {}).keys())

    # Panel 1: Active-overlap cosine evolution
    ax = axes[0, 0]
    for pk in pair_keys:
        means = []
        for cs in checkpoint_stats:
            ao = cs.get('active_overlap', {}).get(pk, {})
            oc = ao.get('overlap_cosine', {})
            means.append(oc.get('mean', np.nan) if isinstance(oc, dict) else np.nan)
        ax.plot(x, means, 'o-', label=pk, linewidth=2, markersize=6)
    ax.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Active-Overlap Cosine')
    ax.set_title('Active-Overlap Cosine Evolution')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: Overlap ratio evolution
    ax = axes[0, 1]
    for pk in pair_keys:
        ratios = []
        for cs in checkpoint_stats:
            ao = cs.get('active_overlap', {}).get(pk, {})
            oratio = ao.get('overlap_ratio', {})
            ratios.append(oratio.get('mean', np.nan) if isinstance(oratio, dict) else np.nan)
        ax.plot(x, ratios, 's-', label=pk, linewidth=2, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Overlap Ratio')
    ax.set_title('Gradient Overlap Ratio Evolution')
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Panel 3: Raw cosine
    ax = axes[1, 0]
    for pk in pair_keys:
        means = []
        for cs in checkpoint_stats:
            v = cs.get('pairwise_cosine', {}).get(pk, {})
            means.append(v.get('mean', np.nan) if isinstance(v, dict) else np.nan)
        ax.plot(x, means, 'o-', label=pk, linewidth=2, markersize=6)
    ax.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Raw Cosine Similarity')
    ax.set_title('Raw Full-Vector Cosine (for comparison)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 4: Conflict frequency evolution
    ax = axes[1, 1]
    for pk in pair_keys:
        freqs = []
        for cs in checkpoint_stats:
            cf = cs.get('conflict_frequency', {}).get(pk, {})
            freqs.append(cf.get('ratio', np.nan))
        ax.plot(x, freqs, 'o-', label=pk, linewidth=2, markersize=6)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='50% threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Conflict Frequency')
    ax.set_title('Conflict Frequency Evolution')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Gradient Dynamics Across Training Checkpoints', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'cosine_evolution.png'), dpi=600)
    plt.close()

    # --- Figure 2: Gradient norm evolution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    task_names = sorted(checkpoint_stats[0].get('gradient_norms', {}).keys())
    for task in task_names:
        means, stds = [], []
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
    ax.set_title('Task Gradient Magnitude Evolution (Shaded = ±1σ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_norm_evolution.png'), dpi=600)
    plt.close()

    # --- Figure 3: Magnitude ratio evolution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'plan' in task_names:
        for task in task_names:
            if task == 'plan':
                continue
            ratios = []
            for cs in checkpoint_stats:
                n_task = cs.get('gradient_norms', {}).get(task, {}).get('mean', 1)
                n_plan = cs.get('gradient_norms', {}).get('plan', {}).get('mean', 1)
                ratios.append(n_task / max(n_plan, 1e-10))
            color = TASK_COLORS.get(task, 'gray')
            ax.plot(x, ratios, 'o-', label=f'{task}/plan', color=color, linewidth=2, markersize=6)

        ax.axhline(1, color='gray', linestyle=':', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Norm Ratio (task / plan)')
        ax.set_title('Gradient Magnitude Imbalance vs Plan Task')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'magnitude_ratio_evolution.png'), dpi=600)
    plt.close()

    print(f"  Epoch dynamics visualizations saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Visualize gradient conflict analysis results')
    parser.add_argument('--results-dir', default='notebooks/gradient_analysis_results',
                        help='Root directory containing epoch subdirectories')
    parser.add_argument('--epochs', nargs='+', default=None,
                        help='Epoch subdirectory names (e.g., ep1 ep2 ep3). '
                             'If not specified, auto-detects all ep* directories.')
    parser.add_argument('--focus-task', default='plan',
                        help='Focus task for planning-centric analysis (default: plan)')
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        # Try relative to script location first, then cwd
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        candidate = os.path.join(project_root, results_dir)
        if os.path.exists(candidate):
            results_dir = candidate

    # Auto-detect epochs
    if args.epochs is None:
        epochs = sorted([d for d in os.listdir(results_dir)
                        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('ep')])
    else:
        epochs = args.epochs

    if not epochs:
        print(f"No epoch directories found in {results_dir}")
        return

    print(f"Found epochs: {epochs}")
    print(f"Results directory: {results_dir}")
    print()

    # --- Per-epoch visualizations ---
    for ep in epochs:
        ep_dir = os.path.join(results_dir, ep)
        if not os.path.exists(os.path.join(ep_dir, 'gradient_conflict_stats.json')):
            print(f"Skipping {ep}: no gradient_conflict_stats.json found")
            continue

        print(f"{'='*60}")
        print(f"Generating visualizations for {ep}")
        print(f"{'='*60}")

        stats = load_stats(ep_dir)
        group_keys = list(stats.get('per_group_cosine', {}).keys())
        group_meta = reconstruct_group_meta(group_keys)

        # Basic
        try:
            create_basic_visualizations(stats, ep_dir)
        except Exception as e:
            print(f"  Warning: basic viz failed: {e}")

        # Per-layer
        try:
            create_per_layer_visualizations(stats, group_meta, ep_dir)
        except Exception as e:
            print(f"  Warning: per-layer viz failed: {e}")

        # Advanced
        try:
            create_advanced_visualizations(stats, group_meta, ep_dir, focus_task=args.focus_task)
        except Exception as e:
            print(f"  Warning: advanced viz failed: {e}")

        # Active-overlap
        try:
            create_active_overlap_visualizations(stats, group_meta, ep_dir)
        except Exception as e:
            print(f"  Warning: active-overlap viz failed: {e}")

        # Baseline
        baseline = load_baseline(ep_dir)
        if baseline:
            try:
                create_baseline_visualizations(stats, baseline, ep_dir)
            except Exception as e:
                print(f"  Warning: baseline viz failed: {e}")

        print()

    # --- Cross-epoch dynamics ---
    if len(epochs) >= 2:
        print(f"{'='*60}")
        print(f"Generating epoch dynamics visualizations")
        print(f"{'='*60}")

        dynamics_path = os.path.join(results_dir, 'epoch_dynamics', 'epoch_dynamics.json')
        if os.path.exists(dynamics_path):
            with open(dynamics_path) as f:
                checkpoint_stats = json.load(f)
            dynamics_dir = os.path.join(results_dir, 'epoch_dynamics')
            try:
                create_epoch_dynamics_visualizations(checkpoint_stats, dynamics_dir)
            except Exception as e:
                print(f"  Warning: epoch dynamics viz failed: {e}")
        else:
            print(f"  No epoch_dynamics.json found, skipping cross-epoch plots")

    print(f"\nDone! Check {results_dir} for generated PNG files.")


if __name__ == '__main__':
    main()
