#!/usr/bin/env python3
"""De-aggregated gradient-conflict visualization.

Notebook phase2_v3 averages conflict over layers / epochs. This script does the
opposite: keeps every (epoch, task_pair, layer) cell separate so the raw signal
is visible instead of being washed out by means.

Source: gradient_analysis_results/v3/ckpt_*/layer_conflict/conflict_*_summary.csv
Output: gradient_analysis_results/v3/granular/
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path('/home/yongjae/e2e/HiP-AD-pcgrad/gradient_analysis_results/v3')
OUT = ROOT / 'granular'
OUT.mkdir(exist_ok=True)

PAIRS = [('det', 'map'), ('det', 'motion'), ('det', 'plan'),
         ('map', 'motion'), ('map', 'plan'), ('motion', 'plan')]


def epoch_of(ckpt_dir):
    return int(re.sub(r'\D', '', ckpt_dir.name))


# ---- load every per-layer cell, no aggregation -----------------------------
rows = []
for ckpt in sorted(ROOT.glob('ckpt_*'), key=epoch_of):
    ep = epoch_of(ckpt)
    for a, b in PAIRS:
        f = ckpt / 'layer_conflict' / f'conflict_{a}_{b}_summary.csv'
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df['epoch'] = ep
        df['pair'] = f'{a}-{b}'
        rows.append(df)

data = pd.concat(rows, ignore_index=True)
data['layer'] = data['group']
# pseudo-shared groups: cosine not interpretable -> mark, keep separate
data['real'] = ~data['is_pseudo_shared_group'].fillna(False)
real = data[data['real']].copy()

epochs = sorted(data['epoch'].unique())
layers = sorted(data['layer'].unique())
pairs = [f'{a}-{b}' for a, b in PAIRS]
print(f'loaded {len(data)} cells | epochs={epochs} | {len(layers)} layers | {len(pairs)} pairs')
print(f'pseudo-shared (excluded from line plots): {(~data["real"]).sum()} cells')

cmap = plt.cm.viridis
lcolor = {L: cmap(i / max(len(layers) - 1, 1)) for i, L in enumerate(layers)}


# ---- 1. per-layer lines over epochs, faceted by pair -----------------------
def line_grid(metric, ref, fname, title):
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), squeeze=False)
    for ax, pair in zip(axes.flat, pairs):
        sub = real[real['pair'] == pair]
        for L, ldf in sub.groupby('layer'):
            ldf = ldf.sort_values('epoch')
            ax.plot(ldf['epoch'], ldf[metric], marker='o', ms=4,
                    color=lcolor[L], label=L, lw=1.3)
        ax.axhline(ref, color='gray', ls='--', lw=1)
        ax.set_title(pair, fontsize=12, fontweight='bold')
        ax.set_xscale('log'); ax.set_xticks(epochs)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel('epoch'); ax.set_ylabel(metric)
        if metric == 'conflict_ratio':
            ax.set_ylim(0, 1)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=7, ncol=1,
               bbox_to_anchor=(1.005, 0.5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 0.93, 0.97))
    fig.savefig(OUT / fname, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {fname}')


line_grid('conflict_ratio', 0.5, 'lines_conflict_ratio.png',
          'conflict_ratio per layer over epochs (no layer/epoch averaging)')
line_grid('mean_cos', 0.0, 'lines_mean_cos.png',
          'mean_cos per layer over epochs (no layer/epoch averaging)')


# ---- 2. layer x epoch heatmaps, one panel per pair -------------------------
def heatmap_grid(metric, fname, title, center):
    fig, axes = plt.subplots(2, 3, figsize=(22, 14), squeeze=False)
    if center == 0.5:
        vals = data[metric].dropna()
        norm = TwoSlopeNorm(vcenter=0.5, vmin=0, vmax=1)
        cm = 'RdBu_r'
    else:
        amax = np.nanmax(np.abs(data[metric].values))
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-amax, vmax=amax)
        cm = 'RdBu_r'
    im = None
    for ax, pair in zip(axes.flat, pairs):
        sub = data[data['pair'] == pair]
        mat = sub.pivot_table(index='layer', columns='epoch', values=metric)
        mat = mat.reindex(index=layers, columns=epochs)
        im = ax.imshow(mat.values, aspect='auto', cmap=cm, norm=norm)
        ax.set_xticks(range(len(epochs))); ax.set_xticklabels(epochs)
        ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers, fontsize=7)
        ax.set_title(pair, fontsize=12, fontweight='bold')
        ax.set_xlabel('epoch')
        for i in range(len(layers)):
            for j in range(len(epochs)):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=6, color='black')
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.savefig(OUT / fname, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {fname}')


heatmap_grid('conflict_ratio', 'heatmap_conflict_ratio.png',
             'conflict_ratio: layer x epoch (per pair)', 0.5)
heatmap_grid('mean_cos', 'heatmap_mean_cos.png',
             'mean_cos: layer x epoch (per pair)', 0.0)


# ---- 3. strongest-signal ranking (what de-aggregation reveals) -------------
real = real.copy()
real['abs_cos'] = real['mean_cos'].abs()
real['conflict_dev'] = (real['conflict_ratio'] - 0.5).abs()
cols = ['epoch', 'pair', 'layer', 'mean_cos', 'std_cos', 'conflict_ratio',
        'n_valid']
top_cos = real.nlargest(15, 'abs_cos')[cols]
top_conf = real.nlargest(15, 'conflict_dev')[cols]
top_cos.to_csv(OUT / 'top_signal_by_abs_cos.csv', index=False)
top_conf.to_csv(OUT / 'top_signal_by_conflict_dev.csv', index=False)

print('\n=== Top 15 cells by |mean_cos| (strongest directional signal) ===')
print(top_cos.to_string(index=False))
print('\n=== Top 15 cells by |conflict_ratio - 0.5| (most decisive) ===')
print(top_conf.to_string(index=False))

# noise-floor summary: how much of the data is indistinguishable from random
near_zero = (real['abs_cos'] < 0.02).mean()
near_half = (real['conflict_dev'] < 0.05).mean()
print(f'\nnoise-floor check (real cells, n={len(real)}):')
print(f'  |mean_cos| < 0.02         : {near_zero:.1%}')
print(f'  |conflict_ratio-0.5|<0.05 : {near_half:.1%}')
print(f'\noutputs in {OUT}')
