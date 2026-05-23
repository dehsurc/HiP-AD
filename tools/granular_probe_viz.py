#!/usr/bin/env python3
"""Per-layer probe affinity, de-averaged over epochs.

The notebook's plot_probe_layer_grid collapses all checkpoints into one
epoch-averaged 4x4 grid per layer. This script keeps epoch as an explicit
axis so a single layer's evolution across 1->3->6->18 epochs is visible.

Source : gradient_analysis_results/v3/ckpt_*/probe/probe_per_batch.csv
Output : gradient_analysis_results/v3/granular/
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path('/home/yongjae/e2e/HiP-AD-pcgrad/gradient_analysis_results/v3')
OUT = ROOT / 'granular'
OUT.mkdir(exist_ok=True)

TASKS = ['det', 'map', 'motion', 'plan']
VARIANT = 'normalized'   # matches notebook default; raw is fp16-noise dominated

# narrow-white diverging cmap: tiny +/- stays visibly colored
NARROW = LinearSegmentedColormap.from_list('rdbu_narrow', [
    (0.0, '#053061'), (0.25, '#4393c3'), (0.495, '#d1e5f0'),
    (0.5, '#ffffff'), (0.505, '#fddbc7'), (0.75, '#d6604d'), (1.0, '#67001f'),
], N=512)


def epoch_of(d):
    return int(re.sub(r'\D', '', d.name))


def parse_layer(name):
    parts = str(name).split('_', 1)
    return parts[0], (parts[1] if len(parts) > 1 else '_')


# ---- load every probe batch row, keep epoch -------------------------------
frames = []
for ckpt in sorted(ROOT.glob('ckpt_*'), key=epoch_of):
    f = ckpt / 'probe' / 'probe_per_batch.csv'
    if not f.exists():
        print(f'[skip] {f} missing'); continue
    df = pd.read_csv(f)
    df['epoch'] = epoch_of(ckpt)
    frames.append(df)
probe = pd.concat(frames, ignore_index=True)
probe = probe[(probe['variant'] == VARIANT) & (probe['steps'] == 1)].copy()

# ---- aggregate over BATCHES ONLY (epoch & layer kept separate) ------------
cell = (probe.groupby(['epoch', 'layer', 'source_task', 'target_task'])
             .agg(mean_delta=('delta', 'mean'),
                  std_delta=('delta', 'std'),
                  helpful_rate=('delta', lambda s: (s < 0).mean()),
                  n=('delta', 'size'))
             .reset_index())
cell['effect_size'] = cell['mean_delta'] / (cell['std_delta'] + 1e-12)

epochs = sorted(cell['epoch'].unique())
layers = sorted(cell['layer'].unique(), key=lambda L: (parse_layer(L)[0], parse_layer(L)[1]))
print(f'variant={VARIANT} | epochs={epochs} | {len(layers)} layers '
      f'| {len(cell)} (epoch,layer,src,tgt) cells')
cell.to_csv(OUT / 'probe_epoch_layer_cells.csv', index=False)


def matrix(sub, metric):
    m = sub.pivot_table(index='source_task', columns='target_task', values=metric)
    return m.reindex(index=TASKS, columns=TASKS)


# ---- 1. heat-grid: rows = layer, cols = epoch (flagship view) -------------
def heatgrid(metric, fname, fmt, title):
    amax = float(np.nanmax(np.abs(cell[metric].values))) or 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-amax, vmax=amax)
    nR, nC = len(layers), len(epochs)
    fig, axes = plt.subplots(nR, nC, figsize=(2.15 * nC, 2.15 * nR), squeeze=False)
    im = None
    for r, L in enumerate(layers):
        for c, ep in enumerate(epochs):
            ax = axes[r][c]
            sub = cell[(cell['layer'] == L) & (cell['epoch'] == ep)]
            mat = matrix(sub, metric)
            im = ax.imshow(mat.values, cmap=NARROW, norm=norm, aspect='auto')
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(TASKS, fontsize=6); ax.set_yticklabels(TASKS, fontsize=6)
            ax.grid(False)
            for i in range(4):
                for j in range(4):
                    v = mat.values[i, j]
                    if np.isfinite(v):
                        col = 'white' if abs(v) > 0.55 * amax else 'black'
                        ax.text(j, i, fmt.format(v), ha='center', va='center',
                                fontsize=5.5, color=col)
            if r == 0:
                ax.set_title(f'epoch {ep}', fontsize=10, fontweight='bold')
            if c == 0:
                ax.set_ylabel(L, fontsize=8, fontweight='bold')
    fig.suptitle(f'{title}\nrow=layer, col=epoch | cell: source(row)->target(col) | '
                 f'blue=helpful(delta<0)  red=harmful | variant={VARIANT}',
                 fontsize=12, fontweight='bold', y=1.002)
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02)
    fig.savefig(OUT / fname, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {fname}')


heatgrid('mean_delta', 'probe_heatgrid_mean_delta.png', '{:+.0e}',
         'Probe task-affinity per layer across epochs (mean_delta)')
heatgrid('effect_size', 'probe_heatgrid_effect_size.png', '{:+.2f}',
         'Probe effect_size (mean/std) per layer across epochs')


# ---- 2. line trajectories: one facet per layer, epoch on x ----------------
pairs = [(s, t) for s in TASKS for t in TASKS if s != t]   # 12 cross-task
cmap = plt.cm.tab20
pcolor = {p: cmap(i / len(pairs)) for i, p in enumerate(pairs)}

nC = 3
nR = int(np.ceil(len(layers) / nC))
fig, axes = plt.subplots(nR, nC, figsize=(6 * nC, 3.4 * nR), squeeze=False)
for idx, L in enumerate(layers):
    ax = axes[idx // nC][idx % nC]
    sub = cell[cell['layer'] == L]
    for (s, t) in pairs:
        line = (sub[(sub['source_task'] == s) & (sub['target_task'] == t)]
                .sort_values('epoch'))
        if line.empty:
            continue
        ax.plot(line['epoch'], line['mean_delta'], marker='o', ms=4, lw=1.2,
                color=pcolor[(s, t)], label=f'{s}->{t}')
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_title(L, fontsize=10, fontweight='bold')
    ax.set_xscale('log'); ax.set_xticks(epochs)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('epoch'); ax.set_ylabel('mean_delta')
for k in range(len(layers), nR * nC):
    axes[k // nC][k % nC].axis('off')
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', fontsize=8,
           bbox_to_anchor=(1.01, 0.5))
fig.suptitle('Per-layer cross-task probe trajectory over epochs '
             '(mean_delta, blue<0=helpful)', fontsize=13, fontweight='bold')
fig.tight_layout(rect=(0, 0, 0.93, 0.98))
fig.savefig(OUT / 'probe_lines_by_layer.png', dpi=120, bbox_inches='tight')
plt.close(fig)
print('  wrote probe_lines_by_layer.png')

# ---- 3. how much epoch-averaging actually hid -----------------------------
piv = cell.pivot_table(index=['layer', 'source_task', 'target_task'],
                       columns='epoch', values='mean_delta')
spread = (piv.max(axis=1) - piv.min(axis=1))           # epoch range per cell
sign_flip = piv.apply(lambda r: r.dropna().lt(0).any() and r.dropna().gt(0).any(),
                      axis=1)
print(f'\nepoch-spread of mean_delta per (layer,src,tgt) cell:')
print(f'  median range across epochs : {spread.median():.2e}')
print(f'  cells that flip sign       : {sign_flip.sum()}/{len(sign_flip)} '
      f'({sign_flip.mean():.0%})')
print(f'\noutputs in {OUT}')
