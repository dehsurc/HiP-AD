#!/usr/bin/env python3
"""Is the per-layer epoch variation a real trend or just noise?

For each (layer, source, target) cell we have mean_delta at 4 epochs, each
estimated from n=100 batches with std_delta. Standard error SE = std/sqrt(n).
A "real" epoch trend means the spread across epochs exceeds measurement noise.

Two questions:
  (a) within a cell: does mean_delta change across epochs beyond noise?
  (b) across layers: do layers agree on the direction of change?
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('/home/yongjae/e2e/HiP-AD-pcgrad/gradient_analysis_results/v3/granular')
cell = pd.read_csv(OUT / 'probe_epoch_layer_cells.csv')

cell['se'] = cell['std_delta'] / np.sqrt(cell['n'])
epochs = sorted(cell['epoch'].unique())

rows = []
for (layer, s, t), g in cell.groupby(['layer', 'source_task', 'target_task']):
    g = g.set_index('epoch').reindex(epochs)
    md = g['mean_delta'].values
    se = g['se'].values
    if np.isnan(md).any():
        continue
    # between-epoch F-like ratio: variance of epoch means vs mean SE^2
    between = np.var(md, ddof=1)
    within = np.nanmean(se ** 2)
    F = between / within if within > 0 else np.nan
    rng = md.max() - md.min()
    pooled_se = np.sqrt(np.nanmean(se ** 2))
    # monotonic? sign of consecutive diffs all equal
    diffs = np.diff(md)
    monotonic = np.all(diffs > 0) or np.all(diffs < 0)
    rows.append(dict(layer=layer, source=s, target=t, kind='self' if s == t else 'cross',
                      range=rng, pooled_se=pooled_se, range_in_se=rng / pooled_se,
                      F=F, significant=F > 3.0, monotonic=monotonic,
                      net_change=md[-1] - md[0]))
res = pd.DataFrame(rows)

print(f'cells analysed: {len(res)}  (n=100 batches per epoch point)\n')
for kind in ['self', 'cross', 'all']:
    sub = res if kind == 'all' else res[res['kind'] == kind]
    print(f'--- {kind} (n={len(sub)}) ---')
    print(f'  epoch-range / SE   median : {sub["range_in_se"].median():.2f}  '
          f'(>~3 = trend visible above noise)')
    print(f'  F>3 (real trend)         : {sub["significant"].mean():.0%}')
    print(f'  monotonic across 4 ep    : {sub["monotonic"].mean():.0%}  '
          f'(random expectation ~25%)')
    print()

# (b) cross-layer agreement: for each (source,target) pair, do layers agree
# on the sign of net change epoch1 -> epoch18?
print('--- cross-layer consistency: net change sign agreement per (src->tgt) ---')
for (s, t), g in res.groupby(['source', 'target']):
    pos = (g['net_change'] > 0).sum()
    neg = (g['net_change'] < 0).sum()
    frac = max(pos, neg) / len(g)
    flag = 'CONSISTENT' if frac >= 0.8 else ''
    print(f'  {s:>6}->{t:<6}: {pos:2d} up / {neg:2d} down of {len(g):2d} layers '
          f'-> {frac:.0%} agree  {flag}')
