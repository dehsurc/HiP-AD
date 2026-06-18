"""Scene-level gradient analysis across 1/3/9ep:
  (1) planning sensitivity  = ||g_plan|| per probe-batch/layer  (per-scene-ish)
  (2) does the gradient-level signal (planning sensitivity, det-plan conflict,
      det->plan transfer) concentrate in the same scenes (object density / maneuver)
      where the OUTCOME analysis said det/map matter?

Probe is TRAIN split, batch_size=6.  Recover each batch's 6 train tokens via the
verified DataLoader recipe, attach mean scene features, join to probe metrics.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
import lib_io
from strata_analysis import scene_features

GA = '/home/yongjae/e2e/HiP-AD-pcgrad/gradient_analysis_results/inter_gnn_b1000_pcgrad'
TRAIN_PKL = '/home/yongjae/e2e/HiP-AD-pcgrad/data/infos/nuscenes_infos_train.pkl'
EPOCHS = [1, 3, 9]


def recover_batches(n_batches=1000, bs=6):
    d = pickle.load(open(TRAIN_PKL, 'rb'))
    infos = d['infos'] if isinstance(d, dict) else d
    infos = sorted(infos, key=lambda e: e['timestamp'])
    g = torch.Generator(); g.manual_seed(42)
    torch.empty((), dtype=torch.int64).random_(generator=g)          # base_seed draw FIRST
    perm = torch.randperm(len(infos), generator=g).tolist()
    batches = [perm[i:i + bs] for i in range(0, len(perm), bs) if len(perm[i:i + bs]) == bs]
    return infos, batches[:n_batches]


def batch_features(infos, batches):
    feats = ['n_near15', 'n_boxes', 'n_moving', 'map_curv', 'is_turn', 'nearest_dist']
    rows = []
    for bi, idxs in enumerate(batches):
        fs = [scene_features(infos[j]) for j in idxs]
        rec = {'batch_idx': bi}
        for k in feats:
            rec[k] = float(np.mean([f[k] for f in fs]))
        rows.append(rec)
    return pd.DataFrame(rows)


def probe_metrics(epoch):
    p = pd.read_csv(f'{GA}/ckpt_{epoch}ep/probe/probe_per_batch.csv')
    p = p[p.steps == p.steps.min()]
    out = {}
    # planning sensitivity = ||g_plan|| (source=plan), mean over layers
    ps = p[(p.source_task == 'plan')].groupby('batch_idx')['grad_norm'].mean()
    out['plan_sens'] = ps
    # per-layer plan sens (dec1)
    for L in ['dec0_inter_gnn_0', 'dec1_inter_gnn_0', 'dec2_inter_gnn_0']:
        out['plan_sens_' + L[:4]] = p[(p.source_task == 'plan') & (p.layer == L)].groupby('batch_idx')['grad_norm'].mean()
    # det->plan transfer (raw delta) and det-plan / motion-plan / map-plan conflict (cos)
    raw = p[p.variant == 'raw']
    out['detplan_delta'] = raw[(raw.source_task == 'det') & (raw.target_task == 'plan')].groupby('batch_idx')['delta'].mean()
    pc = p[p.variant.str.startswith('pcgrad')]
    for src in ['det', 'motion', 'map']:
        sub = pc[(pc.source_task == src) & (pc.other_task == 'plan')]
        out[f'{src}plan_cos'] = sub.groupby('batch_idx')['cos_source_other'].mean()
    return pd.DataFrame(out).reset_index()


def floor95(df, a, b, n=300, seed=1):
    rng = np.random.RandomState(seed); x = df[a].values
    null = [pd.Series(rng.permutation(x)).corr(df[b], method='spearman') for _ in range(n)]
    return np.nanpercentile(np.abs(null), 95)


if __name__ == '__main__':
    print('recovering train batch tokens + features ...', flush=True)
    infos, batches = recover_batches()
    BF = batch_features(infos, batches)
    BF['ctrl'] = np.random.RandomState(0).rand(len(BF))
    print(f'  {len(BF)} batches (×6 scenes); mean n_near15={BF.n_near15.mean():.1f}, turn frac={BF.is_turn.mean():.2f}')

    print('\n' + '=' * 80)
    print('(2) PLANNING SENSITIVITY  ||g_plan|| per batch, by layer & checkpoint')
    print('=' * 80)
    for e in EPOCHS:
        m = probe_metrics(e)
        print(f'  {e}ep: plan_sens mean={m.plan_sens.mean():.3f}  median={m.plan_sens.median():.3f}  '
              f'| by layer dec0={m.plan_sens_dec0.mean():.2f} dec1={m.plan_sens_dec1.mean():.2f} dec2={m.plan_sens_dec2.mean():.2f}')

    print('\n' + '=' * 80)
    print('GRADIENT-LEVEL SIGNAL vs SCENE FEATURES  (Spearman; *=above perm floor)')
    print('  Q: does gradient conflict/sensitivity concentrate where outcome said det/map matter?')
    print('=' * 80)
    METRICS = ['plan_sens', 'detplan_delta', 'detplan_cos', 'motionplan_cos', 'mapplan_cos']
    FEATS = ['n_near15', 'n_boxes', 'n_moving', 'map_curv', 'is_turn']
    for e in EPOCHS:
        m = probe_metrics(e)
        d = BF.merge(m, on='batch_idx')
        print(f'\n-- {e}ep --   (ctrl floor shown per metric)')
        hdr = f'{"metric":15}' + ''.join(f'{f:>11}' for f in FEATS) + f'{"|ctrl":>9}'
        print(hdr)
        for met in METRICS:
            fl = floor95(d, met, 'ctrl')
            cells = []
            for f in FEATS:
                rho = d[met].corr(d[f], method='spearman')
                star = '*' if abs(rho) > fl else ' '
                cells.append(f'{rho:+.3f}{star}')
            cells.append(f'{fl:.3f}')
            print(f'{met:15}' + ''.join(f'{c:>11}' for c in cells))
