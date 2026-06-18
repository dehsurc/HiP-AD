"""det/map/motion contribution to planning at 1/3/9ep, on BOTH metrics
(L2 accuracy and collision safety), with scene-density stratification.

C_X (L2)   = L2(no_X) - L2(full)          (+ = task helps accuracy)
C_col_X    = col(no_X) - col(full)        (+ = task helps avoid collisions)
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import lib_io
from metrics import scene_l2
from strata_analysis import scene_features

EPOCHS = [1, 3, 9]
TASKS = {'det': 'no_det', 'map': 'no_map', 'motion': 'no_motion'}


def avg3(l2):
    return float(np.mean([l2[1], l2[3], l2[5]]))


def build():
    infos, _ = lib_io.load_val_infos()
    P = {(e, v): lib_io.load_slim_cache(e, v)['plan']
         for e in EPOCHS for v in ['full', 'no_det', 'no_map', 'no_motion']}
    rows = []
    for i, info in enumerate(infos):
        lf, inc = scene_l2(P[(9, 'full')][i], info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])
        if not inc:
            continue
        rec = {'token': info['token']}
        for e in EPOCHS:
            Lf = avg3(scene_l2(P[(e, 'full')][i], info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])[0])
            for t, v in TASKS.items():
                Lv = avg3(scene_l2(P[(e, v)][i], info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])[0])
                rec[f'CL2_{t}_e{e}'] = Lv - Lf
        rec.update(scene_features(info))
        rows.append(rec)
    df = pd.DataFrame(rows)
    col = pd.read_csv(os.path.join(lib_io.CACHE_DIR, 'collision_table.csv'))
    for e in EPOCHS:
        for t, v in TASKS.items():
            col[f'Ccol_{t}_e{e}'] = (col[f'col_e{e}_{v}'] - col[f'col_e{e}_full'])
    df = df.merge(col[['token'] + [f'Ccol_{t}_e{e}' for e in EPOCHS for t in TASKS]], on='token', how='left')
    return df


# scene-feature groups most relevant per task
TASK_FEATS = {
    'det':    ['n_boxes', 'n_near15', 'n_front', 'nearest_dist'],
    'motion': ['n_moving', 'n_fast', 'n_moving_front', 'n_near15'],
    'map':    ['map_curv', 'n_map', 'map_len', 'is_turn', 'ego_path_curv', 'ego_lateral'],
}


def corr_floor(df, col, n=300, seed=1):
    rng = np.random.RandomState(seed)
    x = df[col].values
    null = [pd.Series(rng.permutation(x)).corr(df['ctrl_random'], method='spearman') for _ in range(n)]
    return np.percentile(np.abs(null), 95)


def report(df):
    print('=' * 84)
    print('AGGREGATE CONTRIBUTION   (mean over scenes; + = task HELPS planning)')
    print('=' * 84)
    print(f'{"":8}{"--- L2 accuracy (m) ---":^33} {"--- Collision safety (%) ---":^33}')
    print(f'{"task":8}{"1ep":>11}{"3ep":>11}{"9ep":>11} {"1ep":>11}{"3ep":>11}{"9ep":>11}')
    for t in TASKS:
        l2 = [df[f'CL2_{t}_e{e}'].mean() for e in EPOCHS]
        co = [df[f'Ccol_{t}_e{e}'].mean() * 100 for e in EPOCHS]
        print(f'{t:8}' + ''.join(f'{v:+11.4f}' for v in l2) + ' ' + ''.join(f'{v:+11.3f}' for v in co))
    print('  (L2: lower error is better, so +C = removing task raised error = task helped accuracy)')
    print('  (Collision: +C = removing task raised collisions = task helped safety)')

    for metric, pref, unit, scale in [('L2', 'CL2', 'm', 1.0), ('Collision', 'Ccol', '%', 100.0)]:
        print('\n' + '=' * 84)
        print(f'WHERE each task helps — {metric}: best scene-stratifier per task×epoch')
        print('=' * 84)
        for t in TASKS:
            for e in EPOCHS:
                ycol = f'{pref}_{t}_e{e}'
                y = df[ycol].values
                # pick stratifier with largest |spearman| among task-relevant feats
                best = None
                for fcol in TASK_FEATS[t]:
                    rho = pd.Series(df[fcol].values).corr(pd.Series(y), method='spearman')
                    if best is None or abs(rho) > abs(best[1]):
                        best = (fcol, rho)
                fcol, rho = best
                floor = corr_floor(df, ycol)
                sig = 'REAL' if abs(rho) > floor else 'noise'
                # binned trend on the best feature
                try:
                    b = pd.qcut(df[fcol].rank(method='first'), 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                    g = df.groupby(b)[ycol].mean() * scale
                    trend = ' -> '.join(f'{v:+.3f}' for v in g.values)
                except Exception:
                    trend = 'n/a'
                print(f'  {t:6} @{e}ep: best={fcol:13} rho={rho:+.3f} (floor {floor:.3f} -> {sig:5}) | '
                      f'{fcol} Q1..Q4 [{unit}]: {trend}')


if __name__ == '__main__':
    out = os.path.join(lib_io.CACHE_DIR, 'task_epoch_table.csv')
    if os.path.exists(out):
        df = pd.read_csv(out); print(f'[loaded {out}] n={len(df)}')
    else:
        df = build(); df.to_csv(out, index=False); print(f'[built {out}] n={len(df)}')
    report(df)
