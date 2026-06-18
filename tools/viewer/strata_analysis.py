"""Stage-1 distribution analysis: per-scene transfer (C_det/C_map/C_motion) vs
per-scene interpretable features. Val split, @9ep. No GPU.

Transfer convention (matches the report's TG): C_X = L2(no_X) - L2(full), per scene.
  C_X > 0  =>  task X HELPS planning in that scene.
  C_X < 0  =>  task X HARMS planning in that scene.
Aggregate mean(C_X) must match tg_report: det -0.0045, map -0.0333, motion -0.0132.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import lib_io
from metrics import scene_l2

EP = 9
np.random.seed(0)


def avg3(l2):  # mean of L2 at 1.0/2.0/3.0s
    return float(np.mean([l2[1], l2[3], l2[5]]))


def poly_len_curv(poly):
    p = np.asarray(poly, float)
    if len(p) < 2:
        return 0.0, 0.0
    seg = np.diff(p, axis=0)
    length = float(np.linalg.norm(seg, axis=1).sum())
    if len(seg) < 2:
        return length, 0.0
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    dang = np.abs(np.diff(ang))
    dang = np.minimum(dang, 2 * np.pi - dang)
    return length, float(dang.sum())


def scene_features(info):
    f = {}
    boxes = np.asarray(info['gt_boxes'], float)          # (N,7) x,y,z,...
    xy = boxes[:, :2] if len(boxes) else np.zeros((0, 2))
    dist = np.linalg.norm(xy, axis=1) if len(xy) else np.array([])
    front = (xy[:, 0] > 0) & (np.abs(xy[:, 1]) < 6) & (xy[:, 0] < 30) if len(xy) else np.array([], bool)
    f['n_boxes'] = len(boxes)
    f['n_near15'] = int((dist < 15).sum())
    f['nearest_dist'] = float(dist.min()) if len(dist) else 60.0
    f['n_front'] = int(front.sum())
    # motion
    vel = np.asarray(info['gt_velocity'], float)
    spd = np.linalg.norm(vel, axis=1) if len(vel) else np.array([])
    f['n_moving'] = int((spd > 0.5).sum())
    f['n_fast'] = int((spd > 3.0).sum())
    f['n_moving_front'] = int(((spd > 0.5)[:len(front)] & front[:len(spd)]).sum()) if len(spd) and len(front) else 0
    # ego
    ego = np.cumsum(np.asarray(info['gt_ego_fut_trajs'], float)[:6], axis=0)
    f['ego_speed'] = float(np.linalg.norm(np.asarray(info['gt_ego_fut_trajs'])[0]) * 2)
    cmd = int(np.argmax(info['gt_ego_fut_cmd']))
    f['is_turn'] = int(cmd != 2)               # 2==straight (per our label guess)
    # ego path curvature (total turning over the planned GT path)
    _, f['ego_path_curv'] = poly_len_curv(np.vstack([[0, 0], ego]))
    f['ego_lateral'] = float(abs(ego[-1, 1]))  # final lateral offset
    # map complexity
    ma = info['map_annos']
    n_elem = 0; tot_len = 0.0; tot_curv = 0.0
    per_cls = {0: 0, 1: 0, 2: 0}
    for cls, lst in ma.items():
        per_cls[cls] = per_cls.get(cls, 0) + len(lst)
        for poly in lst:
            n_elem += 1
            L, C = poly_len_curv(poly)
            tot_len += L; tot_curv += C
    f['n_map'] = n_elem
    f['map_len'] = tot_len
    f['map_curv'] = tot_curv
    f['n_div'] = per_cls.get(0, 0); f['n_ped'] = per_cls.get(1, 0); f['n_bound'] = per_cls.get(2, 0)
    # nonsense control feature (must NOT predict anything)
    f['ctrl_lidarpts'] = int(info.get('num_lidar_pts', np.array([0])).sum()) if hasattr(info.get('num_lidar_pts', 0), 'sum') else 0
    f['ctrl_random'] = float(np.random.rand())
    return f


def build_table():
    infos, _ = lib_io.load_val_infos()
    P = {v: lib_io.load_slim_cache(EP, v)['plan'] for v in ['full', 'no_det', 'no_map', 'no_motion']}
    rows = []
    for i, info in enumerate(infos):
        lf, inc = scene_l2(P['full'][i], info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])
        if not inc:
            continue
        Lf = avg3(lf)
        rec = {'token': info['token'], 'scene_token': info['scene_token'], 'L2_full': Lf}
        for v, key in [('no_det', 'C_det'), ('no_map', 'C_map'), ('no_motion', 'C_motion')]:
            lv, _ = scene_l2(P[v][i], info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])
            rec[key] = avg3(lv) - Lf       # C_X = L2(no_X) - L2(full)
        rec.update(scene_features(info))
        rows.append(rec)
    return pd.DataFrame(rows)


def dist_report(df):
    tasks = ['C_det', 'C_map', 'C_motion']
    print('=' * 78); print(f'PER-SCENE TRANSFER DISTRIBUTION  (n={len(df)})'); print('=' * 78)
    print(f'{"task":8} {"mean":>9} {"median":>9} {"%help>0":>8} {"%|<.05|":>8} {"std":>7} {"skew":>7} {"p1":>7} {"p99":>7}')
    for t in tasks:
        x = df[t].values
        sk = float(((x - x.mean())**3).mean() / (x.std()**3 + 1e-12))
        print(f'{t:8} {x.mean():9.4f} {np.median(x):9.4f} {(x>0).mean()*100:7.1f}% {(np.abs(x)<0.05).mean()*100:7.1f}% '
              f'{x.std():7.3f} {sk:7.2f} {np.percentile(x,1):7.2f} {np.percentile(x,99):7.2f}')
    print('  (mean should match report TG: det -0.0045, map -0.0333, motion -0.0132)')

    print('\n' + '=' * 78); print('FEATURE DISTRIBUTION (quantiles)'); print('=' * 78)
    feats = ['n_boxes','n_near15','nearest_dist','n_front','n_moving','n_fast','n_moving_front',
             'ego_speed','is_turn','ego_path_curv','ego_lateral','n_map','map_len','map_curv','n_div','n_ped','n_bound']
    q = df[feats].quantile([0,.25,.5,.75,1.0]).round(2)
    print(q.T.to_string())

    print('\n' + '=' * 78); print('SPEARMAN corr( feature , transfer )  + permutation p-value'); print('=' * 78)
    print('  >0 : higher feature -> task helps more.   ctrl_* must be ~0 (lie-detector).')
    hdr = f'{"feature":16}' + ''.join(f'{t:>14}' for t in tasks)
    print(hdr)
    rng = np.random.RandomState(1)
    for fcol in feats + ['ctrl_lidarpts','ctrl_random']:
        cells = []
        fv = df[fcol].values.astype(float)
        for t in tasks:
            tv = df[t].values
            rho = pd.Series(fv).corr(pd.Series(tv), method='spearman')
            # permutation p-value (200 shuffles)
            null = np.array([pd.Series(fv).corr(pd.Series(rng.permutation(tv)), method='spearman') for _ in range(200)])
            p = (np.abs(null) >= abs(rho)).mean()
            star = '***' if p < 0.005 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            cells.append(f'{rho:+.3f}{star:<3}')
        print(f'{fcol:16}' + ''.join(f'{c:>14}' for c in cells))

    print('\n' + '=' * 78); print('STRATIFIED MEANS — top feature per task (binned)'); print('=' * 78)
    pairs = [('C_det','n_front'), ('C_det','n_near15'), ('C_motion','n_moving_front'),
             ('C_map','map_curv'), ('C_map','n_map'), ('C_map','is_turn'), ('C_map','ego_path_curv')]
    for t, fcol in pairs:
        try:
            if df[fcol].nunique() <= 3:
                g = df.groupby(fcol)[t].agg(['mean','count'])
            else:
                df['_b'] = pd.qcut(df[fcol].rank(method='first'), 4, labels=['Q1lo','Q2','Q3','Q4hi'])
                g = df.groupby('_b')[t].agg(['mean','count'])
            means = ' | '.join(f'{idx}:{r["mean"]:+.4f}(n={int(r["count"])})' for idx,r in g.iterrows())
            print(f'  {t} ~ {fcol:14}: {means}')
        except Exception as e:
            print(f'  {t} ~ {fcol}: err {e}')
    df.drop(columns=['_b'], errors='ignore', inplace=True)


if __name__ == '__main__':
    out = os.path.join(lib_io.CACHE_DIR, 'strata_table_e9.csv')
    if os.path.exists(out):
        df = pd.read_csv(out)
        print(f'[loaded cached table {out}]  n={len(df)}')
    else:
        df = build_table()
        df.to_csv(out, index=False)
        print(f'[built table -> {out}]  n={len(df)}')
    dist_report(df)
