"""Detection-focused collision analysis across checkpoints 1/3/9ep, stratified by
object density.  C_col_X = obj_box_col(no_X) - obj_box_col(full) per scene.
  C_col_X > 0  =>  removing task X INCREASES collisions => X helps avoid collisions.

Reuses the verified collision path (gate: 9ep full = 0.082%). The eval's double
x-flip cancels for collision, so plan coords are used as-is against fut_boxes.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
import lib_io
from collision import build_fut_boxes
from projects.mmdet3d_plugin.datasets.evaluation_nuscenes.planning.planning_eval import PlanningMetric

EPOCHS = [1, 3, 9]
VARIANTS = ['full', 'no_det', 'no_map', 'no_motion']
PM = PlanningMetric()
OUT = os.path.join(lib_io.CACHE_DIR, 'collision_table.csv')


def avg3(x):
    cum = np.array([x[:i + 1].mean() for i in range(len(x))])
    return float(np.mean([cum[1], cum[3], cum[5]]))


def build():
    infos, _ = lib_io.load_val_infos()
    feats = pd.read_csv(os.path.join(lib_io.CACHE_DIR, 'strata_table_e9.csv'))
    feat_by_tok = feats.set_index('token')[['n_boxes', 'n_near15', 'n_front', 'nearest_dist']].to_dict('index')

    # build fut_boxes + GT collision once per scene (plan-independent)
    print('building fut_boxes + GT collision per scene ...', flush=True)
    t0 = time.time()
    fb_cache, gtc_cache, idx_ok = {}, {}, []
    for i, info in enumerate(infos):
        fb = build_fut_boxes(infos, i, use_valid_flag=True)
        if fb is None:
            continue
        gt_cum = np.cumsum(np.asarray(info['gt_ego_fut_trajs'], float)[:6, :2], axis=0)
        gtc = PM.evaluate_single_coll(torch.tensor(gt_cum, dtype=torch.float32), [[b] for b in fb])
        fb_cache[i] = [[b] for b in fb]; gtc_cache[i] = gtc; idx_ok.append(i)
    print(f'  {len(idx_ok)} scenes, {time.time()-t0:.0f}s', flush=True)

    # per (epoch, variant) collision
    plans = {(e, v): lib_io.load_slim_cache(e, v)['plan'] for e in EPOCHS for v in VARIANTS}
    rows = []
    for i in idx_ok:
        rec = {'token': infos[i]['token']}
        rec.update(feat_by_tok.get(infos[i]['token'], {}))
        for e in EPOCHS:
            for v in VARIANTS:
                pc = PM.evaluate_single_coll(torch.tensor(np.asarray(plans[(e, v)][i])[:6, :2], dtype=torch.float32), fb_cache[i])
                obj = (pc & ~gtc_cache[i]).numpy().astype(float)
                rec[f'col_e{e}_{v}'] = avg3(obj)
        rows.append(rec)
        if len(rows) % 1000 == 0:
            print(f'  ...{len(rows)} scenes done ({time.time()-t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f'[wrote {OUT}]  n={len(df)}', flush=True)
    return df


def analyze(df):
    print('\n' + '=' * 78); print('AGGREGATE collision (%) and C_col_X = no_X - full  (cross-check vs report)'); print('=' * 78)
    print(f'{"epoch":6} {"full":>8} | {"C_col_det":>10} {"C_col_map":>10} {"C_col_motion":>12}')
    for e in EPOCHS:
        full = df[f'col_e{e}_full'].mean() * 100
        cd = (df[f'col_e{e}_no_det'] - df[f'col_e{e}_full']).mean() * 100
        cm = (df[f'col_e{e}_no_map'] - df[f'col_e{e}_full']).mean() * 100
        cmo = (df[f'col_e{e}_no_motion'] - df[f'col_e{e}_full']).mean() * 100
        print(f'{e:>4}ep {full:7.3f}% | {cd:+9.3f}% {cm:+9.3f}% {cmo:+11.3f}%')
    print('  report TG_col: det 1ep +0.190, 3ep +0.001, 9ep +0.003 ; map 1ep +0.039, 9ep +0.010')

    print('\n' + '=' * 78); print('DETECTION: C_col_det stratified by object density (% scenes, mean C_col_det in %)'); print('=' * 78)
    for e in EPOCHS:
        df['_cd'] = (df[f'col_e{e}_no_det'] - df[f'col_e{e}_full'])
        print(f'\n-- {e}ep --  (C_col_det>0 = removing detection caused MORE collisions = detection helped)')
        for fcol in ['n_boxes', 'n_near15', 'n_front']:
            try:
                df['_b'] = pd.qcut(df[fcol].rank(method='first'), 4, labels=['Q1lo', 'Q2', 'Q3', 'Q4hi'])
                g = df.groupby('_b')['_cd'].agg(['mean', 'count'])
                cells = ' | '.join(f'{idx}:{r["mean"]*100:+.3f}%(n={int(r["count"])})' for idx, r in g.iterrows())
                print(f'   by {fcol:9}: {cells}')
            except Exception as ex:
                print(f'   by {fcol}: {ex}')
        # how concentrated: fraction of scenes where removing det added a collision the full model avoided
        added = (df[f'col_e{e}_no_det'] > df[f'col_e{e}_full']).mean()
        removed = (df[f'col_e{e}_no_det'] < df[f'col_e{e}_full']).mean()
        print(f'   scenes where no_det collides MORE: {added*100:.2f}% | LESS: {removed*100:.2f}% | same: {(1-added-removed)*100:.2f}%')


if __name__ == '__main__':
    if os.path.exists(OUT):
        df = pd.read_csv(OUT); print(f'[loaded {OUT}] n={len(df)}')
    else:
        df = build()
    analyze(df)
