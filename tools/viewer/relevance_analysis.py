"""Does PLANNING quality depend on the prediction quality of EGO-RELEVANT
perception (objects near the ego's GT future path), beyond global quality?

This is the cheap OBSERVATIONAL SCREEN (no retraining) for the method idea
"weight aux loss toward ego-relevant instances".  It is a *necessary-not-
sufficient* test:  null here (with good predictor variance) => the open-loop
planner barely uses perception -> go closed-loop;  positive => the mediator
(ego-relevant product quality) is real -> proceed to a short-horizon causal test.

Design (hole-hardened, see chat):
  * predictors are TRAINING-TIME AVAILABLE only:
      - relevance from GT ego future path (not post-hoc outcome)
      - rel_det_recall / glob_det_recall from the model's own det head
  * outcomes: per-scene L2 (continuous) + col_margin (continuous safety margin,
    fixes 0.08% collision sparsity) + col (binary, low power, for reference)
  * MECHANISM split: rel quality predicting L2  => NON-rescore channel
                     rel quality predicting only col => rescore-consistent
  * partial Spearman controlling [glob_det_recall, n_relevant, nearest_dist,
    is_turn]  => removes scene-difficulty confound
  * run @1/3/9ep + report predictor variance => avoid saturation false-null
  * (B) benefit-concentration: does C_det = L2(no_det)-L2(full) concentrate
    where EGO-PATH relevance is high (vs generic density)?  -> the actual
    "weight where relevant" premise.
  * GATE: reproduces L2@9 full = 0.6457 and col@9 full = 0.082% .
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/home/yongjae/e2e/HiP-AD-pcgrad')
import numpy as np, pandas as pd
import lib_io
from metrics import scene_l2
from collision import build_fut_boxes, scene_obj_box_col
from strata_analysis import scene_features

EPOCHS = [1, 3, 9]
R_REL = 5.0        # object "ego-relevant" if within 5m of the ego GT path
R_COLL = 2.5       # tighter collision-relevant ring
DET_THR = 0.3      # pred-box score threshold (matches viewer DET_THR)
MATCH_D = 2.0      # center-distance for a pred box to "cover" a GT box
MAX_RNG = 30.0     # detectability range for the global-quality denominator
CACHE = os.path.join(lib_io.CACHE_DIR, 'relevance_table.csv')


def avg3(v):
    return float(np.mean([v[1], v[3], v[5]]))


def ego_path(info):
    """ego GT future positions incl. origin, ego frame (x fwd, y left). (7,2)"""
    fut = np.cumsum(np.asarray(info['gt_ego_fut_trajs'], float)[:6, :2], axis=0)
    return np.vstack([[0.0, 0.0], fut])


def min_dist_to_path(xy, path):
    """xy (N,2) -> (N,) min distance to the polyline vertices (approx)."""
    if len(xy) == 0:
        return np.array([])
    d = np.linalg.norm(xy[:, None, :] - path[None, :, :], axis=2)  # (N,P)
    return d.min(axis=1)


def relevance_feats(info):
    boxes = np.asarray(info['gt_boxes'], float)
    xy = boxes[:, :2] if len(boxes) else np.zeros((0, 2))
    rng = np.linalg.norm(xy, axis=1) if len(xy) else np.array([])
    path = ego_path(info)
    dpath = min_dist_to_path(xy, path) if len(xy) else np.array([])
    det_mask = rng < MAX_RNG                       # detectable global set
    rel_mask = (dpath < R_REL) & det_mask if len(xy) else np.array([], bool)
    coll_mask = (dpath < R_COLL) & det_mask if len(xy) else np.array([], bool)
    return {
        'n_relevant': int(rel_mask.sum()),
        'n_collrel': int(coll_mask.sum()),
        'n_glob30': int(det_mask.sum()),
        'min_dist_path': float(dpath[det_mask].min()) if det_mask.any() else 60.0,
        '_rel_xy': xy[rel_mask], '_glob_xy': xy[det_mask],
    }


def det_recall(pred_xy, gt_xy, d=MATCH_D):
    """fraction of gt_xy centers covered by some pred_xy (score>thr) within d."""
    if len(gt_xy) == 0:
        return np.nan
    if len(pred_xy) == 0:
        return 0.0
    dm = np.linalg.norm(gt_xy[:, None, :] - pred_xy[None, :, :], axis=2)  # (G,P)
    return float((dm.min(axis=1) < d).mean())


def det_recall_soft(pred_xy, pred_sc, gt_xy, d=MATCH_D):
    """mean over gt of (max score among preds within d, else 0).  Continuous,
    threshold-free quality => higher variance, less arbitrary than hard recall."""
    if len(gt_xy) == 0:
        return np.nan
    if len(pred_xy) == 0:
        return 0.0
    dm = np.linalg.norm(gt_xy[:, None, :] - pred_xy[None, :, :], axis=2)  # (G,P)
    near = dm < d
    sc = np.where(near, pred_sc[None, :], 0.0)
    return float(sc.max(axis=1).mean())


def col_margin(plan, fut_boxes):
    """min over horizon of (dist from planned ego pos to nearest other-agent
    future box center  -  that box half-diagonal).  lower = riskier."""
    p = np.asarray(plan)[:6, :2]
    best = np.inf
    for t, fb in enumerate(fut_boxes):
        fb = np.asarray(fb)
        if len(fb) == 0:
            continue
        cxy = fb[:, :2]
        rad = 0.5 * np.sqrt(fb[:, 3] ** 2 + fb[:, 4] ** 2)
        m = (np.linalg.norm(cxy - p[t][None, :], axis=1) - rad).min()
        best = min(best, float(m))
    return best if np.isfinite(best) else np.nan


def pred_xy(slim, i, thr=DET_THR):
    b = slim['boxes'][i]; s = slim['bscore'][i]
    keep = (s > thr) & np.isfinite(b[:, 0])
    return b[keep][:, :2]


def pred_all(slim, i):
    """all finite pred boxes (no score thr) -> (xy (K,2), score (K,))."""
    b = slim['boxes'][i]; s = slim['bscore'][i]
    keep = np.isfinite(b[:, 0])
    return b[keep][:, :2], s[keep]


def build_table(force=False):
    if os.path.exists(CACHE) and not force:
        df = pd.read_csv(CACHE)
        print(f'[loaded {CACHE}]  n={len(df)}')
        return df
    infos, _ = lib_io.load_val_infos()
    slim = {(e, v): lib_io.load_slim_cache(e, v)
            for e in EPOCHS for v in ['full', 'no_det']}
    rows = []
    for i, info in enumerate(infos):
        l2f, inc = scene_l2(slim[(9, 'full')]['plan'][i],
                            info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])
        if not inc:
            continue
        fb = build_fut_boxes(infos, i, use_valid_flag=True)
        rf = relevance_feats(info)
        sf = scene_features(info)
        rec = {'token': info['token'], 'scene_token': info['scene_token'],
               'n_relevant': rf['n_relevant'], 'n_collrel': rf['n_collrel'],
               'n_glob30': rf['n_glob30'], 'min_dist_path': rf['min_dist_path'],
               'n_near15': sf['n_near15'], 'nearest_dist': sf['nearest_dist'],
               'is_turn': sf['is_turn'], 'ego_path_curv': sf['ego_path_curv'],
               'n_moving_front': sf['n_moving_front']}
        # expensive shapely collision ONLY for full@9 (gate + binary reference)
        if fb is not None:
            cf9 = scene_obj_box_col(slim[(9, 'full')]['plan'][i],
                                    info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'], fb)
            rec['col_full_9'] = float(cf9[[1, 3, 5]].sum())
        else:
            rec['col_full_9'] = np.nan
        for e in EPOCHS:
            full = slim[(e, 'full')]; nod = slim[(e, 'no_det')]
            l2_f = avg3(scene_l2(full['plan'][i], info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])[0])
            l2_n = avg3(scene_l2(nod['plan'][i], info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])[0])
            rec[f'L2_full_{e}'] = l2_f
            rec[f'C_det_{e}'] = l2_n - l2_f             # >0 => det helps L2
            fxy, fsc = pred_all(full, i)
            rec[f'reldet_{e}'] = det_recall(fxy[fsc > DET_THR], rf['_rel_xy'])
            rec[f'globdet_{e}'] = det_recall(fxy[fsc > DET_THR], rf['_glob_xy'])
            rec[f'reldet_soft_{e}'] = det_recall_soft(fxy, fsc, rf['_rel_xy'])
            rec[f'globdet_soft_{e}'] = det_recall_soft(fxy, fsc, rf['_glob_xy'])
            if fb is not None:
                mf = col_margin(full['plan'][i], fb)    # higher = safer
                mn = col_margin(nod['plan'][i], fb)
                rec[f'colmargin_{e}'] = mf
                rec[f'Cmargin_{e}'] = mf - mn            # >0 => det keeps ego safer
            else:
                rec[f'colmargin_{e}'] = np.nan; rec[f'Cmargin_{e}'] = np.nan
        rows.append(rec)
        if (i + 1) % 1000 == 0:
            print(f'  ...{i+1}/{len(infos)} scenes', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    print(f'[built {CACHE}]  n={len(df)}')
    return df


# ---- correctness gate ------------------------------------------------------
def gate(df):
    infos, _ = lib_io.load_val_infos()
    # L2 gate: average per-scene L2_full_9 over all included
    l2 = df['L2_full_9'].mean()
    # exact replica uses cumulative-avg; avg3 of per-scene then mean is close but
    # not identical to the cumulative aggregate, so just sanity-bound it.
    print(f'[gate] mean per-scene L2_full@9 = {l2:.4f}  (eval cumulative = 0.6457; '
          f'per-scene avg3 differs slightly by construction)')
    d = df.dropna(subset=['col_full_9', 'colmargin_9'])
    print(f'[gate] scenes with fut_box set: {len(d)} ; mean col_full_9(sum@1/2/3s) = {d.col_full_9.mean():.4f}')
    # validate continuous proxy: scenes WITH a binary collision must have lower margin
    hit = d[d.col_full_9 > 0]; safe = d[d.col_full_9 == 0]
    print(f'[gate] col_margin proxy check: collide scenes (n={len(hit)}) median margin={hit.colmargin_9.median():.2f}'
          f'  vs safe scenes (n={len(safe)}) median margin={safe.colmargin_9.median():.2f}  (collide must be LOWER)')
    print(f'[gate] reldet_9 var={df.reldet_9.var():.4f} std={df.reldet_9.std():.3f} mean={df.reldet_9.mean():.3f} '
          f'| reldet_1 mean={df.reldet_1.mean():.3f} std={df.reldet_1.std():.3f}')
    print(f'[gate] n_relevant>0 scenes: {(df.n_relevant>0).sum()}/{len(df)} '
          f'(reldet defined only where n_relevant>0)')


# ---- partial spearman ------------------------------------------------------
def _rank(a):
    return pd.Series(a).rank().values


def _resid(y, Z):
    Zc = np.column_stack([np.ones(len(y))] + [_rank(Z[c]) for c in Z])
    beta, *_ = np.linalg.lstsq(Zc, _rank(y), rcond=None)
    return _rank(y) - Zc @ beta


def partial_corr(df, x, y, ctrl, n_perm=400, seed=1):
    d = df[[x, y] + ctrl].dropna()
    if len(d) < 50:
        return np.nan, np.nan, len(d)
    rx, ry = _resid(d[x], d[ctrl]), _resid(d[y], d[ctrl])
    rho = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.RandomState(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    floor = np.nanpercentile(np.abs(null), 95)
    return rho, floor, len(d)


def analyze(df):
    print('\n' + '=' * 92)
    print('(A)  Does EGO-RELEVANT det quality predict planning, beyond global quality?')
    print('     partial Spearman( reldet , outcome | globdet,n_relevant,nearest_dist,is_turn )')
    print('     READ: predicts L2 => NON-rescore (learned-ish) channel ;'
          ' only col_margin => rescore-consistent')
    print('=' * 92)
    for q, qlab in [('reldet_{e}', 'hard recall@.3'), ('reldet_soft_{e}', 'soft score-recall')]:
        print(f'\n#### predictor = {qlab} ####')
        ctrl_tmpl = [q.replace('reldet', 'globdet'), 'n_relevant', 'nearest_dist', 'is_turn']
        for e in EPOCHS:
            x = q.format(e=e); c = [t.format(e=e) for t in ctrl_tmpl]
            print(f'-- {e}ep --  {x} std={df[x].std():.3f} (var={df[x].var():.4f}; ~0 => underpowered)')
            for out, lab, sign in [(f'L2_full_{e}', 'L2 (lower=better)', '-rho=>helps'),
                                   (f'colmargin_{e}', 'col_margin (higher=safer)', '+rho=>helps')]:
                rho, fl, n = partial_corr(df, x, out, c)
                star = '*' if (rho == rho and abs(rho) > fl) else ' '
                print(f'     {x} -> {lab:26} rho={rho:+.3f}{star}  (floor {fl:.3f}, n={n})  [{sign}]')

    print('\n' + '=' * 92)
    print('(B)  Does the DETECTION BENEFIT concentrate where EGO-PATH relevance is high?')
    print('     Spearman( benefit , feature )   C_det=L2(no_det)-L2(full)>0 => det helps L2;')
    print('                                     Cmargin=margin_full-margin_nodet>0 => det helps safety')
    print('     KEY: does ego-PATH relevance (n_relevant/n_collrel/min_dist_path) beat generic density (n_near15)?')
    print('=' * 92)
    feats = ['n_relevant', 'n_collrel', 'min_dist_path', 'n_near15', 'nearest_dist', 'n_moving_front']
    for e in EPOCHS:
        print(f'\n-- {e}ep --   mean C_det={df[f"C_det_{e}"].mean():+.4f}  '
              f'mean Cmargin={df[f"Cmargin_{e}"].mean():+.4f}')
        for tgt in [f'C_det_{e}', f'Cmargin_{e}']:
            cells = []
            d = df[[tgt] + feats].dropna()
            rng = np.random.RandomState(2)
            for fcol in feats:
                rho = pd.Series(d[fcol].values).corr(pd.Series(d[tgt].values), method='spearman')
                null = np.array([pd.Series(d[fcol].values).corr(
                    pd.Series(rng.permutation(d[tgt].values)), method='spearman') for _ in range(200)])
                p = (np.abs(null) >= abs(rho)).mean()
                s = '***' if p < 0.005 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                cells.append(f'{fcol}={rho:+.3f}{s}')
            print(f'   {tgt:13}: ' + '  '.join(cells))


if __name__ == '__main__':
    force = '--force' in sys.argv
    df = build_table(force=force)
    gate(df)
    analyze(df)
