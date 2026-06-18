"""MAP arm of the ego-relevance observational screen (the cleaner test: map has
NO rescore shortcut, so rel_map_quality -> planning would be a LEARNED channel).

Per scene:
  rel_map_recall  = of GT lane polylines within R_REL of the ego GT future path,
                    fraction matched by a predicted vector (same class, score>thr)
                    within Chamfer distance MATCH_CD.
  glob_map_recall = same over ALL GT polylines (the control for overall map quality).
Then partial Spearman( rel_map , planning | glob_map, n_relmap, nearest..., is_turn ),
reusing the det-arm table (L2 / col_margin / controls) from relevance_table.csv.

READ: rel_map predicts L2/col_margin beyond glob_map  => planning uses WHICH lanes
are right near the ego path (learned channel) -> relevance-weighting has fuel.
Null with good variance => planner barely uses map quality (open-loop) -> closed-loop.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import lib_io
from relevance_analysis import ego_path, min_dist_to_path, partial_corr, R_REL, CACHE

EPOCHS = [1, 3, 9]
MAP_THR = 0.4
MATCH_CD = 1.5          # Chamfer dist (m) for a pred vector to "cover" a GT polyline
OUT = os.path.join(lib_io.CACHE_DIR, 'relevance_map_table.csv')


def chamfer(A, B):
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return 0.5 * (D.min(axis=1).mean() + D.min(axis=0).mean())


def gt_polys(info):
    """list of (class, pts(N,2)) GT map polylines."""
    out = []
    for cls, lst in info['map_annos'].items():
        for p in lst:
            p = np.asarray(p, float)[:, :2]
            if len(p) >= 2:
                out.append((int(cls), p))
    return out


def pred_polys(ib):
    """list of (label, pts(20,2)) predicted vectors above MAP_THR."""
    V = np.asarray(ib['vectors']); sc = np.asarray(ib['scores']); lb = np.asarray(ib['labels'])
    keep = sc > MAP_THR
    return [(int(lb[k]), np.asarray(V[k], float)) for k in np.where(keep)[0]]


def map_recall(gt_list, pred_list, d=MATCH_CD):
    if not gt_list:
        return np.nan
    if not pred_list:
        return 0.0
    hit = 0
    for cls, P in gt_list:
        cands = [Q for (lb, Q) in pred_list if lb == cls]
        if cands and min(chamfer(P, Q) for Q in cands) < d:
            hit += 1
    return hit / len(gt_list)


def build():
    if os.path.exists(OUT):
        return pd.read_csv(OUT)
    base = pd.read_csv(CACHE)                       # det-arm table (token-indexed)
    infos, t2i = lib_io.load_val_infos()
    keep_tokens = set(base.token)
    # precompute relevant/global GT polylines per token (geometry only, epoch-free)
    rel_gt, glob_gt = {}, {}
    for tok in base.token:
        info = infos[t2i[tok]]
        path = ego_path(info)
        gp = gt_polys(info)
        glob_gt[tok] = gp
        rel = [(c, P) for (c, P) in gp if min_dist_to_path(P, path).min() < R_REL]
        rel_gt[tok] = rel
    cols = {tok: {} for tok in base.token}
    for e in EPOCHS:
        print(f'[map arm] loading full results.pkl @ {e}ep ...', flush=True)
        res = lib_io.load_results(e, 'full')
        for tok in base.token:
            ib = res[t2i[tok]]['img_bbox']
            pp = pred_polys(ib)
            cols[tok][f'relmap_{e}'] = map_recall(rel_gt[tok], pp)
            cols[tok][f'globmap_{e}'] = map_recall(glob_gt[tok], pp)
            cols[tok][f'n_relmap'] = len(rel_gt[tok])
            cols[tok][f'n_globmap'] = len(glob_gt[tok])
        print(f'[map arm] {e}ep done', flush=True)
    add = pd.DataFrame([{'token': tok, **cols[tok]} for tok in base.token])
    df = base.merge(add, on='token')
    df.to_csv(OUT, index=False)
    print(f'[built {OUT}]  n={len(df)}')
    return df


def analyze(df):
    print('\n' + '=' * 92)
    print('MAP ARM — does ego-relevant map quality predict planning, beyond global map quality?')
    print('  partial Spearman( relmap , outcome | globmap, n_relmap, nearest_dist, is_turn )')
    print('  map has NO rescore shortcut: relmap->L2 or ->col_margin => LEARNED channel.')
    print('=' * 92)
    for e in EPOCHS:
        x = f'relmap_{e}'; c = [f'globmap_{e}', 'n_relmap', 'nearest_dist', 'is_turn']
        print(f'-- {e}ep --  relmap mean={df[x].mean():.3f} std={df[x].std():.3f} '
              f'(var={df[x].var():.4f}; ~0 => underpowered)  n_relmap mean={df.n_relmap.mean():.1f}')
        for out, lab, sign in [(f'L2_full_{e}', 'L2 (lower=better)', '-rho=>helps'),
                               (f'colmargin_{e}', 'col_margin (higher=safer)', '+rho=>helps')]:
            rho, fl, n = partial_corr(df, x, out, c)
            star = '*' if (rho == rho and abs(rho) > fl) else ' '
            print(f'     relmap -> {lab:26} rho={rho:+.3f}{star}  (floor {fl:.3f}, n={n})  [{sign}]')
    # side-by-side reminder of det arm
    print('\n  (det arm for comparison: reldet->L2 rho ~ -0.08..-0.11*, ->col_margin ~ +0.07..+0.10*)')


if __name__ == '__main__':
    df = build()
    analyze(df)
