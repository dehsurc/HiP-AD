"""Viewer IO foundation: data-index + token mapping + slim TG prediction cache.

Conventions verified against planning_eval.py (the correctness anchor):
  * val infos MUST be sorted by timestamp (raw pkl is NOT pre-sorted).
  * results.pkl[i]  <->  sorted_val_infos[i]  (positional, shuffle=False at test).
  * GT ego trajectory = info['gt_ego_fut_trajs'] (6,2) per-step displacements -> cumsum = positions.
  * pred ego trajectory = results[i]['img_bbox']['plan_temp_2hz'] (6,2) already absolute positions.
"""
import os, pickle
import numpy as np

REPO = '/home/yongjae/e2e/HiP-AD-pcgrad'
VAL_PKL = os.path.join(REPO, 'data/infos/nuscenes_infos_val.pkl')
CACHE_DIR = os.path.join(REPO, 'tools/viewer/cache')
os.makedirs(CACHE_DIR, exist_ok=True)

TG_EPOCH_DIR = {
    9: os.path.join(REPO, 'work_dirs/exp/stage2_tg_9ep/eval'),
    3: os.path.join(REPO, 'work_dirs/exp/stage2_tg_3ep/eval'),
    1: os.path.join(REPO, 'work_dirs/exp/stage2_tg_1ep/eval'),
}
VARIANT_ART = {  # variant -> artifacts subdir name
    'full': 'artifacts_tg_full',
    'no_det': 'artifacts_tg_no_det',
    'no_map': 'artifacts_tg_no_map',
    'no_motion': 'artifacts_tg_no_motion',
    'no_det_motion': 'artifacts_tg_no_det_motion',
    'pretrain_ref': 'artifacts_pretrain_ref',
}


_INFOS_CACHE = None
_SLIM_CACHE = {}


def load_val_infos():
    """Return (infos_sorted_by_timestamp, token2idx). Memoized for server reuse."""
    global _INFOS_CACHE
    if _INFOS_CACHE is not None:
        return _INFOS_CACHE
    with open(VAL_PKL, 'rb') as f:
        d = pickle.load(f)
    infos = d['infos'] if isinstance(d, dict) else d
    infos = sorted(infos, key=lambda e: e['timestamp'])
    token2idx = {e['token']: i for i, e in enumerate(infos)}
    assert len(infos) == 6019, f'expected 6019 val infos, got {len(infos)}'
    _INFOS_CACHE = (infos, token2idx)
    return _INFOS_CACHE


def _results_pkl(epoch, variant):
    return os.path.join(TG_EPOCH_DIR[epoch], VARIANT_ART[variant], 'results.pkl')


def build_slim_cache(epoch, variant, force=False):
    """Load the ~1.9GB results.pkl once and persist a slim per-index npz
    holding only what the viewer needs (plan_temp_2hz, boxes, scores, labels,
    map vectors). Returns the cache path."""
    out = os.path.join(CACHE_DIR, f'slim_e{epoch}_{variant}.npz')
    if os.path.exists(out) and not force:
        return out
    import mmcv
    src = _results_pkl(epoch, variant)
    print(f'[build_slim_cache] loading {src} ...', flush=True)
    res = mmcv.load(src)
    n = len(res)
    plan = np.zeros((n, 6, 2), np.float32)
    # boxes_3d: (300,10) -> keep [x,y,z,w,l,h,yaw] (first 7) + score + label, top-K by score
    boxes = np.full((n, 300, 7), np.nan, np.float32)
    bscore = np.zeros((n, 300), np.float32)
    blabel = np.full((n, 300), -1, np.int64)
    for i in range(n):
        ib = res[i]['img_bbox']
        plan[i] = np.asarray(ib['plan_temp_2hz'])
        b = np.asarray(ib['boxes_3d'])           # (300,10)
        boxes[i] = b[:, :7]
        bscore[i] = np.asarray(ib['scores_3d'])
        blabel[i] = np.asarray(ib['labels_3d'])
    np.savez_compressed(out, plan=plan, boxes=boxes, bscore=bscore, blabel=blabel)
    print(f'[build_slim_cache] wrote {out}  (n={n})', flush=True)
    return out


def load_slim_cache(epoch, variant):
    key = (epoch, variant)
    if key in _SLIM_CACHE:
        return _SLIM_CACHE[key]
    out = os.path.join(CACHE_DIR, f'slim_e{epoch}_{variant}.npz')
    if not os.path.exists(out):
        build_slim_cache(epoch, variant)
    data = dict(np.load(out))
    _SLIM_CACHE[key] = data
    return data


_RESULTS_CACHE = {}
_RESULTS_ORDER = []


def load_results(epoch, variant, max_keep=4):
    """Lazily hold a full results.pkl (predicted det/map/motion) in memory.
    Used only when the viewer needs predicted map vectors / motion trajectories
    (predicted det boxes already live in the slim cache)."""
    key = (epoch, variant)
    if key in _RESULTS_CACHE:
        return _RESULTS_CACHE[key]
    import mmcv
    res = mmcv.load(_results_pkl(epoch, variant))
    _RESULTS_CACHE[key] = res
    _RESULTS_ORDER.append(key)
    while len(_RESULTS_ORDER) > max_keep:
        old = _RESULTS_ORDER.pop(0)
        _RESULTS_CACHE.pop(old, None)
    return res
