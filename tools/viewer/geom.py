"""Scene geometry as JSON for the client-side canvas BEV (VAD-viewer style).

All coords in the LiDAR/ego frame (x=forward, y=left), so the canvas just applies
one world->screen transform. Predicted det/map/motion are model outputs (raw frame);
the EGO planned trajectory uses the eval x-flip (lidar-frame = (-x, y)).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import lib_io
from metrics import scene_l2
from render_scene import load_lidar, to_bev_frame_traj, gt_positions
from collision import build_fut_boxes
from projects.mmdet3d_plugin.datasets.evaluation_nuscenes.planning.planning_eval import PlanningMetric

_PM = PlanningMetric()
DET_THR, MAP_THR, MOT_THR = 0.3, 0.4, 0.3


def _boxes(arr, score=None, thr=0.0):
    """rows [x,y,z,dx,dy,dz,yaw,...] -> [[x,y,yaw,dx,dy,score],...]"""
    out = []
    for k, b in enumerate(arr):
        sc = float(score[k]) if score is not None else 1.0
        if sc < thr:
            continue
        out.append([round(float(b[0]), 2), round(float(b[1]), 2), round(float(b[6]), 3),
                    round(float(b[3]), 2), round(float(b[4]), 2), round(sc, 2)])
    return out


def _coll_steps(plan, info, fb):
    if fb is None:
        return []
    gt = np.cumsum(np.asarray(info['gt_ego_fut_trajs'], float)[:6, :2], axis=0)
    w = [[b] for b in fb]
    gtc = _PM.evaluate_single_coll(torch.tensor(gt, dtype=torch.float32), w)
    pc = _PM.evaluate_single_coll(torch.tensor(np.asarray(plan)[:6, :2], dtype=torch.float32), w)
    return [int(t) for t in np.where((pc & ~gtc).numpy())[0]]


def scene_geom(token, epoch, task, n_lidar=14000):
    va, vb = 'full', 'no_' + task
    infos, t2i = lib_io.load_val_infos()
    i = t2i[token]; info = infos[i]
    A = lib_io.load_slim_cache(epoch, va); B = lib_io.load_slim_cache(epoch, vb)
    planA, planB = A['plan'][i], B['plan'][i]
    fb = build_fut_boxes(infos, i, use_valid_flag=True)
    l2A = scene_l2(planA, info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])[0]
    l2B = scene_l2(planB, info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])[0]
    a3 = lambda x: round(float(np.mean([x[1], x[3], x[5]])), 3)

    pts = load_lidar(info)
    lidar = []
    if pts is not None:
        m = (np.abs(pts[:, 0]) < 60) & (np.abs(pts[:, 1]) < 60)
        pp = pts[m]
        if len(pp) > n_lidar:
            pp = pp[np.random.choice(len(pp), n_lidar, replace=False)]
        lidar = np.round(pp[:, :2], 1).astype(np.float32).flatten().tolist()

    def traj(pos):
        return [[0.0, 0.0]] + [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in pos[:6]]

    cmd = ['LEFT', 'RIGHT', 'STRAIGHT'][int(np.argmax(info['gt_ego_fut_cmd']))]
    return {
        'token': token, 'epoch': epoch, 'task': task,
        'lidar': lidar,
        'gt_boxes': _boxes(np.asarray(info['gt_boxes'])),
        'a_det': _boxes(A['boxes'][i], A['bscore'][i], DET_THR),
        'b_det': _boxes(B['boxes'][i], B['bscore'][i], DET_THR),
        'gt_traj': traj(gt_positions(info)),
        'a_traj': traj(to_bev_frame_traj(planA)),
        'b_traj': traj(to_bev_frame_traj(planB)),
        'colA': _coll_steps(planA, info, fb),
        'colB': _coll_steps(planB, info, fb),
        'meta': {'cmd': cmd, 'n_near15': int((np.linalg.norm(np.asarray(info['gt_boxes'])[:, :2], axis=1) < 15).sum()) if len(info['gt_boxes']) else 0,
                 'n_boxes': len(info['gt_boxes']), 'L2a': a3(l2A), 'L2b': a3(l2B),
                 'va': va, 'vb': vb},
    }


def _vectors(ib, color_by_label=True):
    out = []
    V = ib['vectors']; sc = np.asarray(ib['scores']); lb = np.asarray(ib['labels'])
    for k, v in enumerate(V):
        if sc[k] < MAP_THR:
            continue
        p = np.asarray(v)
        out.append({'pts': np.round(p, 2).tolist(), 'label': int(lb[k])})
    return out


def _motion(ib):
    out = []
    T = np.asarray(ib['trajs_3d']); ts = np.asarray(ib['trajs_score']); ds = np.asarray(ib['scores_3d'])
    for k in range(len(T)):
        if ds[k] < MOT_THR:
            continue
        m = int(np.argmax(ts[k]))
        out.append(np.round(T[k, m], 2).tolist())
    return out


def scene_percep(token, epoch, task):
    """map vectors + motion (predicted) for full and ablated — loads results.pkl (lazy)."""
    va, vb = 'full', 'no_' + task
    infos, t2i = lib_io.load_val_infos(); i = t2i[token]
    ibA = lib_io.load_results(epoch, va)[i]['img_bbox']
    ibB = lib_io.load_results(epoch, vb)[i]['img_bbox']
    return {'a_map': _vectors(ibA), 'b_map': _vectors(ibB),
            'a_motion': _motion(ibA), 'b_motion': _motion(ibB)}
