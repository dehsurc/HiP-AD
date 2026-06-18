"""Panel renderers for the interactive inspector:
  render_cameras(token)                         -> 6-cam montage PNG (left panel)
  render_bev(token, epoch, va, vb, layers,...)  -> LiDAR BEV PNG (right panel)

Layers (comma list): lidar, gt_boxes, gt_traj, a_traj, b_traj,
  a_det, b_det (predicted detection boxes), a_map, b_map (predicted map vectors),
  a_motion, b_motion (predicted agent motion, top mode).
mode: 'overlay' (A+B in one BEV) | 'side' (full | ablated, two BEVs).

Frame: boxes/map/motion are model outputs in the lidar/ego frame -> drawn raw.
Only the EGO planned trajectory uses the eval x-flip (lidar-frame = (-x, y)).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon as MplPoly
import lib_io
from metrics import scene_l2
from render_scene import load_lidar, to_bev_frame_traj, gt_positions, _xy_to_px, _draw_box
from collision import build_fut_boxes
from projects.mmdet3d_plugin.datasets.evaluation_nuscenes.planning.planning_eval import PlanningMetric

_PM = PlanningMetric()
CAM_GRID = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
            ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]
DET_SCORE_THR = 0.3
MAP_SCORE_THR = 0.4
COL = dict(gt='#16a34a', a='#f59e0b', b='#ef4444',
           a_det='#06b6d4', b_det='#db2777', a_map='#38bdf8', b_map='#f472b6',
           a_mot='#a78bfa', b_mot='#fb923c', lidar='#9aa0a6')
MAP_LABEL_C = {0: '#3b82f6', 1: '#f59e0b', 2: '#10b981'}


def _t2i(token):
    infos, t2i = lib_io.load_val_infos()
    return infos, infos[t2i[token]], t2i[token]


def render_cameras(token, out=None):
    _, info, _ = _t2i(token)
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.6))
    for r in range(2):
        for c in range(3):
            ax = axes[r, c]; ax.axis('off'); cam = CAM_GRID[r][c]
            p = info['cams'][cam]['data_path']; p = p[2:] if p.startswith('./') else p
            fp = os.path.join(lib_io.REPO, p)
            if os.path.exists(fp):
                try:
                    ax.imshow(mpimg.imread(fp))
                except Exception:
                    pass
            ax.set_title(cam.replace('CAM_', ''), fontsize=8, pad=2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.04, hspace=0.12)
    return _out(fig, out)


def collide_steps(plan, info, fb):
    if fb is None:
        return np.zeros(6, bool)
    gt_cum = np.cumsum(np.asarray(info['gt_ego_fut_trajs'], float)[:6, :2], axis=0)
    w = [[b] for b in fb]
    gtc = _PM.evaluate_single_coll(torch.tensor(gt_cum, dtype=torch.float32), w)
    pc = _PM.evaluate_single_coll(torch.tensor(np.asarray(plan)[:6, :2], dtype=torch.float32), w)
    return (pc & ~gtc).numpy().astype(bool)


def _draw_traj(ax, pos, s, cx, cy, color, label, ms, lw, cols=None):
    pp = np.vstack([[0, 0], pos])
    px = np.array([_xy_to_px(p[0], p[1], s, cx, cy) for p in pp])
    ax.plot(px[:, 0], px[:, 1], '-o', color=color, lw=lw, ms=ms, label=label, zorder=6)
    if cols is not None:
        for t in np.where(cols)[0]:
            ax.plot(px[t + 1, 0], px[t + 1, 1], marker='*', color='#dc2626',
                    ms=9 * (ms / 3), mec='k', mew=0.5, zorder=8)


def _draw_pred_det(ax, boxes, scores, s, cx, cy, color, msize):
    for b, sc in zip(boxes, scores):
        if sc < DET_SCORE_THR:
            continue
        _draw_box(ax, b, s, cx, cy, color, lw=1.1 * msize)


def _draw_pred_map(ax, vectors, scores, labels, s, cx, cy, msize, single_color=None):
    for v, sc, lb in zip(vectors, scores, labels):
        if sc < MAP_SCORE_THR:
            continue
        p = np.asarray(v)
        px = np.array([_xy_to_px(q[0], q[1], s, cx, cy) for q in p])
        ax.plot(px[:, 0], px[:, 1], '-', lw=1.3 * msize,
                color=single_color or MAP_LABEL_C.get(int(lb), '#888'), alpha=0.9, zorder=4)


def _draw_pred_motion(ax, trajs, tscore, dscore, s, cx, cy, color, msize):
    for k in range(len(trajs)):
        if dscore[k] < DET_SCORE_THR:
            continue
        m = int(np.argmax(tscore[k]))
        p = np.asarray(trajs[k, m])           # (12,2) absolute
        px = np.array([_xy_to_px(q[0], q[1], s, cx, cy) for q in p])
        ax.plot(px[:, 0], px[:, 1], '-', lw=1.0 * msize, color=color, alpha=0.7, zorder=5)


def _bev_axes(ax, info, s, cx, cy, zoom, layers, msize, max_pts=80000):
    ax.set_xlim(0, 800); ax.set_ylim(800, 0); ax.set_aspect('equal'); ax.axis('off')
    if 'lidar' in layers:
        pts = load_lidar(info)
        if pts is not None:
            m = (np.abs(pts[:, 0]) < zoom) & (np.abs(pts[:, 1]) < zoom)
            pp = pts[m]
            if len(pp) > max_pts:
                pp = pp[np.random.choice(len(pp), max_pts, replace=False)]
            col, row = _xy_to_px(pp[:, 0], pp[:, 1], s, cx, cy)
            ax.scatter(col, row, s=0.3 * msize, c=COL['lidar'], alpha=0.45, linewidths=0)
    if 'gt_boxes' in layers:
        for b in info['gt_boxes']:
            _draw_box(ax, b, s, cx, cy, COL['gt'], lw=1.0 * msize)
    ax.plot(cx, cy, marker='^', color='k', ms=8 * msize, zorder=7)


def render_bev(token, epoch, va='full', vb='no_det', layers=None, mode='side',
               zoom=32.0, msize=1.0, out=None):
    if layers is None:
        layers = {'lidar', 'gt_boxes', 'gt_traj', 'a_traj', 'b_traj', 'a_det', 'b_det'}
    layers = set(layers)
    infos, info, i = _t2i(token)
    s = 800 / (2 * zoom); cx = cy = 400.0
    planA = lib_io.load_slim_cache(epoch, va)['plan'][i]
    planB = lib_io.load_slim_cache(epoch, vb)['plan'][i]
    fb = build_fut_boxes(infos, i, use_valid_flag=True)
    gt = gt_positions(info); tA = to_bev_frame_traj(planA); tB = to_bev_frame_traj(planB)
    colA = collide_steps(planA, info, fb); colB = collide_steps(planB, info, fb)
    l2A = scene_l2(planA, info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])[0]
    l2B = scene_l2(planB, info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])[0]
    a3 = lambda x: float(np.mean([x[1], x[3], x[5]]))

    def det(v):
        c = lib_io.load_slim_cache(epoch, v)
        return c['boxes'][i], c['bscore'][i]

    def percep(v):
        ib = lib_io.load_results(epoch, v)[i]['img_bbox']
        return ib

    if mode == 'side':
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7.6))
        for ax, tag, v, traj, col, c_det, c_map, c_mot in [
                (axL, va, va, tA, COL['a'], COL['a_det'], COL['a_map'], COL['a_mot']),
                (axR, vb, vb, tB, COL['b'], COL['b_det'], COL['b_map'], COL['b_mot'])]:
            _bev_axes(ax, info, s, cx, cy, zoom, layers, msize)
            ckey = 'a' if ax is axL else 'b'
            if 'gt_traj' in layers:
                _draw_traj(ax, gt, s, cx, cy, COL['gt'], 'GT', 3 * msize, 1.6 * msize)
            if f'{ckey}_det' in layers:
                b, sc = det(v); _draw_pred_det(ax, b, sc, s, cx, cy, c_det, msize)
            if f'{ckey}_map' in layers:
                ib = percep(v); _draw_pred_map(ax, ib['vectors'], np.asarray(ib['scores']), np.asarray(ib['labels']), s, cx, cy, msize)
            if f'{ckey}_motion' in layers:
                ib = percep(v); _draw_pred_motion(ax, np.asarray(ib['trajs_3d']), np.asarray(ib['trajs_score']), np.asarray(ib['scores_3d']), s, cx, cy, c_mot, msize)
            cols = colA if ckey == 'a' else colB
            tj = tA if ckey == 'a' else tB
            l2 = l2A if ckey == 'a' else l2B
            _draw_traj(ax, tj, s, cx, cy, col, f'{tag} plan', 3 * msize, 2.0 * msize, cols=cols)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_title(f'{tag}   L2={a3(l2):.2f}m   collisions={int(cols.sum())}', fontsize=11)
        fig.suptitle(f'{token[:10]} @{epoch}ep   (★=collision; pred boxes=cyan/pink, map=blue/pink)', fontsize=11)
    else:  # overlay
        fig, ax = plt.subplots(figsize=(11, 9))
        _bev_axes(ax, info, s, cx, cy, zoom, layers, msize)
        if 'gt_traj' in layers:
            _draw_traj(ax, gt, s, cx, cy, COL['gt'], 'GT', 3 * msize, 1.6 * msize)
        if 'a_det' in layers:
            b, sc = det(va); _draw_pred_det(ax, b, sc, s, cx, cy, COL['a_det'], msize)
        if 'b_det' in layers:
            b, sc = det(vb); _draw_pred_det(ax, b, sc, s, cx, cy, COL['b_det'], msize)
        if 'a_map' in layers:
            ib = percep(va); _draw_pred_map(ax, ib['vectors'], np.asarray(ib['scores']), np.asarray(ib['labels']), s, cx, cy, msize, single_color=COL['a_map'])
        if 'b_map' in layers:
            ib = percep(vb); _draw_pred_map(ax, ib['vectors'], np.asarray(ib['scores']), np.asarray(ib['labels']), s, cx, cy, msize, single_color=COL['b_map'])
        if 'a_motion' in layers:
            ib = percep(va); _draw_pred_motion(ax, np.asarray(ib['trajs_3d']), np.asarray(ib['trajs_score']), np.asarray(ib['scores_3d']), s, cx, cy, COL['a_mot'], msize)
        if 'b_motion' in layers:
            ib = percep(vb); _draw_pred_motion(ax, np.asarray(ib['trajs_3d']), np.asarray(ib['trajs_score']), np.asarray(ib['scores_3d']), s, cx, cy, COL['b_mot'], msize)
        if 'a_traj' in layers:
            _draw_traj(ax, tA, s, cx, cy, COL['a'], f'{va} (L2={a3(l2A):.2f})', 3 * msize, 2.0 * msize, cols=colA)
        if 'b_traj' in layers:
            _draw_traj(ax, tB, s, cx, cy, COL['b'], f'{vb} (L2={a3(l2B):.2f})', 3 * msize, 2.0 * msize, cols=colB)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title(f'{token[:10]} @{epoch}ep   overlay   {va} coll={int(colA.sum())}  {vb} coll={int(colB.sum())}', fontsize=11)
    fig.tight_layout()
    return _out(fig, out)


def _out(fig, out):
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=95, bbox_inches='tight'); plt.close(fig); return out
    return fig


if __name__ == '__main__':
    tk = '72fe05b8fd804c7eb2edba7554f1f796'
    o = os.path.join(lib_io.REPO, 'tools/viewer/out')
    render_cameras(tk, os.path.join(o, 'cam_test.png'))
    render_bev(tk, 1, 'full', 'no_det',
               layers={'lidar', 'gt_boxes', 'gt_traj', 'a_traj', 'b_traj', 'a_det', 'b_det', 'a_map', 'b_map'},
               mode='side', out=os.path.join(o, 'bev_side_test.png'))
    print('wrote cam_test.png, bev_side_test.png')
