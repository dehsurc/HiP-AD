"""Per-scene collision (obj_box_col) by reusing the repo's PlanningMetric.

fut_boxes reconstruction replicates nuscenes_3d_dataset.get_ann_info (l.559-587):
future frames' GT boxes transformed into the current ego frame via global poses.

GATE: aggregate per-scene obj_box_col @9ep full must reproduce tg_metrics_raw = 0.082%.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/home/yongjae/e2e/HiP-AD-pcgrad')
import numpy as np, torch
import lib_io
from projects.mmdet3d_plugin.datasets.evaluation_nuscenes.planning.planning_eval import PlanningMetric
from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import get_T_global


def _mask(inf, use_valid_flag):
    if use_valid_flag and 'valid_flag' in inf:
        return inf['valid_flag']
    return inf['num_lidar_pts'] > 0


def build_fut_boxes(infos, idx, use_valid_flag=True):
    info = infos[idx]
    if int(np.asarray(info['gt_ego_fut_masks'])[:6].sum()) < 6:
        return None                                  # not in planning-eval set
    cur_T = get_T_global(info)
    fut = []
    for i in range(1, 7):
        fidx = idx + i
        if fidx >= len(infos):
            return None
        finfo = infos[fidx]
        if finfo['scene_token'] != info['scene_token']:
            return None
        m = _mask(finfo, use_valid_flag)
        fb = np.asarray(finfo['gt_boxes'])[m].astype(np.float64).copy()   # (M,7)
        fT = get_T_global(finfo)
        T = np.linalg.inv(cur_T) @ fT
        if len(fb):
            fb[:, :3] = fb[:, :3] @ T[:3, :3].T + T[:3, 3]
            yaw = np.stack([np.cos(fb[:, 6]), np.sin(fb[:, 6])], -1) @ T[:2, :2].T
            fb[:, 6] = np.arctan2(yaw[..., 1], yaw[..., 0])
        fut.append(torch.tensor(fb, dtype=torch.float32))
    return fut


def scene_obj_box_col(plan, gt_fut, gt_mask, fut_boxes):
    pm = PlanningMetric()
    trajs = torch.tensor(np.asarray(plan)[:6, :2][None], dtype=torch.float32).clone()
    gt = torch.tensor(np.cumsum(np.asarray(gt_fut)[:6, :2], axis=0)[None], dtype=torch.float32).clone()
    mask = torch.tensor(np.asarray(gt_mask)[:6], dtype=torch.float32)[None, :, None].repeat(1, 1, 2)
    fb = [[b] for b in fut_boxes]
    pm.update(trajs, gt, mask, fb)
    return pm.compute()['obj_box_col'].numpy()       # (6,) in {0,1}-ish per step


def _agg(colsum, total):
    value = colsum / total
    cum = np.array([value[:i + 1].mean() for i in range(len(value))])
    return float(np.mean([cum[1], cum[3], cum[5]]) * 100), cum * 100   # percent


def gate(epoch=9, variant='full', expected=0.082, use_valid_flag=True):
    infos, _ = lib_io.load_val_infos()
    plan = lib_io.load_slim_cache(epoch, variant)['plan']
    colsum = np.zeros(6); total = 0
    for i, info in enumerate(infos):
        fb = build_fut_boxes(infos, i, use_valid_flag)
        if fb is None:
            continue
        c = scene_obj_box_col(plan[i], info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'], fb)
        colsum += c; total += 1
    avg, cum = _agg(colsum, total)
    print(f'[col gate] e{epoch}/{variant} valid_flag={use_valid_flag}: avg obj_box_col = {avg:.3f}% '
          f'| expected {expected}% | {"PASS" if abs(avg-expected)<0.01 else "FAIL"} (total={total})')
    print(f'           per-step %: {np.round(cum,3).tolist()}')
    return avg


if __name__ == '__main__':
    gate(9, 'full', 0.082)
