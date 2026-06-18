"""Mode-A (TG, val split) per-scene renderer: LiDAR top-down BEV with GT boxes and
the BEFORE (variant A) vs AFTER (variant B) planned ego trajectory + GT, plus a
zoomed trajectory-diff panel and a per-scene metric strip.

Frame convention (verified via the L2 gate):
  * lidar/box frame: x=forward, y=left (nuScenes LIDAR_TOP).
  * planner output plan_temp_2hz is compared after x-sign flip in planning_eval; so the
    lidar-frame trajectory is (-plan_x, plan_y). GT positions = cumsum(gt_ego_fut_trajs).
  * BEV image: forward(x)->up, left(y)->left.  col = cx - y*s ; row = cy - x*s.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
import lib_io
from metrics import scene_l2

REPO = lib_io.REPO
BEV_RANGE = 51.2          # meters half-extent for the full BEV
IMG = 800                 # px


def load_lidar(info):
    p = info['lidar_path']
    p = p[2:] if p.startswith('./') else p
    path = os.path.join(REPO, p)
    if not os.path.exists(path):
        return None
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3]
    return pts


def to_bev_frame_traj(plan):
    """planner (6,2) -> lidar-frame positions (6,2) via x-sign flip."""
    plan = np.asarray(plan, np.float64)[:6, :2].copy()
    plan[:, 0] = -plan[:, 0]
    return plan


def gt_positions(info):
    g = np.cumsum(np.asarray(info['gt_ego_fut_trajs'], np.float64)[:6, :2], axis=0)
    g[:, 0] = -g[:, 0]
    return g


def _xy_to_px(x, y, s, cx, cy):
    return cx - y * s, cy - x * s            # (col, row)


def _draw_box(ax, box, s, cx, cy, color, lw=1.2):
    # box: [x,y,z,dx,dy,dz,yaw]; rectangle dx along heading, dy perp
    x, y, _, dx, dy, _, yaw = box[:7]
    c, sn = np.cos(yaw), np.sin(yaw)
    corners = np.array([[ dx/2,  dy/2], [ dx/2, -dy/2], [-dx/2, -dy/2], [-dx/2,  dy/2]])
    R = np.array([[c, -sn], [sn, c]])
    pts = (corners @ R.T) + np.array([x, y])
    px = [_xy_to_px(p[0], p[1], s, cx, cy) for p in pts]
    ax.add_patch(MplPoly(px, closed=True, fill=False, edgecolor=color, linewidth=lw))


def _draw_traj(ax, pos, s, cx, cy, color, label, lw=2.4, ms=5):
    pos = np.vstack([[0, 0], pos])            # start at ego origin
    px = np.array([_xy_to_px(p[0], p[1], s, cx, cy) for p in pos])
    ax.plot(px[:, 0], px[:, 1], '-o', color=color, lw=lw, ms=ms, label=label)


def render(token, epoch, var_a, var_b, out_png, max_pts=120000):
    infos, token2idx = lib_io.load_val_infos()
    i = token2idx[token]
    info = infos[i]
    A = lib_io.load_slim_cache(epoch, var_a)
    B = lib_io.load_slim_cache(epoch, var_b)
    planA, planB = A['plan'][i], B['plan'][i]
    gt = gt_positions(info)
    tA, tB = to_bev_frame_traj(planA), to_bev_frame_traj(planB)
    l2A, _ = scene_l2(planA, info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])
    l2B, _ = scene_l2(planB, info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)
    s = IMG / (2 * BEV_RANGE)
    cx = cy = IMG / 2

    # --- big BEV: lidar + GT boxes + 3 trajectories ---
    ax = fig.add_subplot(gs[:, :2]); ax.set_xlim(0, IMG); ax.set_ylim(IMG, 0); ax.set_aspect('equal'); ax.axis('off')
    pts = load_lidar(info)
    if pts is not None:
        m = (np.abs(pts[:, 0]) < BEV_RANGE) & (np.abs(pts[:, 1]) < BEV_RANGE)
        pp = pts[m]
        if len(pp) > max_pts:
            pp = pp[np.random.choice(len(pp), max_pts, replace=False)]
        col, row = _xy_to_px(pp[:, 0], pp[:, 1], s, cx, cy)
        ax.scatter(col, row, s=0.4, c='#9aa0a6', alpha=0.5, linewidths=0)
    for b in info['gt_boxes']:
        _draw_box(ax, b, s, cx, cy, '#22c55e', lw=1.0)
    ax.plot(cx, cy, marker='^', color='k', ms=12)             # ego
    _draw_traj(ax, gt, s, cx, cy, '#16a34a', f'GT')
    _draw_traj(ax, tA, s, cx, cy, '#f59e0b', f'{var_a} (L2={np.mean([l2A[1],l2A[3],l2A[5]]):.2f})')
    _draw_traj(ax, tB, s, cx, cy, '#ef4444', f'{var_b} (L2={np.mean([l2B[1],l2B[3],l2B[5]]):.2f})')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_title(f'LiDAR BEV  (forward=up)   scene {token[:10]}  scene_token {info["scene_token"][:8]}', fontsize=11)

    # --- zoomed trajectory diff (forward ~40m) ---
    Z = 40; sz = IMG / (2 * Z)
    axz = fig.add_subplot(gs[0, 2]); axz.set_xlim(0, IMG); axz.set_ylim(IMG, 0); axz.set_aspect('equal'); axz.axis('off')
    _draw_traj(axz, gt, sz, cx, cy, '#16a34a', 'GT')
    _draw_traj(axz, tA, sz, cx, cy, '#f59e0b', var_a)
    _draw_traj(axz, tB, sz, cx, cy, '#ef4444', var_b)
    axz.plot(cx, cy, marker='^', color='k', ms=10)
    axz.set_title(f'planned traj (zoom {Z}m)', fontsize=10); axz.legend(fontsize=9, loc='lower left')

    # --- metric strip ---
    axm = fig.add_subplot(gs[1, 2]); axm.axis('off')
    cmd = ['LEFT', 'RIGHT', 'STRAIGHT'][int(np.argmax(info['gt_ego_fut_cmd']))]
    dl2 = float(np.mean([l2A[1], l2A[3], l2A[5]]) - np.mean([l2B[1], l2B[3], l2B[5]]))
    steps = ['1.0s', '2.0s', '3.0s']; idx = [1, 3, 5]
    txt = [f'epoch {epoch}   cmd={cmd}   #GT boxes={len(info["gt_boxes"])}',
           f'A = {var_a}   B = {var_b}', '',
           f'{"step":>6} {"A_L2":>8} {"B_L2":>8} {"A-B":>8}']
    for st, k in zip(steps, idx):
        txt.append(f'{st:>6} {l2A[k]:8.3f} {l2B[k]:8.3f} {l2A[k]-l2B[k]:8.3f}')
    txt += ['', f'avg dL2 (A-B) = {dl2:+.3f}   ( >0 means B better )']
    axm.text(0.02, 0.98, '\n'.join(txt), va='top', ha='left', family='monospace', fontsize=12)
    fig.suptitle(f'TG before/after  —  {var_a}  vs  {var_b}  @ epoch {epoch}', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110); plt.close(fig)
    return dl2


def top_dl2_tokens(epoch, var_a, var_b, k=6, sign=+1):
    """Return tokens where (L2_a - L2_b)*sign is largest (sign=+1 -> B improves most)."""
    infos, _ = lib_io.load_val_infos()
    A = lib_io.load_slim_cache(epoch, var_a)['plan']
    B = lib_io.load_slim_cache(epoch, var_b)['plan']
    rows = []
    for i, info in enumerate(infos):
        la, inc = scene_l2(A[i], info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])
        if not inc:
            continue
        lb, _ = scene_l2(B[i], info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])
        da = np.mean([la[1], la[3], la[5]]); db = np.mean([lb[1], lb[3], lb[5]])
        rows.append((info['token'], (da - db) * sign, da, db))
    rows.sort(key=lambda r: -r[1])
    return rows[:k]


if __name__ == '__main__':
    EP, A, B = 9, 'full', 'no_map'
    out = os.path.join(REPO, 'tools/viewer/out')
    top = top_dl2_tokens(EP, A, B, k=3, sign=+1)
    print('top scenes where no_map improves planning most (dL2 = L2_full - L2_no_map):')
    for tk, d, da, db in top:
        print(f'  {tk[:12]}  dL2={d:+.3f}  full={da:.3f} no_map={db:.3f}')
    for r, (tk, d, da, db) in enumerate(top):
        p = os.path.join(out, f'tg_e{EP}_{A}_vs_{B}_top{r}_{tk[:8]}.png')
        render(tk, EP, A, B, p)
        print('wrote', p)
