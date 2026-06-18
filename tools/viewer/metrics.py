"""Per-scene planning metrics, ported to exactly replicate planning_eval.py.

L2 correctness GATE: averaging per-scene L2 over the full val set for tg_full @9ep
must reproduce tg_metrics_raw.csv  L2 = 0.6457 (the frame/pairing convention check).
"""
import numpy as np


def scene_l2(pred_plan, gt_fut, gt_mask):
    """pred_plan (6,2) absolute; gt_fut (6,2) per-step displacement; gt_mask (6,).
    Returns (l2_per_step (6,), included:bool). Mirrors planning_eval skip rule
    (skip if any gt mask entry is 0)."""
    gt_mask = np.asarray(gt_mask).reshape(-1)[:6]
    if not np.all(gt_mask > 0):
        return None, False
    gt_pos = np.cumsum(np.asarray(gt_fut, np.float64)[:6, :2], axis=0)
    pred = np.asarray(pred_plan, np.float64)[:6, :2]
    # x-sign flip cancels in L2 (both flipped), so compute directly on positions
    l2 = np.sqrt(((pred - gt_pos) ** 2).sum(axis=-1))   # (6,)
    return l2, True


def aggregate_l2(l2_sum, total):
    """Replicate planning_eval's cumulative-average + {1.0,2.0,3.0s} mean.
    l2_sum: summed per-step L2 over included samples (6,); total: count."""
    value = l2_sum / total                              # per-step mean
    cum = np.array([value[:i + 1].mean() for i in range(len(value))])
    avg = float(np.mean([cum[1], cum[3], cum[5]]))      # 1.0s,2.0s,3.0s
    return avg, cum


def gate(epoch=9, variant='full', expected=0.6457, tol=1e-2):
    import lib_io
    infos, _ = lib_io.load_val_infos()
    slim = lib_io.load_slim_cache(epoch, variant)
    plan = slim['plan']
    l2_sum = np.zeros(6); total = 0; skipped = 0
    for i, info in enumerate(infos):
        l2, inc = scene_l2(plan[i], info['gt_ego_fut_trajs'], info['gt_ego_fut_masks'])
        if not inc:
            skipped += 1; continue
        l2_sum += l2; total += 1
    avg, cum = aggregate_l2(l2_sum, total)
    ok = abs(avg - expected) <= tol
    print(f'[L2 gate] e{epoch}/{variant}: reproduced avg L2 = {avg:.4f} | expected {expected} '
          f'| {"PASS" if ok else "FAIL"} (total={total}, skipped={skipped})')
    print(f'           per-step cumulative L2 (0.5..3.0s): {np.round(cum,4).tolist()}')
    return ok, avg


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import lib_io  # noqa
    gate(9, 'full', 0.6457)
