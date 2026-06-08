"""Evaluate the MapTRv2 *teacher* map mAP from its cached predictions.

Why this exists
---------------
The feature-KD experiments (E9/E10) distill a MapTRv2 teacher into the HiP-AD
map branch but barely beat GT-only training. Before spending ~14h on another
KD run we want to know the *teacher ceiling*: if the teacher itself is not much
better than the student, there is simply nothing to transfer.

What it does
------------
Reuses the exact student map evaluator (``VectorEvaluate`` ->
ped_crossing/divider/boundary AP + mAP_normal) but feeds it the teacher's
cached polylines instead of student predictions.

Caveats (read before trusting the number)
-----------------------------------------
* The only teacher cache available is over the **train** split, and the teacher
  was trained on it -> this number is an *optimistic* upper bound. The student
  numbers we compare against (~0.34 mAP_normal) are on **val**. So this is a
  ceiling, not an apples-to-apples comparison.
* If even this optimistic train ceiling is close to the student's val mAP,
  there is no headroom and KD cannot help. If it is much higher, KD *might*
  help (necessary, not sufficient).

Usage
-----
    python tools/eval_teacher_map.py \
        projects/configs/experiments/E10_feature_distill_antiabsorb.py \
        --max-samples 3000          # quick estimate; omit for full train set
"""
import argparse
import contextlib
import gc
import os
import os.path as osp
from copy import deepcopy

import mmcv
import numpy as np
from mmcv import Config

# Register NuScenes3DDataset / VectorizeMap / etc. into the mm registries.
import projects.mmdet3d_plugin  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate teacher map mAP from cache")
    p.add_argument("config", help="experiment config (for eval_config + class perm)")
    p.add_argument(
        "--cache",
        default="data/cache/map/maptrv2_teacher_train.pkl",
        help="teacher map cache: {token: {logits[100,3], pts[100,20,2], scores[100]}}",
    )
    p.add_argument(
        "--ann",
        default="data/infos/nuscenes_infos_train.pkl",
        help="annotation file whose GT the teacher is scored against (must match the cache split)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="if >0, uniformly subsample the GT set to ~this many frames (sets load_interval)",
    )
    p.add_argument(
        "--score-thr",
        type=float,
        default=0.0,
        help="drop teacher instances below this score (0.0 = keep all, true AP)",
    )
    p.add_argument(
        "--out",
        default="work_dirs/teacher_map_eval/teacher_submission_vector.json",
        help="where to write/read the submission json",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="skip cache conversion and evaluate the existing --out submission json",
    )
    p.add_argument(
        "--label-remap",
        choices=("none", "config"),
        default="none",
        help=(
            "how to map recovered cache labels into eval labels during conversion. "
            "'none' keeps cache labels as-is; 'config' applies the config's "
            "map_teacher_to_student_class_perm. The current top20_l345 cache "
            "evaluates correctly with 'none'."
        ),
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--gt-workers",
        type=int,
        default=None,
        help="workers for GT dataloader; defaults to --workers",
    )
    p.add_argument(
        "--ap-workers",
        type=int,
        default=None,
        help="workers for AP matching; defaults to --workers",
    )
    p.add_argument(
        "--quiet-gts",
        action="store_true",
        help="suppress verbose GT collection progress when GT is pre-collected",
    )
    return p.parse_args()


def _use_map_only_pipeline(eval_cfg):
    """VectorEvaluate only needs token metadata and vectorized map GT."""
    for step in eval_cfg.get("pipeline", []):
        if step.get("type") == "Collect":
            step["keys"] = ["vectors"]
            step["meta_keys"] = ["token"]


def build_eval_cfg(cfg, ann, max_samples):
    """Clone the experiment's eval_config and point it at the teacher's split."""
    eval_cfg = deepcopy(cfg.eval_config)
    eval_cfg["ann_file"] = ann
    eval_cfg["test_mode"] = True
    _use_map_only_pipeline(eval_cfg)
    if max_samples and max_samples > 0:
        meta = mmcv.load(ann, file_format="pkl")
        n = len(meta["infos"])
        step = max(1, round(n / max_samples))
        infos = []
        for info in meta["infos"][::step]:
            info = deepcopy(info)
            # Map evaluation does not need planning future fields. Removing
            # them avoids index+i future-frame lookups after subsampling.
            for key in (
                "gt_ego_fut_trajs",
                "gt_ego_fut_masks",
                "gt_ego_fut_cmd",
                "gt_ego_fut_trajs_2hz",
                "gt_ego_fut_masks_2hz",
            ):
                info.pop(key, None)
            infos.append(info)
        subset_ann = osp.join(
            "work_dirs",
            "teacher_map_eval",
            f"subset_{osp.splitext(osp.basename(ann))[0]}_step{step}.pkl",
        )
        os.makedirs(osp.dirname(subset_ann), exist_ok=True)
        mmcv.dump({"infos": infos, "metadata": meta["metadata"]}, subset_ann)
        eval_cfg["ann_file"] = subset_ann
        eval_cfg["load_interval"] = 1
        print(
            f"[subsample] {n} frames -> step={step}, subset={len(infos)} frames "
            f"({subset_ann})"
        )
    # NOTE: return the ConfigDict (a dict subclass) directly. build_dataset()
    # checks isinstance(cfg, dict); a Config object is NOT a dict and fails.
    return eval_cfg


def teacher_to_eval_perm(cfg):
    """Columns that reorder teacher logits into the eval map-class order.

    eval order  = cfg.map_class_names      (e.g. ['ped_crossing','divider','boundary'])
    teacher cols= cfg.map_teacher_class_names (e.g. ['divider','ped_crossing','boundary'])
    student_logits[:, i] = teacher_logits[:, perm[i]]
    """
    perm = cfg.get("map_teacher_to_student_class_perm", None)
    if perm is None:
        eval_names = cfg.map_class_names
        teacher_names = cfg.map_teacher_class_names
        perm = tuple(teacher_names.index(n) for n in eval_names)
    return np.asarray(perm, dtype=np.int64)


def build_submission(cache, teacher_label_to_eval, score_thr):
    """Convert the teacher cache into the {token: {vectors, scores, labels}} format.

    MapTRv2 inference selects the top-100 over the *flattened* (query x class)
    sigmoid scores, so each cached entry is a (query, class) pick: ``scores[i]``
    is the sigmoid of one specific class (NOT necessarily the query's argmax),
    and they come pre-sorted descending. We therefore:
      * take ``scores[i]`` as the instance score (the teacher's own confidence), and
      * recover the entry's label as the class whose sigmoid matches that score,
    then map cache class ids into the eval class order via ``teacher_label_to_eval``.
    """
    results = {}
    n_eval_cls = len(teacher_label_to_eval)
    per_cls = np.zeros(n_eval_cls, dtype=np.int64)
    n_kept = 0
    max_label_score_gap = 0.0
    for token, entry in cache.items():
        logits = np.asarray(entry["logits"], dtype=np.float32)   # [100, 3] teacher order
        pts = np.asarray(entry["pts"], dtype=np.float32)         # [100, 20, 2]
        scores = np.asarray(entry["scores"], dtype=np.float32)   # [100], sorted desc

        sig = 1.0 / (1.0 + np.exp(-logits))                      # [100, 3]
        # recover which class each cached score belongs to
        lbl_teacher = np.abs(sig - scores[:, None]).argmin(axis=1)
        max_label_score_gap = max(
            max_label_score_gap,
            float(np.abs(sig[np.arange(len(lbl_teacher)), lbl_teacher] - scores).max()),
        )
        lbl_eval = teacher_label_to_eval[lbl_teacher]

        keep = scores >= score_thr
        vectors, out_scores, out_labels = [], [], []
        for i in np.nonzero(keep)[0]:
            v = pts[i]
            if v.shape[0] < 2:
                continue
            vectors.append(v.tolist())
            out_scores.append(float(scores[i]))
            out_labels.append(int(lbl_eval[i]))
            per_cls[lbl_eval[i]] += 1
            n_kept += 1
        results[token] = {"vectors": vectors, "scores": out_scores, "labels": out_labels}

    print(f"[submission] {len(results)} tokens, {n_kept} kept instances, "
          f"per-class counts (eval order) {per_cls.tolist()}")
    print(f"[sanity] max |sigmoid[label]-score| over all entries = {max_label_score_gap:.2e} "
          f"(should be ~0 -> label recovery exact)")
    return {"results": results}


class QuietProgressBar:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    eval_cfg = build_eval_cfg(cfg, args.ann, args.max_samples)
    logit_perm = teacher_to_eval_perm(cfg)
    if args.label_remap == "config":
        label_to_eval = np.argsort(logit_perm)  # teacher/cache label id -> eval label id
    else:
        label_to_eval = np.arange(len(logit_perm), dtype=np.int64)
    print(
        f"[classes] eval order = {list(eval_cfg.get('map_classes'))}, "
        f"config logit perm = {logit_perm.tolist()}, "
        f"cache-label->eval = {label_to_eval.tolist()} ({args.label_remap})"
    )

    if not args.eval_only:
        print(f"[cache] loading {args.cache} ...")
        cache = mmcv.load(args.cache, file_format="pkl")
        print(f"[cache] {len(cache)} teacher samples")

        submission = build_submission(cache, label_to_eval, args.score_thr)
        os.makedirs(osp.dirname(args.out), exist_ok=True)
        mmcv.dump(submission, args.out)
        print(f"[submission] written to {args.out}")
        del cache, submission
        gc.collect()
    else:
        print(f"[submission] eval-only: reading existing {args.out}")

    # Reuse the student's exact map evaluator.
    from projects.mmdet3d_plugin.datasets.evaluation_nuscenes.map.vector_eval import (
        VectorEvaluate,
    )
    import projects.mmdet3d_plugin.datasets.evaluation_nuscenes.map.vector_eval as vector_eval_module

    gt_workers = args.workers if args.gt_workers is None else args.gt_workers
    ap_workers = args.workers if args.ap_workers is None else args.ap_workers
    print(
        "[eval] building GT dataset and scoring teacher "
        f"(gt_workers={gt_workers}, ap_workers={ap_workers}) ..."
    )
    evaluator = VectorEvaluate(eval_cfg, n_workers=gt_workers)
    if ap_workers != gt_workers:
        # Materialize GTs before switching n_workers so dataloader workers and
        # AP workers can be controlled independently.
        if args.quiet_gts:
            print("[eval] collecting GTs quietly ...")
            old_progress_bar = vector_eval_module.mmcv.ProgressBar
            vector_eval_module.mmcv.ProgressBar = QuietProgressBar
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                _ = evaluator.gts
            vector_eval_module.mmcv.ProgressBar = old_progress_bar
            print("[eval] GT collection done")
        else:
            _ = evaluator.gts
        evaluator.n_workers = ap_workers
    result = evaluator.evaluate(args.out, logger=None)

    print("\n================ TEACHER MAP CEILING ================")
    for k in ("ped_crossing", "divider", "boundary", "mAP_normal"):
        if k in result:
            print(f"  {k:14s}= {result[k]:.4f}")
    print("=====================================================")
    print("NOTE: this is the teacher's score on TRAIN (optimistic). "
          "Compare against the student's VAL mAP_normal (~0.34).")


if __name__ == "__main__":
    main()
