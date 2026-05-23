"""Unit tests for ``tools.gradient_analysis.planning_importance``.

All tests use toy CSVs / in-memory frames so the HiP-AD model env is not required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.gradient_analysis.planning_importance import (
    _detect_inter_gnn,
    _normalized_entropy,
    build_alignment_transfer_agreement,
    build_importance_summary,
    build_planning_layer_sensitivity,
    build_scene_winner,
    load_plan_alignment,
    load_plan_transfer,
    load_task_collinearity,
    run_planning_importance,
)


# --------------------------------------------------------------------- helpers


def _conflict_per_batch(
    groups, batch_idx, cos_vals, norm_a=1.0, norm_b=1.0,
) -> pd.DataFrame:
    n = len(cos_vals)
    return pd.DataFrame({
        "batch_idx": list(batch_idx),
        "group": list(groups),
        "cos": list(cos_vals),
        "coop_mag": [0.0] * n,
        "conf_mag": [0.0] * n,
        "norm_a": [norm_a] * n if np.isscalar(norm_a) else list(norm_a),
        "norm_b": [norm_b] * n if np.isscalar(norm_b) else list(norm_b),
    })


def _write_pair(dir_: Path, a: str, b: str, df: pd.DataFrame) -> None:
    df.to_csv(dir_ / f"conflict_{a}_{b}_per_batch.csv", index=False)


# --------------------------------------------------------------------- helpers


def test_detect_inter_gnn_picks_up_keys():
    keys = ["dec0_inter_gnn_0", "dec5_inter_gnn_0", "dec3_ffn_0", "neck",
            "dec1_norm_0", None, ""]
    assert _detect_inter_gnn(keys) == ["dec0_inter_gnn_0", "dec5_inter_gnn_0"]


def test_normalized_entropy_extremes():
    # one bucket always wins → 0
    assert _normalized_entropy(np.array([10.0, 0.0, 0.0])) == pytest.approx(0.0)
    # uniform across 3 → 1
    assert _normalized_entropy(np.array([3.0, 3.0, 3.0])) == pytest.approx(1.0, abs=1e-9)
    # two-of-three uniform → log2(2)/log2(3) ≈ 0.6309
    val = _normalized_entropy(np.array([5.0, 5.0, 0.0]))
    assert val == pytest.approx(np.log2(2) / np.log2(3), rel=1e-6)
    # empty / zero sum → nan
    assert np.isnan(_normalized_entropy(np.array([0.0, 0.0, 0.0])))


# --------------------------------------------------------------------- loaders


def test_load_plan_alignment_maps_filename_to_aux(tmp_path):
    g = ["dec0_inter_gnn_0", "dec1_inter_gnn_0", "neck", "dec0_inter_gnn_0"]
    _write_pair(tmp_path, "det", "plan", _conflict_per_batch(g, [0, 0, 0, 1], [0.5, 0.3, 0.9, -0.4]))
    _write_pair(tmp_path, "map", "plan", _conflict_per_batch(g, [0, 0, 0, 1], [-0.1, 0.0, 0.7, 0.2]))
    out = load_plan_alignment(tmp_path, aux_tasks=("det", "map"))
    # filter to inter_gnn auto-detected → neck row dropped
    assert set(out["group"].unique()) == {"dec0_inter_gnn_0", "dec1_inter_gnn_0"}
    # each aux present
    assert set(out["aux_task"].unique()) == {"det", "map"}
    # det at dec0 batch=0 → 0.5
    row = out[(out["aux_task"] == "det")
              & (out["batch_idx"] == 0)
              & (out["group"] == "dec0_inter_gnn_0")]
    assert row["cos"].iloc[0] == pytest.approx(0.5)


def test_load_plan_alignment_drops_pseudo_shared(tmp_path):
    g = ["dec0_inter_gnn_0"] * 2
    df = _conflict_per_batch(g, [0, 1], [0.5, -0.4],
                             norm_a=[1.0, 1e-12], norm_b=[1.0, 1.0])
    _write_pair(tmp_path, "det", "plan", df)
    out = load_plan_alignment(tmp_path, aux_tasks=("det",))
    # batch 1 had norm_a < EPS → cos forced to NaN
    cos_by_batch = out.set_index("batch_idx")["cos"]
    assert cos_by_batch.loc[0] == pytest.approx(0.5)
    assert np.isnan(cos_by_batch.loc[1])


def test_load_plan_alignment_handles_reversed_filename(tmp_path):
    g = ["dec0_inter_gnn_0"]
    # filename has plan first, motion second — the loader must try both orderings.
    _write_pair(tmp_path, "plan", "motion", _conflict_per_batch(g, [0], [0.2]))
    out = load_plan_alignment(tmp_path, aux_tasks=("motion",))
    assert len(out) == 1
    assert out["aux_task"].iloc[0] == "motion"
    assert out["cos"].iloc[0] == pytest.approx(0.2)


def test_load_plan_transfer_filters_and_signs(tmp_path):
    rows = []
    for src in ("det", "map", "motion"):
        for batch in (0, 1):
            rows.append({
                "batch_idx": batch, "source_task": src, "target_task": "plan",
                "steps": 1, "variant": "normalized",
                "layer": "dec0_inter_gnn_0",
                "grad_norm": 1.0, "grad_dot": 0.0, "step_size": 1e-3,
                "baseline_loss": 1.0, "stepped_loss": 0.9,
                "delta": -0.1, "rel_delta": -0.1,
            })
        # source=det, target=det (self-step) → must be excluded
        rows.append({
            "batch_idx": 0, "source_task": src, "target_task": src,
            "steps": 1, "variant": "normalized",
            "layer": "dec0_inter_gnn_0",
            "grad_norm": 1.0, "grad_dot": 0.0, "step_size": 1e-3,
            "baseline_loss": 1.0, "stepped_loss": 0.5,
            "delta": -0.5, "rel_delta": -0.5,
        })
    # noise row at the wrong variant
    rows.append({
        "batch_idx": 0, "source_task": "det", "target_task": "plan",
        "steps": 1, "variant": "raw",
        "layer": "dec0_inter_gnn_0",
        "grad_norm": 1.0, "grad_dot": 0.0, "step_size": 1e-3,
        "baseline_loss": 1.0, "stepped_loss": 1.0,
        "delta": 0.0, "rel_delta": 0.0,
    })
    # noise row at a non-inter_gnn layer
    rows.append({
        "batch_idx": 0, "source_task": "det", "target_task": "plan",
        "steps": 1, "variant": "normalized",
        "layer": "dec0_ffn_0",
        "grad_norm": 1.0, "grad_dot": 0.0, "step_size": 1e-3,
        "baseline_loss": 1.0, "stepped_loss": 0.7,
        "delta": -0.3, "rel_delta": -0.3,
    })
    csv = tmp_path / "probe_per_batch.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    out = load_plan_transfer(csv)
    assert set(out["aux_task"].unique()) == {"det", "map", "motion"}
    assert set(out["layer"].unique()) == {"dec0_inter_gnn_0"}
    # gain = -delta
    assert (out["gain"] + out["delta"]).abs().max() < 1e-12
    assert (out["gain"] > 0).all()


def test_load_task_collinearity_default_motion_vs_det(tmp_path):
    g = ["dec0_inter_gnn_0", "dec1_inter_gnn_0"]
    _write_pair(tmp_path, "det", "motion",
                _conflict_per_batch(g, [0, 0], [0.85, 0.40]))
    out = load_task_collinearity(tmp_path)  # default target=motion, ref=det
    assert list(out["collinearity"]) == pytest.approx([0.85, 0.40])
    assert set(out["group"].unique()) == set(g)


# --------------------------------------------------------------------- summaries


def test_build_importance_summary_basic():
    align = pd.DataFrame({
        "aux_task": ["det"] * 4 + ["map"] * 4 + ["motion"] * 4,
        "batch_idx": [0, 1, 2, 3] * 3,
        "group":    ["dec0_inter_gnn_0"] * 12,
        "cos":      [0.4, 0.5, 0.3, 0.2,    # det: align_pos_rate=1.0
                     -0.1, -0.2, 0.0, 0.1,  # map mixed
                     0.05, 0.05, 0.05, 0.05],
    })
    transfer = pd.DataFrame({
        "aux_task":  ["det"] * 4 + ["map"] * 4 + ["motion"] * 4,
        "batch_idx": [0, 1, 2, 3] * 3,
        "layer":     ["dec0_inter_gnn_0"] * 12,
        "delta":     [-0.4, -0.3, -0.5, -0.2,
                       0.1,  0.0, -0.1, -0.2,
                       0.02, -0.02, 0.01, -0.01],
        "gain":      [0.4, 0.3, 0.5, 0.2,
                     -0.1, 0.0,  0.1, 0.2,
                     -0.02, 0.02, -0.01, 0.01],
        "grad_dot":  [0.0] * 12,
        "grad_norm": [1.0] * 12,
    })
    coll = pd.DataFrame({
        "batch_idx": [0, 1, 2, 3],
        "group":     ["dec0_inter_gnn_0"] * 4,
        "collinearity": [0.9, 0.85, 0.82, 0.95],
    })
    out = build_importance_summary(align, transfer, coll, n_boot=200)
    out_by = out.set_index("aux_task")
    # det has all-positive align & gain
    assert out_by.loc["det", "align_pos_rate"] == pytest.approx(1.0)
    assert out_by.loc["det", "helpful_rate"] == pytest.approx(1.0)
    assert out_by.loc["det", "align_score"] == pytest.approx(np.mean([0.4, 0.5, 0.3, 0.2]))
    # motion gets collinearity + flag
    assert out_by.loc["motion", "motion_det_collinearity"] == pytest.approx(
        np.mean([0.9, 0.85, 0.82, 0.95]))
    assert bool(out_by.loc["motion", "flag_det_collinear"]) is True
    # non-motion rows do NOT inherit collinearity
    assert np.isnan(out_by.loc["det", "motion_det_collinearity"])
    assert bool(out_by.loc["det", "flag_det_collinear"]) is False


def test_build_scene_winner_alternating_and_fixed():
    # 6 batches, det and map alternate winning → high entropy
    rows = []
    for b in range(6):
        rows += [
            {"aux_task": "det",    "batch_idx": b, "cos": 1.0 if b % 2 == 0 else 0.1},
            {"aux_task": "map",    "batch_idx": b, "cos": 0.1 if b % 2 == 0 else 1.0},
            {"aux_task": "motion", "batch_idx": b, "cos": 0.0},
        ]
    df = pd.DataFrame(rows)
    wpb, dist, H = build_scene_winner(df, "cos")
    counts = dict(zip(dist["aux_task"], dist["win_count"]))
    assert counts == {"det": 3, "map": 3, "motion": 0}
    # H for 3-3-0 = log2(2)/log2(3) ≈ 0.6309
    assert H == pytest.approx(np.log2(2) / np.log2(3), rel=1e-6)
    assert (wpb["margin"] >= 0).all()

    # fixed winner → entropy 0
    rows_fixed = [
        {"aux_task": "det",    "batch_idx": b, "cos": 1.0} for b in range(5)
    ] + [
        {"aux_task": "map",    "batch_idx": b, "cos": 0.1} for b in range(5)
    ] + [
        {"aux_task": "motion", "batch_idx": b, "cos": 0.0} for b in range(5)
    ]
    _, dist2, H2 = build_scene_winner(pd.DataFrame(rows_fixed), "cos")
    counts2 = dict(zip(dist2["aux_task"], dist2["win_count"]))
    assert counts2 == {"det": 5, "map": 0, "motion": 0}
    assert H2 == pytest.approx(0.0)


def test_alignment_transfer_agreement_monotonic_and_sign():
    # Construct (batch, task) pairs where cos and gain are perfectly monotonic ⇒ Spearman ≈ 1
    rows_a, rows_t = [], []
    pairs = [
        ("det",   0.9), ("map",   0.5), ("motion", 0.1),
        ("det",   0.8), ("map",   0.4), ("motion", 0.05),
    ]
    batches = [0, 0, 0, 1, 1, 1]
    for (task, val), b in zip(pairs, batches):
        rows_a.append({"aux_task": task, "batch_idx": b,
                       "group": "dec0_inter_gnn_0", "cos": val})
        rows_t.append({"aux_task": task, "batch_idx": b,
                       "layer": "dec0_inter_gnn_0", "delta": -val,
                       "gain": val, "grad_dot": 0.0, "grad_norm": 1.0})
    agreement = build_alignment_transfer_agreement(pd.DataFrame(rows_a), pd.DataFrame(rows_t))
    assert agreement["spearman"] == pytest.approx(1.0, abs=1e-6)
    assert agreement["sign_agreement"] == pytest.approx(1.0)
    assert agreement["per_batch_winner_agreement"] == pytest.approx(1.0)
    assert agreement["n_joined"] == 6


def test_alignment_transfer_agreement_anti_monotonic():
    # cos and gain are anti-monotonic ⇒ Spearman ≈ -1, sign_agree < 1
    rows_a, rows_t = [], []
    for b in range(3):
        for task, val in [("det", 1.0), ("map", 0.5), ("motion", 0.1)]:
            rows_a.append({"aux_task": task, "batch_idx": b,
                           "group": "dec0_inter_gnn_0", "cos": val})
            rows_t.append({"aux_task": task, "batch_idx": b,
                           "layer": "dec0_inter_gnn_0", "delta": val,
                           "gain": -val, "grad_dot": 0.0, "grad_norm": 1.0})
    out = build_alignment_transfer_agreement(pd.DataFrame(rows_a), pd.DataFrame(rows_t))
    assert out["spearman"] == pytest.approx(-1.0, abs=1e-6)
    # signs differ everywhere
    assert out["sign_agreement"] == pytest.approx(0.0)


def test_build_planning_layer_sensitivity_picks_source_plan(tmp_path):
    rows = []
    for L, gn in [("dec0_inter_gnn_0", 0.5),
                  ("dec5_inter_gnn_0", 0.8),
                  ("dec0_ffn_0",       0.1)]:
        # one source=plan row per batch (deduplicated by drop_duplicates internally)
        for b in range(3):
            rows.append({
                "batch_idx": b, "source_task": "plan", "target_task": "det",
                "steps": 1, "variant": "normalized", "layer": L,
                "grad_norm": gn, "grad_dot": 0.0, "step_size": 1e-3,
                "baseline_loss": 1.0, "stepped_loss": 1.0,
                "delta": 0.0, "rel_delta": 0.0,
            })
        # noise row: source!=plan
        rows.append({
            "batch_idx": 0, "source_task": "det", "target_task": "plan",
            "steps": 1, "variant": "normalized", "layer": L,
            "grad_norm": 99.0, "grad_dot": 0.0, "step_size": 1e-3,
            "baseline_loss": 1.0, "stepped_loss": 0.9,
            "delta": -0.1, "rel_delta": -0.1,
        })
    csv = tmp_path / "probe_per_batch.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    out = build_planning_layer_sensitivity(csv)
    # only inter_gnn layers survive
    assert set(out["layer"]) == {"dec0_inter_gnn_0", "dec5_inter_gnn_0"}
    by = out.set_index("layer")["mean_grad_norm"]
    assert by.loc["dec0_inter_gnn_0"] == pytest.approx(0.5)
    assert by.loc["dec5_inter_gnn_0"] == pytest.approx(0.8)


# --------------------------------------------------------------------- runner


def test_run_planning_importance_end_to_end(tmp_path):
    lc_dir = tmp_path / "layer_conflict"
    lc_dir.mkdir()
    # write per-pair alignment data including the pair files the loader uses
    g = ["dec0_inter_gnn_0"] * 4
    _write_pair(lc_dir, "det",    "plan", _conflict_per_batch(g, [0, 1, 2, 3], [0.4, 0.5, 0.3, 0.2]))
    _write_pair(lc_dir, "map",    "plan", _conflict_per_batch(g, [0, 1, 2, 3], [-0.1, 0.0, 0.1, -0.2]))
    _write_pair(lc_dir, "motion", "plan", _conflict_per_batch(g, [0, 1, 2, 3], [0.05] * 4))
    _write_pair(lc_dir, "det",    "motion", _conflict_per_batch(g, [0, 1, 2, 3], [0.9, 0.88, 0.92, 0.85]))

    rows = []
    for src, gain in [("det", 0.4), ("map", -0.1), ("motion", 0.02)]:
        for b in range(4):
            rows.append({
                "batch_idx": b, "source_task": src, "target_task": "plan",
                "steps": 1, "variant": "normalized", "layer": "dec0_inter_gnn_0",
                "grad_norm": 1.0, "grad_dot": 0.0, "step_size": 1e-3,
                "baseline_loss": 1.0, "stepped_loss": 1.0 - gain,
                "delta": -gain, "rel_delta": -gain,
            })
    # plan self-step rows so layer-sensitivity has data
    for b in range(4):
        rows.append({
            "batch_idx": b, "source_task": "plan", "target_task": "det",
            "steps": 1, "variant": "normalized", "layer": "dec0_inter_gnn_0",
            "grad_norm": 0.7, "grad_dot": 0.0, "step_size": 1e-3,
            "baseline_loss": 1.0, "stepped_loss": 1.0,
            "delta": 0.0, "rel_delta": 0.0,
        })
    probe_csv = tmp_path / "probe_per_batch.csv"
    pd.DataFrame(rows).to_csv(probe_csv, index=False)

    out_dir = tmp_path / "planning_importance"
    res = run_planning_importance(lc_dir, probe_csv, out_dir, n_boot=200)

    for name in ("importance_summary.csv", "scene_winner_per_batch.csv",
                 "scene_winner_distribution.csv", "alignment_transfer_agreement.csv",
                 "planning_layer_sensitivity.csv"):
        assert (out_dir / name).exists(), f"missing {name}"

    summary = pd.read_csv(out_dir / "importance_summary.csv").set_index("aux_task")
    assert summary.loc["motion", "flag_det_collinear"]  # mean ≈ 0.89 ≥ 0.8
    assert summary.loc["det", "align_pos_rate"] == pytest.approx(1.0)

    layer_sens = pd.read_csv(out_dir / "planning_layer_sensitivity.csv")
    assert "dec0_inter_gnn_0" in set(layer_sens["layer"])
    # res entropy fields are numeric (incl. NaN allowed)
    assert "entropy_alignment" in res
    assert "entropy_transfer" in res
