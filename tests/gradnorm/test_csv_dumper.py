import csv
import datetime as dt
import os

import pytest

from projects.mmdet3d_plugin.core.gradnorm.csv_dumper import GradNormCSVDumper


def _row_dict(step):
    return {
        "step": step,
        "epoch": 0,
        "iter_in_epoch": step,
        "phase_id": 3,
        "lr_model": 1e-4,
        "w_det": 1.2, "w_map": 2.1, "w_motion": 0.3, "w_plan": 0.9, "w_ego": 0.5,
        "grad_norm_det": 0.1, "grad_norm_map": 0.2, "grad_norm_motion": 0.3,
        "grad_norm_plan": 0.4, "grad_norm_ego": 0.5,
        "rt_det": 1.0, "rt_map": 1.1, "rt_motion": 0.9, "rt_plan": 1.0, "rt_ego": 1.0,
        "L_det": 0.1, "L_map": 0.2, "L_motion": 0.3, "L_plan": 0.4, "L_ego": 0.5,
        "L0_det": 0.1, "L0_map": 0.2, "L0_motion": 0.3, "L0_plan": 0.4, "L0_ego": 0.5,
        "weighted_L_det": 0.12, "weighted_L_map": 0.42, "weighted_L_motion": 0.09,
        "weighted_L_plan": 0.36, "weighted_L_ego": 0.25,
        "contribution_frac_det": 0.1, "contribution_frac_map": 0.3,
        "contribution_frac_motion": 0.1, "contribution_frac_plan": 0.3,
        "contribution_frac_ego": 0.2,
        "grad_norm_share_det": 0.1, "grad_norm_share_map": 0.2,
        "grad_norm_share_motion": 0.3, "grad_norm_share_plan": 0.2,
        "grad_norm_share_ego": 0.2,
        "w_ratio_det": 1.0, "w_ratio_map": 1.05, "w_ratio_motion": 0.95,
        "w_ratio_plan": 1.0, "w_ratio_ego": 1.0,
        "gn_loss": 0.01,
        "timestamp_iso": dt.datetime.utcnow().isoformat(),
    }


def test_dumper_writes_header_then_rows(tmp_path):
    path = tmp_path / "gradnorm_log.csv"
    d = GradNormCSVDumper(str(path), rank=0)
    d.append(_row_dict(0))
    d.append(_row_dict(1))

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert len(rows) == 2
    assert int(rows[0]["step"]) == 0
    assert int(rows[1]["step"]) == 1
    expected_cols = set(_row_dict(0).keys())
    assert set(fieldnames) == expected_cols


def test_dumper_rank_nonzero_is_noop(tmp_path):
    path = tmp_path / "gradnorm_log.csv"
    d = GradNormCSVDumper(str(path), rank=1)
    d.append(_row_dict(0))
    assert not os.path.exists(path)


def test_dumper_rolls_existing_file(tmp_path):
    path = tmp_path / "gradnorm_log.csv"
    path.write_text("stale\n")
    GradNormCSVDumper(str(path), rank=0, roll_existing=True)
    assert not path.exists()
    rolled = list(tmp_path.glob("gradnorm_log_*.csv"))
    assert len(rolled) == 1
