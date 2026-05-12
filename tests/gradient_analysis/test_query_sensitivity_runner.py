"""Smoke test for the query sensitivity runner CSV schema."""
from pathlib import Path

import pandas as pd
import torch
from torch import nn


class _Adapter:
    def split_losses(self, fwd, task):
        return fwd[task]
    def snapshot_temporal_state(self, model):
        return None
    def restore_temporal_state(self, model, snap):
        pass
    def freeze_stochastic_state(self):
        from contextlib import nullcontext
        return nullcontext()
    def get_task_queries(self, model, fwd_artifacts):
        return fwd_artifacts["task_queries"]


class _Collector:
    def __init__(self):
        torch.manual_seed(0)
        self.qs = {
            "det":    torch.randn(3, 2, requires_grad=True),
            "map":    torch.randn(3, 2, requires_grad=True),
            "motion": torch.randn(3, 2, requires_grad=True),
            "plan":   torch.randn(3, 2, requires_grad=True),
        }
        self.W = nn.Linear(2, 1)
        self.tasks = ["det", "map", "motion", "plan"]
        self.adapter = _Adapter()
    @property
    def model(self):
        return self.W
    def forward_losses(self, data):
        plan = self.W(self.qs["plan"]).pow(2).mean() \
             + self.W(self.qs["det"]).pow(2).mean() \
             + self.W(self.qs["map"]).pow(2).mean() \
             + self.W(self.qs["motion"]).pow(2).mean()
        return {"plan": plan, "task_queries": dict(self.qs)}


def test_run_query_sensitivity_writes_expected_schema(tmp_path):
    from tools.gradient_analysis.query_sensitivity import run_query_sensitivity

    col = _Collector()
    df = run_query_sensitivity(
        collector=col,
        dataloader=[None],
        num_batches=1,
        out_dir=tmp_path,
        forward_seed=None,
        freeze_matching=False,
    )
    df_csv = pd.read_csv(tmp_path / "qs.csv")
    expected = {"model", "checkpoint", "checkpoint_order", "batch_idx",
                "scene_token", "layer", "task_query_type", "query_index",
                "query_norm", "grad_plan_wrt_query_norm"}
    assert expected.issubset(df_csv.columns)
    assert len(df_csv) >= 12  # 4 tasks × 3 query rows
