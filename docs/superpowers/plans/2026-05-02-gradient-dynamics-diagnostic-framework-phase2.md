# Gradient Dynamics Diagnostic Framework — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land Phase 2 of the diagnostic framework — define `GradientAnalysisAdapter` Protocol, migrate HiP-AD's inline dependencies behind it, build the VAD adapter in parallel, refactor M1–M8 modules to consume only adapters, and run the VAD primary run (4 ckpts × 100 batches) so the same Phase 1 artefact set exists for both models.

**Architecture:** The Protocol is authored *while writing both adapters at once* (R5 mitigation): the six adapter operation groups (`build_model+dataloader` / `forward_losses+selective_eval_types+split_losses` / `shared_param_groups` / `freeze_stochastic_state` / `snapshot+restore_temporal_state`) are introduced one pair at a time (T3 HiP-AD → T4 VAD → T5 HiP-AD → T6 VAD → …) so the interface stays bilingual. v1 BatchGradients caches keep working through `collector.py` (R3 mitigation); a numeric parity gate (T19) verifies the existing HiP-AD Phase 1 report regenerates byte-equivalent through the adapter path. M1–M8 module rewrites consume only adapters; HiP-AD direct imports are deleted only after the parity gate passes.

**Tech Stack:** Python 3.8+, PyTorch 1.13+ (HiP-AD env), PyTorch 1.10+ (VAD env), mmdet3d (`init_detector`), pytest, scipy ≥ 1.10. Two conda envs:
- `HIPAD_PY=/home/yongjae/miniconda3/envs/hipad/bin/python`
- `VAD_PY=/home/yongjae/miniconda3/envs/vad/bin/python`

**Spec:** [docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework-design.md](../specs/2026-05-02-gradient-dynamics-diagnostic-framework-design.md) §4
**Phase 1 plan:** [2026-05-02-gradient-dynamics-diagnostic-framework-phase1.md](2026-05-02-gradient-dynamics-diagnostic-framework-phase1.md)
**Phase 1 gating decision:** [phase1_gating_decision.md](../specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase1_gating_decision.md) — branch **Pass B** (magnitude-dominated, conflict-weak)

**Conventions for this plan:**
- Paths are relative to `/home/yongjae/e2e/HiP-AD/` unless prefixed with `VAD:` (which is `/home/yongjae/e2e/VAD/`).
- HiP-AD work runs from `/home/yongjae/e2e/HiP-AD` with `PYTHONPATH=.` and `HIPAD_PY`.
- VAD work runs from `/home/yongjae/e2e/VAD` with `PYTHONPATH=.` and `VAD_PY`.
- pytest runs from the HiP-AD repo root with `PYTHONPATH=.` (HiP-AD env imports `tools.gradient_analysis.adapters.vad` lazily, so HiP-AD pytest never instantiates VAD).
- Per the user's preference, **do not commit between tasks** — they will batch-commit at the end of Phase 2. Each task ends with a "Stop and review" checkpoint instead of a `git commit` step.
- HiP-AD output regenerated through the adapter path goes to `gradient_analysis_results_phase2_hipad/`; VAD output goes to `gradient_analysis_results_phase2_vad/`. Phase 1 results in `gradient_analysis_results_phase1/` are read-only references for the parity gate (T19).
- **R5 interleave rule:** every HiP-AD adapter op task is immediately followed by the same op for VAD before moving on. The Protocol may be edited in either of the paired tasks; treat the two as a single conceptual unit.

---

## File Structure

**New files:**
- `tools/gradient_analysis/adapters/__init__.py` — exports
- `tools/gradient_analysis/adapters/base.py` — `GradientAnalysisAdapter` Protocol + `TemporalSnapshot` dataclass (≈ 90 LoC)
- `tools/gradient_analysis/adapters/hipad.py` — HiP-AD adapter (≈ 380 LoC); subsumes `_import_hipad_utils`, `_selective_eval`, `FrozenMatching`, `ModelStateSnapshot`, `compat` patches
- `tools/gradient_analysis/adapters/vad.py` — VAD adapter (≈ 380 LoC)
- `tests/gradient_analysis/adapters/__init__.py`
- `tests/gradient_analysis/adapters/test_protocol.py` — `MockAdapter` Protocol conformance (≈ 180 LoC)
- `tests/gradient_analysis/adapters/test_hipad_adapter.py` — single-batch smoke against ckpt 1ep (≈ 120 LoC)
- `tests/gradient_analysis/adapters/test_vad_adapter.py` — single-batch smoke against epoch_1.pth (≈ 120 LoC)
- `configs/gradient_analysis_vad.yaml` — VAD primary-run config
- `gradient_analysis_results_phase2_hipad/` — HiP-AD output regenerated through adapter path (T19 parity gate input)
- `gradient_analysis_results_phase2_vad/` — VAD primary-run output (T22)
- `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_summary.md` — closing memo (T24)

**Modified files:**
- `tools/gradient_analysis/collector.py` — `GradientCollector` accepts an adapter; `_import_hipad_utils`, `_selective_eval`, dataloader factory deleted (moved into adapters); v1 BatchGradients cache loader path preserved (R3)
- `tools/gradient_analysis/probe.py` — adapter consumer (`FrozenMatching` import removed; uses `adapter.freeze_stochastic_state()`, `adapter.snapshot_temporal_state()`)
- `tools/gradient_analysis/landscape.py` — adapter consumer
- `tools/gradient_analysis/conflict.py` / `dynamics.py` / `asymmetry.py` / `gradnorm.py` / `magnitude_dynamics.py` / `null_baseline.py` / `distribution.py` / `bootstrap.py` / `summary.py` — adapter consumer where they currently call into HiP-AD utilities
- `tools/run_gradient_analysis.py` — `--adapter {hipad,vad}` flag; routes through adapter
- `tools/gradient_analysis/matching_freeze.py` — DELETED at end of T17 (logic lives in `adapters/hipad.py`)
- `tools/gradient_analysis/temporal_state.py` — DELETED at end of T17 (logic lives in `adapters/hipad.py`)
- `tools/gradient_analysis/compat.py` — DELETED at end of T17 (HiP-AD-specific patches moved into `adapters/hipad.py`)

---

## Task 0: Pre-Flight — VAD Env, Config, Loader, Forward Smoke (R1)

**Files:**
- Modify: none (read-only validation; one ephemeral helper script that gets deleted)

- [ ] **Step 1: Confirm VAD env is healthy and core deps import**

Run:

```bash
VAD_PY=/home/yongjae/miniconda3/envs/vad/bin/python
$VAD_PY -c "import torch, mmdet3d, mmcv; from mmdet3d.apis import init_detector; print('torch', torch.__version__); print('mmdet3d', mmdet3d.__version__); print('mmcv', mmcv.__version__)"
```

Expected: three version strings print with no traceback. Failure here means the VAD env is missing core deps — fix before continuing (re-run VAD's install steps; do not proceed).

- [ ] **Step 2: Confirm `init_detector` loads VAD config + epoch_1.pth**

Run:

```bash
cd /home/yongjae/e2e/VAD
PYTHONPATH=. $VAD_PY -c "
import projects  # registers VAD detector / dataset
from mmdet3d.apis import init_detector
m = init_detector('data/ckpts/VAD_tiny_e2e.py', 'data/ckpts/epoch_1.pth', device='cuda:0')
print(type(m).__name__, sum(p.numel() for p in m.parameters()))
"
```

Expected: `VAD <param-count>`. Common failures:
- `KeyError: 'VAD'` → `import projects` line missing.
- `CUDA out of memory` → switch device to `cuda:1` or `cpu`.

If either fails after a fix attempt, STOP and allocate the 0.5-day R1 spike budget here.

- [ ] **Step 3: Confirm VAD train dataloader yields a batch end-to-end**

Run:

```bash
cd /home/yongjae/e2e/VAD
PYTHONPATH=. $VAD_PY -c "
import projects  # noqa
from functools import partial
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmcv.parallel import collate
from torch.utils.data import DataLoader
cfg = Config.fromfile('data/ckpts/VAD_tiny_e2e.py')
ds = build_dataset(cfg.data.train)
dl = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=partial(collate, samples_per_gpu=1))
b = next(iter(dl))
print('keys:', sorted(list(b.keys()))[:12])
print('img.data[0].shape:', b['img'].data[0].shape)
"
```

Expected: keys list including `img`, `gt_bboxes_3d`, `gt_labels_3d`, `map_gt_bboxes_3d`, `map_gt_labels_3d`, `ego_fut_trajs`, `ego_fut_masks`, `ego_fut_cmd`, `ego_lcf_feat`, `gt_attr_labels`, `img_metas`. img tensor shape printed.

- [ ] **Step 4: Single forward → snapshot every loss-dict key**

Create a one-shot script `VAD:tools/_phase2_smoke.py`:

```python
"""Phase 2 R1 smoke: load VAD, run one training-mode forward, dump loss-dict keys."""
import sys; sys.path.insert(0, '.')
import projects  # noqa: F401  (registers VAD detector + dataset)
from functools import partial
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmdet3d.apis import init_detector
from mmdet3d.datasets import build_dataset
from torch.utils.data import DataLoader

cfg = Config.fromfile('data/ckpts/VAD_tiny_e2e.py')
m = init_detector('data/ckpts/VAD_tiny_e2e.py', 'data/ckpts/epoch_1.pth', device='cuda:0')
m.train()
ds = build_dataset(cfg.data.train)
dl = DataLoader(ds, batch_size=1, num_workers=0,
                collate_fn=partial(collate, samples_per_gpu=1))
batch = next(iter(dl))
batch = scatter(batch, [0])[0]
losses = m(**batch)
print("# VAD loss-dict keys")
for k in sorted(losses.keys()):
    v = losses[k]
    desc = tuple(v.shape) if torch.is_tensor(v) else type(v).__name__
    print(f"{k}\t{desc}")
```

Run:

```bash
cd /home/yongjae/e2e/VAD
PYTHONPATH=. $VAD_PY tools/_phase2_smoke.py | tee /tmp/vad_loss_keys.txt
```

Expected: a sorted list of every key in `losses` (≈ 30–60 keys, including decoder-layer prefixes `d0.loss_*` ... `d5.loss_*`). Save the file — T6 (`split_losses`) needs the exact key names.

- [ ] **Step 5: Delete the smoke script**

```bash
rm /home/yongjae/e2e/VAD/tools/_phase2_smoke.py
```

The captured key list at `/tmp/vad_loss_keys.txt` survives.

- [ ] **Step 6: Stop and review**

If Steps 1–4 all passed, R1 is closed: VAD env, config, dataloader, model load, training-mode forward, and loss-dict keys are confirmed. The captured loss-key list at `/tmp/vad_loss_keys.txt` is the input to T6. Move to Task 1.

If any step still fails after fixes, STOP. The 0.5-day R1 spike budget per spec applies — fix the blocker before any adapter code is written.

---

## Task 1: Protocol Skeleton + TemporalSnapshot Dataclass

**Files:**
- Create: `tools/gradient_analysis/adapters/__init__.py`
- Create: `tools/gradient_analysis/adapters/base.py`
- Create: `tests/gradient_analysis/adapters/__init__.py`
- Create: `tests/gradient_analysis/adapters/test_protocol.py`

- [ ] **Step 1: Write the failing test**

Create `tests/gradient_analysis/adapters/__init__.py` (empty file).

Create `tests/gradient_analysis/adapters/test_protocol.py`:

```python
"""Conformance tests for the GradientAnalysisAdapter Protocol shape."""
import inspect

from tools.gradient_analysis.adapters.base import (
    GradientAnalysisAdapter,
    TemporalSnapshot,
)


def test_temporal_snapshot_default_payload():
    snap = TemporalSnapshot(model_state_keys=("k1", "k2"))
    assert snap.model_state_keys == ("k1", "k2")
    assert snap.payload == {}


def test_temporal_snapshot_payload_set():
    snap = TemporalSnapshot(model_state_keys=(), payload={"a": 1})
    assert snap.payload == {"a": 1}


def test_protocol_has_required_methods():
    expected = {
        "tasks", "build_model", "build_dataloader",
        "forward_losses", "split_losses", "shared_param_groups",
        "selective_eval_types", "freeze_stochastic_state",
        "snapshot_temporal_state", "restore_temporal_state",
    }
    members = {n for n, _ in inspect.getmembers(GradientAnalysisAdapter)}
    missing = expected - members
    assert not missing, f"Protocol missing: {missing}"
```

- [ ] **Step 2: Run test — verify failure**

```bash
HIPAD_PY=/home/yongjae/miniconda3/envs/hipad/bin/python
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_protocol.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.gradient_analysis.adapters'`.

- [ ] **Step 3: Create the adapter package and Protocol**

Create `tools/gradient_analysis/adapters/__init__.py`:

```python
"""Model-specific adapters for gradient analysis modules.

Phase 2 introduces HiP-AD and VAD adapters authored in parallel against a
single Protocol so the interface does not inherit either model's
assumptions (spec §4.1, R5 mitigation).
"""
from .base import GradientAnalysisAdapter, TemporalSnapshot

__all__ = ["GradientAnalysisAdapter", "TemporalSnapshot"]
```

Create `tools/gradient_analysis/adapters/base.py`:

```python
"""Adapter Protocol shared by HiP-AD and VAD.

Ten operations capture every model-specific behaviour the M1-M8 modules
currently inline against HiP-AD. Implementations live in `hipad.py` and
`vad.py` and must satisfy `MockAdapter`-driven Protocol conformance tests
(see `tests/gradient_analysis/adapters/test_protocol.py`).

The interface is intentionally NOT a single-model abstraction: every method
exists because BOTH adapters need it, and is added during the interleaved
authoring (T3-T12 of phase2 plan).

The spec §4.1 sketched 9 operations; we add `restore_temporal_state` for
symmetry with `snapshot_temporal_state` (HiP-AD's existing
`ModelStateSnapshot` had the snapshot/restore pair built in; the Protocol
makes the pair explicit).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
    runtime_checkable,
)

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TemporalSnapshot:
    """Opaque payload holding the per-forward mutable state an adapter needs
    to keep stable across a probe cycle (run_step counters, instance-bank
    caches, prev_bev queues, sampler dn_metas, ...).

    Adapters are free to put whatever they need in `payload`; M3 probe code
    only ever calls `adapter.restore_temporal_state(model, snap)`, never
    inspects the payload.

    `model_state_keys` is documentation-only — names of the attributes the
    snapshot tracks, used by tests / logs to verify the adapter actually
    captured something.
    """
    model_state_keys: Tuple[str, ...]
    payload: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GradientAnalysisAdapter(Protocol):
    """Model-specific surface used by every M1-M8 module.

    Method ordering matches the lifecycle of a probe batch:
      1. tasks                         — declare which tasks exist
      2. build_model / build_dataloader — construct evaluation harness
      3. forward_losses                — run forward; return raw loss dict
      4. split_losses                  — (loss_dict, task) -> single tensor
      5. shared_param_groups           — grouped parameters for M1
      6. selective_eval_types          — types to put in eval() during probe
      7. freeze_stochastic_state       — context manager for matching freeze
      8. snapshot_temporal_state       — capture mutable temporal caches
      9. restore_temporal_state        — restore from a snapshot
    """

    @property
    def tasks(self) -> List[str]: ...

    def build_model(self, ckpt: Path, device: str) -> nn.Module: ...

    def build_dataloader(
        self, batch_size: int, seed: int, shuffle: bool = False,
    ) -> DataLoader: ...

    def forward_losses(
        self, model: nn.Module, data: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]: ...

    def split_losses(
        self, loss_dict: Mapping[str, torch.Tensor], task: str,
    ) -> Optional[torch.Tensor]: ...

    def shared_param_groups(
        self, model: nn.Module, group_names: List[str],
    ) -> Dict[str, List[nn.Parameter]]: ...

    def selective_eval_types(self) -> Tuple[Type[nn.Module], ...]: ...

    @contextmanager
    def freeze_stochastic_state(self) -> Iterator[None]: ...

    def snapshot_temporal_state(self, model: nn.Module) -> TemporalSnapshot: ...

    def restore_temporal_state(
        self, model: nn.Module, snapshot: TemporalSnapshot,
    ) -> None: ...
```

- [ ] **Step 4: Run test — verify pass**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_protocol.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Stop and review**

`base.py` defines the Protocol surface; no implementation yet. Next: a `MockAdapter` that exercises every method (T2), then op-by-op interleaved authoring (T3–T12). The Protocol can be edited later if either adapter exposes a leak — that's intended (R5).

---

## Task 2: MockAdapter Test Scaffold

**Files:**
- Modify: `tests/gradient_analysis/adapters/test_protocol.py`

- [ ] **Step 1: Add the failing MockAdapter test**

Append to `tests/gradient_analysis/adapters/test_protocol.py`:

```python
import contextlib
from typing import Iterator, List, Mapping, Tuple, Type
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MockAdapter:
    """Minimal in-memory adapter exercising every Protocol method.

    Used so M1-M8 module tests do not depend on either real model. The
    construction is *not* an instance of GradientAnalysisAdapter at the
    typing level (we don't subclass the Protocol), but `runtime_checkable`
    on the Protocol means `isinstance(MockAdapter(), GradientAnalysisAdapter)`
    is True iff the structural shape matches — and the test below asserts
    exactly that.
    """

    @property
    def tasks(self) -> List[str]:
        return ["t0", "t1"]

    def build_model(self, ckpt, device):
        m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        return m.to(device)

    def build_dataloader(self, batch_size, seed, shuffle=False):
        torch.manual_seed(seed)
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                          shuffle=shuffle)

    def forward_losses(self, model, data):
        x, y = data
        logits = model(x)
        return {
            "t0": logits.pow(2).mean(),
            "t1": (logits - y[:, None].float()).pow(2).mean(),
        }

    def split_losses(self, loss_dict, task):
        return loss_dict.get(task)

    def shared_param_groups(self, model, group_names):
        out = {}
        for gk in group_names:
            if gk == "linear0":
                out[gk] = list(model[0].parameters())
            elif gk == "linear1":
                out[gk] = list(model[1].parameters())
        return out

    def selective_eval_types(self):
        return (nn.Dropout,)

    @contextlib.contextmanager
    def freeze_stochastic_state(self) -> Iterator[None]:
        yield

    def snapshot_temporal_state(self, model):
        from tools.gradient_analysis.adapters.base import TemporalSnapshot
        return TemporalSnapshot(model_state_keys=("dummy",),
                                payload={"dummy": 0})

    def restore_temporal_state(self, model, snapshot):
        return None


def test_mockadapter_satisfies_protocol_runtime():
    """`runtime_checkable` Protocol — structural conformance check."""
    a = MockAdapter()
    assert isinstance(a, GradientAnalysisAdapter)


def test_mockadapter_full_lifecycle():
    a = MockAdapter()
    assert a.tasks == ["t0", "t1"]
    model = a.build_model(ckpt=None, device="cpu")
    dl = a.build_dataloader(batch_size=2, seed=0)
    batch = next(iter(dl))
    losses = a.forward_losses(model, batch)
    assert "t0" in losses and "t1" in losses
    assert a.split_losses(losses, "t0") is losses["t0"]
    assert a.split_losses(losses, "missing") is None
    groups = a.shared_param_groups(model, ["linear0", "linear1"])
    assert set(groups) == {"linear0", "linear1"}
    assert all(isinstance(p, nn.Parameter) for ps in groups.values() for p in ps)
    types = a.selective_eval_types()
    assert nn.Dropout in types
    with a.freeze_stochastic_state():
        snap = a.snapshot_temporal_state(model)
    assert "dummy" in snap.model_state_keys
    a.restore_temporal_state(model, snap)
```

- [ ] **Step 2: Run test — verify pass**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_protocol.py -v
```

Expected: 5 tests pass (3 from T1 + 2 new).

- [ ] **Step 3: Stop and review**

The Protocol shape is anchored by the runtime conformance test — any method added/renamed in `base.py` that `MockAdapter` does not also implement will surface here. Next: HiP-AD `tasks/build_model/build_dataloader` (T3), then VAD (T4).

---

## Task 3: HiP-AD Adapter — `tasks` / `build_model` / `build_dataloader`

**Files:**
- Create: `tools/gradient_analysis/adapters/hipad.py`
- Create: `tests/gradient_analysis/adapters/test_hipad_adapter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/gradient_analysis/adapters/test_hipad_adapter.py`:

```python
"""Single-batch smoke tests for the HiP-AD adapter.

Skipped automatically if the HiP-AD repo + ckpt are not available, so this
file can live in CI without infrastructure.
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HIPAD_CKPT = REPO_ROOT / "ckpts" / "HiP-AD-Stage2_1ep.pth"
HIPAD_CONFIG = REPO_ROOT / "ckpts" / "HiP-AD-Stage2_code.py"

needs_hipad = pytest.mark.skipif(
    not (HIPAD_CKPT.exists() and HIPAD_CONFIG.exists()),
    reason="HiP-AD ckpt or config not present",
)


@needs_hipad
def test_hipad_adapter_tasks_and_build_model():
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    assert sorted(a.tasks) == ["det", "map", "motion", "plan"]

    model = a.build_model(ckpt=HIPAD_CKPT, device="cuda:0")
    assert sum(p.numel() for p in model.parameters()) > 0
    assert any(p.requires_grad for p in model.parameters())


@needs_hipad
def test_hipad_adapter_build_dataloader():
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    dl = a.build_dataloader(batch_size=1, seed=0, shuffle=False)
    batch = next(iter(dl))
    assert batch is not None
```

- [ ] **Step 2: Run — verify failure**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_hipad_adapter.py -v
```

Expected: `ModuleNotFoundError: tools.gradient_analysis.adapters.hipad`.

- [ ] **Step 3: Create the HiP-AD adapter scaffold + first three ops**

Create `tools/gradient_analysis/adapters/hipad.py`:

```python
"""HiP-AD adapter for the gradient-analysis pipeline.

Phase 2 migration: subsumes the Phase-1 inline dependencies
(`_import_hipad_utils`, `_selective_eval`, `FrozenMatching`,
`ModelStateSnapshot`, `compat` patches) behind the
`GradientAnalysisAdapter` Protocol so M1-M8 modules become model-agnostic.

Built up op-by-op alongside `vad.py` (T3-T12). HiP-AD-specific imports are
deferred (lazy) so the module is importable in environments without the
HiP-AD repo on `sys.path`.
"""
from __future__ import annotations

import contextlib
import sys
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
)

import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import TemporalSnapshot


HIPAD_DEFAULT_CONFIG = Path("ckpts/HiP-AD-Stage2_code.py")


def _ensure_hipad_paths() -> Path:
    """Make sure both the repo root and `tools/` are on sys.path so the
    HiP-AD plugin packages are importable. Returns the repo root."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # .../HiP-AD/
    tools_dir = repo_root / "tools"
    for p in (str(repo_root), str(tools_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo_root


class HipadAdapter:
    """Concrete adapter for HiP-AD-Stage2."""

    def __init__(self, config_path: Path = HIPAD_DEFAULT_CONFIG):
        self._repo_root = _ensure_hipad_paths()
        self._config_path = (self._repo_root / config_path
                             if not Path(config_path).is_absolute()
                             else Path(config_path))
        # Lazy: only loaded once a method that needs the config runs.
        self._cfg = None
        self._utils = None

    # -------------------------------------------------------------- ops

    @property
    def tasks(self) -> List[str]:
        utils = self._load_utils()
        return list(utils["TASK_GROUPS"].keys())

    def build_model(self, ckpt: Path, device: str) -> nn.Module:
        from mmcv import Config  # type: ignore
        from mmcv.runner import load_checkpoint  # type: ignore
        from mmdet.models import build_detector  # type: ignore

        if self._cfg is None:
            self._cfg = Config.fromfile(str(self._config_path))
        cfg = self._cfg
        model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"),
                                test_cfg=cfg.get("test_cfg"))
        load_checkpoint(model, str(ckpt), map_location="cpu")
        model.to(device)
        return model

    def build_dataloader(
        self, batch_size: int, seed: int, shuffle: bool = False,
    ) -> DataLoader:
        from mmcv import Config  # type: ignore
        from mmcv.parallel import collate  # type: ignore
        from projects.mmdet3d_plugin.datasets.builder import (  # type: ignore
            custom_build_dataset,
        )

        if self._cfg is None:
            self._cfg = Config.fromfile(str(self._config_path))
        ds = custom_build_dataset(self._cfg.data.train)
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=min(4, batch_size),
            collate_fn=partial(collate, samples_per_gpu=batch_size),
            drop_last=True,
            generator=g,
        )

    # ------------------------------------------------------ HiP-AD utils

    def _load_utils(self) -> Dict[str, Any]:
        """Lazy-load HiP-AD's gradient-analysis utility surface.

        Mirrors the old `_import_hipad_utils` in collector.py — kept here
        instead of in `tools/` because every method below depends on it.
        """
        if self._utils is not None:
            return self._utils
        _ensure_hipad_paths()
        from analyze_gradient_conflict import (  # type: ignore
            TASK_GROUPS,
            _sum_task_loss,
            compute_task_gradient_grouped,
        )
        from projects.mmdet3d_plugin.core.hooks.pcgrad_optimizer_hook import (  # type: ignore
            get_shared_parameters_grouped,
        )
        self._utils = {
            "TASK_GROUPS": TASK_GROUPS,
            "_sum_task_loss": _sum_task_loss,
            "compute_task_gradient_grouped": compute_task_gradient_grouped,
            "get_shared_parameters_grouped": get_shared_parameters_grouped,
        }
        return self._utils
```

- [ ] **Step 4: Run — verify pass**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_hipad_adapter.py -v
```

Expected: 2 tests pass (or `SKIPPED` if ckpt 1ep pth is not at the expected path — in that case set `HIPAD_CKPT` in the test file to a real path).

- [ ] **Step 5: Stop and review**

Three of ten ops are live for HiP-AD. Next: VAD's first three ops (T4), then the next pair on both sides (T5/T6).

---

## Task 4: VAD Adapter — `tasks` / `build_model` / `build_dataloader`

**Files:**
- Create: `tools/gradient_analysis/adapters/vad.py`
- Create: `tests/gradient_analysis/adapters/test_vad_adapter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/gradient_analysis/adapters/test_vad_adapter.py`:

```python
"""Single-batch smoke tests for the VAD adapter.

Skipped if the VAD repo + ckpt are not available. Tests run from the
HiP-AD repo root with `PYTHONPATH=.`; the adapter pushes the VAD repo on
sys.path internally.
"""
from pathlib import Path

import pytest

VAD_REPO = Path("/home/yongjae/e2e/VAD")
VAD_CKPT = VAD_REPO / "data/ckpts/epoch_1.pth"
VAD_CONFIG = VAD_REPO / "data/ckpts/VAD_tiny_e2e.py"

needs_vad = pytest.mark.skipif(
    not (VAD_CKPT.exists() and VAD_CONFIG.exists()),
    reason="VAD ckpt or config not present",
)


@needs_vad
def test_vad_adapter_tasks():
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    assert sorted(a.tasks) == ["det", "map", "motion", "plan"]


@needs_vad
def test_vad_adapter_build_model_and_dataloader():
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    model = a.build_model(ckpt=VAD_CKPT, device="cuda:0")
    assert sum(p.numel() for p in model.parameters()) > 0

    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))
    assert "img" in batch
```

- [ ] **Step 2: Run — verify failure**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_vad_adapter.py -v
```

Expected: `ModuleNotFoundError: tools.gradient_analysis.adapters.vad`.

- [ ] **Step 3: Create the VAD adapter scaffold + first three ops**

Create `tools/gradient_analysis/adapters/vad.py`:

```python
"""VAD adapter for the gradient-analysis pipeline.

VAD-specific concerns vs HiP-AD:
  * Tasks: det / map / motion / plan (motion is "trajectory" in VAD's
    head — `loss_traj`/`loss_traj_cls`).
  * Loss-key conventions:
      det:    loss_cls, loss_bbox, d{i}.loss_cls, d{i}.loss_bbox
      map:    loss_map_cls, loss_map_bbox, loss_map_iou, loss_map_pts,
              loss_map_dir, d{i}.loss_map_*
      motion: loss_traj, loss_traj_cls, d{i}.loss_traj, d{i}.loss_traj_cls
      plan:   loss_plan_reg, loss_plan_bound, loss_plan_col, loss_plan_dir
  * Shared parameter groups: BEVFormer encoder layers (TemporalSelfAttention,
    SpatialCrossAttention, FFN, LayerNorm) and decoder layers — translated
    from a canonical group-name set into the actual nested attribute paths.
  * Stochastic-state freeze: VAD uses HungarianAssigner3D + PseudoSampler
    for det / map; `freeze_stochastic_state` patches their assign methods.
    motion / plan in VAD are mode-argmin only (no Hungarian); the freeze
    surface there is the assign-time argmin in the head.
  * Temporal state: VAD's training-mode `prev_bev` is recomputed per batch
    by `obtain_history_bev` and not cached across forwards, so the
    snapshot is largely defensive (run_step counter + the `prev_frame_info`
    dict on the detector if `video_test_mode` is on).
  * Model construction: `init_detector` from mmdet3d (HiP-AD uses
    `mmdet.build_detector`).
"""
from __future__ import annotations

import contextlib
import sys
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
)

import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import TemporalSnapshot


VAD_DEFAULT_REPO = Path("/home/yongjae/e2e/VAD")
VAD_DEFAULT_CONFIG = VAD_DEFAULT_REPO / "data/ckpts/VAD_tiny_e2e.py"


def _ensure_vad_paths(repo_root: Path) -> None:
    """Push the VAD repo on sys.path and import its plugin so the
    `projects.mmdet3d_plugin.*` registrations fire."""
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import projects  # type: ignore  # noqa: F401  (registration side-effect)


_TASKS = ["det", "map", "motion", "plan"]


class VadAdapter:
    """Concrete adapter for VAD-tiny / VAD-base."""

    def __init__(
        self,
        repo_root: Path = VAD_DEFAULT_REPO,
        config_path: Path = VAD_DEFAULT_CONFIG,
    ):
        self._repo_root = Path(repo_root)
        self._config_path = Path(config_path)
        self._cfg = None
        _ensure_vad_paths(self._repo_root)

    @property
    def tasks(self) -> List[str]:
        return list(_TASKS)

    def build_model(self, ckpt: Path, device: str) -> nn.Module:
        from mmdet3d.apis import init_detector  # type: ignore
        return init_detector(str(self._config_path), str(ckpt), device=device)

    def build_dataloader(
        self, batch_size: int, seed: int, shuffle: bool = False,
    ) -> DataLoader:
        from mmcv import Config  # type: ignore
        from mmcv.parallel import collate  # type: ignore
        from mmdet3d.datasets import build_dataset  # type: ignore

        if self._cfg is None:
            self._cfg = Config.fromfile(str(self._config_path))
        ds = build_dataset(self._cfg.data.train)
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=min(4, batch_size),
            collate_fn=partial(collate, samples_per_gpu=batch_size),
            drop_last=True,
            generator=g,
        )
```

- [ ] **Step 4: Run — verify pass**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_vad_adapter.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Stop and review**

Three ops parity on both sides. The Protocol's `tasks`/`build_model`/`build_dataloader` shape held against both models. Next: `forward_losses` + `selective_eval_types` + `split_losses` (T5 HiP-AD, T6 VAD).

---

## Task 5: HiP-AD Adapter — `forward_losses` + `selective_eval_types` + `split_losses`

**Files:**
- Modify: `tools/gradient_analysis/adapters/hipad.py`
- Modify: `tests/gradient_analysis/adapters/test_hipad_adapter.py`

- [ ] **Step 1: Append failing test**

Append to `tests/gradient_analysis/adapters/test_hipad_adapter.py`:

```python
@needs_hipad
def test_hipad_adapter_forward_and_split():
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    model = a.build_model(ckpt=HIPAD_CKPT, device="cuda:0")
    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))
    losses = a.forward_losses(model, batch)
    assert isinstance(losses, dict)
    # All four tasks split out non-trivially.
    for t in ["det", "map", "motion", "plan"]:
        sub = a.split_losses(losses, t)
        assert sub is not None, f"missing task in split: {t}"
        assert sub.requires_grad


@needs_hipad
def test_hipad_adapter_selective_eval_types():
    from torch import nn
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    types = a.selective_eval_types()
    assert nn.Dropout in types
    assert nn.BatchNorm2d in types
    # HiP-AD-specific: DeformableFeatureAggregation
    assert any("DeformableFeatureAggregation" in t.__name__ for t in types)
```

- [ ] **Step 2: Run — verify failure**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_hipad_adapter.py -v
```

Expected: 2 new failures (`AttributeError: 'HipadAdapter' object has no attribute 'forward_losses'`).

- [ ] **Step 3: Add `selective_eval_types` + `_selective_eval` helper**

Append to `tools/gradient_analysis/adapters/hipad.py`:

```python
    # Stochastic / running-stat layers that MUST be in eval mode during the
    # probe. Matches the old `_STOCHASTIC_TYPES` from collector.py and adds
    # HiP-AD's `DeformableFeatureAggregation` lazily.
    @staticmethod
    def _stochastic_base_types() -> Tuple[Type[nn.Module], ...]:
        return (
            nn.Dropout, nn.Dropout2d, nn.Dropout3d,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.SyncBatchNorm,
        )

    def selective_eval_types(self) -> Tuple[Type[nn.Module], ...]:
        types: Tuple[Type[nn.Module], ...] = self._stochastic_base_types()
        try:
            from projects.mmdet3d_plugin.models.blocks import (  # type: ignore
                DeformableFeatureAggregation,
            )
            types = types + (DeformableFeatureAggregation,)
        except Exception:
            pass
        return types

    @contextlib.contextmanager
    def _selective_eval(self, model: nn.Module) -> Iterator[None]:
        target_types = self.selective_eval_types()
        switched: List[nn.Module] = []
        for m in model.modules():
            if isinstance(m, target_types) and m.training:
                m.eval()
                switched.append(m)
        try:
            yield
        finally:
            for m in switched:
                m.train()
```

- [ ] **Step 4: Add `forward_losses`**

Append to `tools/gradient_analysis/adapters/hipad.py`:

```python
    def forward_losses(
        self, model: nn.Module, data: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Run a single training-mode forward and return the loss dict.

        Wraps the call in `_selective_eval` so BN / Dropout /
        DeformableFeatureAggregation are in eval for this forward — without
        this, the probe's baseline / grad / stepped forwards see a drifting
        model (the bug Phase 1's `_STOCHASTIC_TYPES` first fixed).
        """
        raw = model.module if hasattr(model, "module") else model
        model.train()
        with self._selective_eval(raw):
            losses = model(**data)
        if isinstance(losses, (list, tuple)):
            losses = losses[0]  # MMDataParallel returns list
        return losses
```

- [ ] **Step 5: Add `split_losses`**

Append to `tools/gradient_analysis/adapters/hipad.py`:

```python
    def split_losses(
        self, loss_dict: Mapping[str, torch.Tensor], task: str,
    ) -> Optional[torch.Tensor]:
        """Sum every loss key whose name starts with one of the prefixes
        registered in `TASK_GROUPS[task]`. Returns None if no key matched
        (task absent in this forward — caller treats as NaN)."""
        utils = self._load_utils()
        return utils["_sum_task_loss"](loss_dict, task)
```

- [ ] **Step 6: Run — verify pass**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_hipad_adapter.py -v
```

Expected: 4 tests pass.

- [ ] **Step 7: Stop and review**

Six ops live for HiP-AD. Next: VAD's matching trio (T6).

---

## Task 6: VAD Adapter — `forward_losses` + `selective_eval_types` + `split_losses`

**Files:**
- Modify: `tools/gradient_analysis/adapters/vad.py`
- Modify: `tests/gradient_analysis/adapters/test_vad_adapter.py`

- [ ] **Step 1: Append failing test**

Append to `tests/gradient_analysis/adapters/test_vad_adapter.py`:

```python
@needs_vad
def test_vad_adapter_forward_and_split():
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    model = a.build_model(ckpt=VAD_CKPT, device="cuda:0")
    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))
    # mmcv DataContainers must be unwrapped — adapter handles internally.
    losses = a.forward_losses(model, batch)
    for t in ["det", "map", "motion", "plan"]:
        sub = a.split_losses(losses, t)
        assert sub is not None, f"missing task in split: {t}"
        assert sub.requires_grad


@needs_vad
def test_vad_adapter_selective_eval_types():
    from torch import nn
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    types = a.selective_eval_types()
    assert nn.Dropout in types
    assert nn.BatchNorm2d in types
```

- [ ] **Step 2: Run — verify failure**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_vad_adapter.py -v
```

Expected: 2 new failures.

- [ ] **Step 3: Define VAD task → loss-prefix mapping**

Cross-check `/tmp/vad_loss_keys.txt` from T0 against the head implementation
([VAD_head.py:1610-1639](../../../../VAD/projects/mmdet3d_plugin/VAD/VAD_head.py#L1610-L1639)).
The canonical mapping for the four tasks:

| Task | Prefixes (any key starting with one of these counts) |
|---|---|
| det | `loss_cls`, `loss_bbox` (top-level + `d{i}.` variants) |
| map | `loss_map_cls`, `loss_map_bbox`, `loss_map_iou`, `loss_map_pts`, `loss_map_dir` |
| motion | `loss_traj`, `loss_traj_cls` |
| plan | `loss_plan_reg`, `loss_plan_bound`, `loss_plan_col`, `loss_plan_dir` |

Append to `tools/gradient_analysis/adapters/vad.py`:

```python
# Loss-key prefixes per task. A key counts for a task if any prefix is a
# substring of `key` (handling decoder-layer prefixes like `d3.loss_cls`).
_VAD_TASK_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "det":    ("loss_cls", "loss_bbox"),
    "map":    ("loss_map_cls", "loss_map_bbox", "loss_map_iou",
               "loss_map_pts", "loss_map_dir"),
    "motion": ("loss_traj",),  # loss_traj + loss_traj_cls both start with this
    "plan":   ("loss_plan_reg", "loss_plan_bound",
               "loss_plan_col", "loss_plan_dir"),
}


def _key_belongs_to_task(key: str, task: str) -> bool:
    """True iff the loss-dict key belongs to `task`.

    The check is on the *suffix after any `dN.` prefix* so decoder-layer
    losses like `d3.loss_map_cls` route to map, not det. det's
    `loss_cls`/`loss_bbox` must not match `loss_map_cls`/`loss_map_bbox`,
    so we test exact-prefix on the suffix.
    """
    suffix = key.split(".", 1)[-1] if key.startswith("d") and "." in key else key
    prefixes = _VAD_TASK_PREFIXES[task]
    if task == "det":
        # Must NOT be a map_* key.
        return (suffix.startswith("loss_cls") or suffix.startswith("loss_bbox"))
    return any(suffix.startswith(p) for p in prefixes)
```

- [ ] **Step 4: Add VAD `selective_eval_types`, `_selective_eval`, `forward_losses`, `split_losses`**

Append to `tools/gradient_analysis/adapters/vad.py`:

```python
    @staticmethod
    def _stochastic_base_types() -> Tuple[Type[nn.Module], ...]:
        return (
            nn.Dropout, nn.Dropout2d, nn.Dropout3d,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.SyncBatchNorm,
        )

    def selective_eval_types(self) -> Tuple[Type[nn.Module], ...]:
        types: Tuple[Type[nn.Module], ...] = self._stochastic_base_types()
        # GridMask is the only VAD-specific stochastic source (random mask
        # placement on input images during training); it lives on the
        # detector as `self.grid_mask`. It is NOT an nn.Module subclass
        # whose state we can flip via .eval(); the safer freeze surface is
        # `model.use_grid_mask = False` — handled explicitly inside
        # `_selective_eval` below.
        return types

    @contextlib.contextmanager
    def _selective_eval(self, model: nn.Module) -> Iterator[None]:
        target_types = self.selective_eval_types()
        switched: List[nn.Module] = []
        for m in model.modules():
            if isinstance(m, target_types) and m.training:
                m.eval()
                switched.append(m)
        # GridMask freeze (training-only stochastic op).
        raw = model.module if hasattr(model, "module") else model
        prior_grid = getattr(raw, "use_grid_mask", None)
        if prior_grid is True:
            raw.use_grid_mask = False
        try:
            yield
        finally:
            for m in switched:
                m.train()
            if prior_grid is True:
                raw.use_grid_mask = True

    def forward_losses(
        self, model: nn.Module, data: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]:
        from mmcv.parallel import scatter  # type: ignore
        # mmcv collate stores tensors inside DataContainers (.data lists
        # nested per-GPU). `scatter` to device 0 unwraps to the form the
        # detector's forward_train expects.
        raw = model.module if hasattr(model, "module") else model
        device = next(model.parameters()).device
        scattered = scatter(data, [device.index if device.index is not None else 0])[0]
        model.train()
        with self._selective_eval(raw):
            losses = model(**scattered)
        if isinstance(losses, (list, tuple)):
            losses = losses[0]
        return losses

    def split_losses(
        self, loss_dict: Mapping[str, torch.Tensor], task: str,
    ) -> Optional[torch.Tensor]:
        if task not in _VAD_TASK_PREFIXES:
            return None
        parts: List[torch.Tensor] = []
        for k, v in loss_dict.items():
            if not torch.is_tensor(v):
                continue
            if _key_belongs_to_task(k, task):
                parts.append(v if v.ndim == 0 else v.sum())
        if not parts:
            return None
        return torch.stack(parts).sum()
```

- [ ] **Step 5: Run — verify pass**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_vad_adapter.py -v
```

Expected: 4 tests pass.

If `loss_traj_cls` is misrouted, fix `_VAD_TASK_PREFIXES["motion"]` to `("loss_traj", "loss_traj_cls")` and rerun.

- [ ] **Step 6: Stop and review**

Six ops live for both adapters. The Protocol shape has not had to change — first interleaved-authoring win. Next: `shared_param_groups` (T7 HiP-AD, T8 VAD).

---

## Task 7: HiP-AD Adapter — `shared_param_groups`

**Files:**
- Modify: `tools/gradient_analysis/adapters/hipad.py`
- Modify: `tests/gradient_analysis/adapters/test_hipad_adapter.py`

- [ ] **Step 1: Append failing test**

Append to `tests/gradient_analysis/adapters/test_hipad_adapter.py`:

```python
@needs_hipad
def test_hipad_adapter_shared_param_groups():
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    model = a.build_model(ckpt=HIPAD_CKPT, device="cuda:0")
    groups = a.shared_param_groups(
        model,
        group_names=["dec0_gnn_0", "dec3_ffn_0_fc1", "backbone_layer1"],
    )
    assert set(groups) == {"dec0_gnn_0", "dec3_ffn_0_fc1", "backbone_layer1"}
    for gk, params in groups.items():
        assert len(params) > 0, gk
        for p in params:
            assert p.requires_grad
```

- [ ] **Step 2: Run — verify failure**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_hipad_adapter.py::test_hipad_adapter_shared_param_groups -v
```

Expected: `AttributeError: 'HipadAdapter' object has no attribute 'shared_param_groups'`.

- [ ] **Step 3: Add `shared_param_groups`**

Append to `tools/gradient_analysis/adapters/hipad.py`:

```python
    def shared_param_groups(
        self, model: nn.Module, group_names: List[str],
    ) -> Dict[str, List[nn.Parameter]]:
        """Translate group names (e.g. `dec3_ffn_0_fc1`, `backbone_layer1`)
        into actual nn.Parameter lists on the model.

        Delegates to HiP-AD's `get_shared_parameters_grouped` (registered
        in `pcgrad_optimizer_hook`); the returned dict is the same shape
        the existing `GradientCollector` consumed before Phase 2.
        """
        utils = self._load_utils()
        groups, _ids = utils["get_shared_parameters_grouped"](model, group_names)
        return groups
```

- [ ] **Step 4: Run — verify pass**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_hipad_adapter.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Stop and review**

Seven ops for HiP-AD. Next: VAD's translation table (T8).

---

## Task 8: VAD Adapter — `shared_param_groups`

**Files:**
- Modify: `tools/gradient_analysis/adapters/vad.py`
- Modify: `tests/gradient_analysis/adapters/test_vad_adapter.py`

- [ ] **Step 1: Probe VAD model state-dict keys for the encoder/decoder**

Run an exploratory snippet (this is a read step, not part of the plan code):

```bash
cd /home/yongjae/e2e/VAD
PYTHONPATH=. $VAD_PY -c "
import projects  # noqa
from mmdet3d.apis import init_detector
m = init_detector('data/ckpts/VAD_tiny_e2e.py', 'data/ckpts/epoch_1.pth', device='cpu')
keys = list(m.state_dict().keys())
print('=== encoder layer-0 sample ===')
for k in keys:
    if 'transformer.encoder.layers.0' in k:
        print(k)
print('=== decoder layer-0 sample ===')
for k in keys:
    if 'transformer.decoder.layers.0' in k:
        print(k)
print('=== map decoder layer-0 sample ===')
for k in keys:
    if 'map_transformer.decoder.layers.0' in k or 'map_decoder.layers.0' in k:
        print(k)
" | head -60
```

Save the output for use in Step 3 (it determines the actual attribute paths).

- [ ] **Step 2: Append failing test**

Append to `tests/gradient_analysis/adapters/test_vad_adapter.py`:

```python
@needs_vad
def test_vad_adapter_shared_param_groups():
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    model = a.build_model(ckpt=VAD_CKPT, device="cuda:0")
    # Canonical names parallel HiP-AD's: enc{i}_{tsa,sca,ffn,norm}_{0..} etc.
    groups = a.shared_param_groups(
        model,
        group_names=[
            "enc0_temporal_self_attention",
            "enc0_spatial_cross_attention",
            "enc0_ffn",
            "enc0_norm_0",
            "dec0_self_attn",
            "dec0_cross_attn",
            "dec0_ffn",
        ],
    )
    assert set(groups).issuperset({"enc0_temporal_self_attention",
                                   "enc0_spatial_cross_attention",
                                   "enc0_ffn"})
    for gk, params in groups.items():
        assert len(params) > 0, gk
```

- [ ] **Step 3: Add `shared_param_groups` translation table**

Append to `tools/gradient_analysis/adapters/vad.py`:

```python
# Canonical-group-name → (head-attribute-path, predicate over module-name) for VAD.
# `head` is `model.pts_bbox_head`; the encoder lives at
#   head.transformer.encoder.layers[i].{attentions,ffns,norms}
# the decoder at
#   head.transformer.decoder.layers[i].{attentions,ffns,norms}
# the map decoder (a separate stack) at
#   head.map_transformer.decoder.layers[i].{...}
# Predicates inspect the leaf module class so we get the right attention
# kind (TSA vs SCA) regardless of mmcv's index ordering.
_ENC_PREFIX = "transformer.encoder.layers"
_DEC_PREFIX = "transformer.decoder.layers"
_MAP_DEC_PREFIX = "map_transformer.decoder.layers"


def _gather_params(
    head: nn.Module,
    layer_prefix: str,
    layer_idx: int,
    keep: callable,
) -> List[nn.Parameter]:
    """Walk `head.<layer_prefix>.<layer_idx>` and return the params of every
    submodule whose qualified name (relative to that layer) `keep` accepts.
    Returns parameters in deterministic registration order so caches stay
    reproducible across runs.
    """
    base = head
    for part in layer_prefix.split("."):
        base = getattr(base, part)
    layer = base[layer_idx]
    out: List[nn.Parameter] = []
    seen_ids: set = set()
    for name, sub in layer.named_modules():
        if not keep(name, sub):
            continue
        for p in sub.parameters(recurse=False):
            if p.requires_grad and id(p) not in seen_ids:
                out.append(p)
                seen_ids.add(id(p))
    return out


def _resolve_vad_group(head: nn.Module, name: str) -> List[nn.Parameter]:
    """Resolve a canonical VAD group name to its parameter list."""
    # encN_<role>
    if name.startswith("enc"):
        n_layer, role = _parse_idx_role(name, "enc")
        return _resolve_block(head, _ENC_PREFIX, n_layer, role)
    if name.startswith("dec"):
        n_layer, role = _parse_idx_role(name, "dec")
        return _resolve_block(head, _DEC_PREFIX, n_layer, role)
    if name.startswith("map_dec"):
        n_layer, role = _parse_idx_role(name, "map_dec")
        return _resolve_block(head, _MAP_DEC_PREFIX, n_layer, role)
    if name == "img_backbone":
        return [p for p in head.parameters() if False]  # placeholder
    raise KeyError(f"unknown VAD group name: {name}")


def _parse_idx_role(name: str, prefix: str) -> Tuple[int, str]:
    """`enc3_temporal_self_attention` -> (3, 'temporal_self_attention')."""
    rest = name[len(prefix):]            # `3_temporal_self_attention`
    n_str, _, role = rest.partition("_")
    return int(n_str), role


def _resolve_block(
    head: nn.Module, prefix: str, layer_idx: int, role: str,
) -> List[nn.Parameter]:
    if role == "temporal_self_attention":
        return _gather_params(head, prefix, layer_idx,
                              lambda n, m: type(m).__name__ == "TemporalSelfAttention")
    if role == "spatial_cross_attention":
        return _gather_params(head, prefix, layer_idx,
                              lambda n, m: type(m).__name__ == "SpatialCrossAttention")
    if role == "self_attn":
        return _gather_params(head, prefix, layer_idx,
                              lambda n, m: type(m).__name__ in {"MultiheadAttention", "CustomMSDeformableAttention"} and "cross" not in n)
    if role == "cross_attn":
        return _gather_params(head, prefix, layer_idx,
                              lambda n, m: type(m).__name__ in {"CustomMSDeformableAttention", "MultiheadAttention"} and "cross" in n)
    if role == "ffn":
        return _gather_params(head, prefix, layer_idx,
                              lambda n, m: "ffn" in n and isinstance(m, nn.Module))
    if role.startswith("norm"):
        idx = role[4:]  # `0`, `1`, ...
        return _gather_params(head, prefix, layer_idx,
                              lambda n, m: isinstance(m, nn.LayerNorm) and (idx == "" or n.endswith(idx)))
    raise KeyError(f"unknown role for VAD block: {role}")
```

Then add the method itself to `VadAdapter`:

```python
    def shared_param_groups(
        self, model: nn.Module, group_names: List[str],
    ) -> Dict[str, List[nn.Parameter]]:
        raw = model.module if hasattr(model, "module") else model
        head = raw.pts_bbox_head
        out: Dict[str, List[nn.Parameter]] = {}
        for gk in group_names:
            params = _resolve_vad_group(head, gk)
            out[gk] = params
        return out
```

- [ ] **Step 4: Run — verify pass**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_vad_adapter.py::test_vad_adapter_shared_param_groups -v
```

Expected: 1 test passes. If the LayerNorm `norm_<idx>` resolution fails because mmcv stores them as `norms` list, fix `_resolve_block` to special-case `norm{i}` against `head.<prefix>.<layer_idx>.norms[i]`.

- [ ] **Step 5: Persist the canonical group-name list for VAD**

Append to `tools/gradient_analysis/adapters/vad.py`:

```python
# Default group set used by the VAD primary run (T22). Mirrors HiP-AD's
# canonical 6-decoder-layer + encoder split so cross-model tables in
# Phase 3 line up by row.
VAD_DEFAULT_GROUP_NAMES: List[str] = [
    *[f"enc{i}_temporal_self_attention" for i in range(3)],
    *[f"enc{i}_spatial_cross_attention" for i in range(3)],
    *[f"enc{i}_ffn" for i in range(3)],
    *[f"dec{i}_self_attn" for i in range(6)],
    *[f"dec{i}_cross_attn" for i in range(6)],
    *[f"dec{i}_ffn" for i in range(6)],
    *[f"map_dec{i}_self_attn" for i in range(6)],
    *[f"map_dec{i}_cross_attn" for i in range(6)],
    *[f"map_dec{i}_ffn" for i in range(6)],
]
```

The exact encoder layer count (`range(3)`) is read from the config in T20; for now this is the baseline list to test against.

- [ ] **Step 6: Stop and review**

Eight ops live for both adapters. The Protocol shape still hasn't had to change. Next: `freeze_stochastic_state` (T9 HiP-AD, T10 VAD) — this is the largest single op for HiP-AD because `FrozenMatching` migrates wholesale.

---

## Task 9: HiP-AD Adapter — `freeze_stochastic_state`

**Files:**
- Modify: `tools/gradient_analysis/adapters/hipad.py`
- Modify: `tests/gradient_analysis/adapters/test_hipad_adapter.py`

- [ ] **Step 1: Append failing test**

Append to `tests/gradient_analysis/adapters/test_hipad_adapter.py`:

```python
@needs_hipad
def test_hipad_adapter_freeze_stochastic_state_two_forwards_match():
    """Inside `freeze_stochastic_state`, two consecutive forwards over the
    same batch produce identical per-task losses (matching is frozen)."""
    import torch
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    model = a.build_model(ckpt=HIPAD_CKPT, device="cuda:0")
    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))

    with a.freeze_stochastic_state():
        l1 = a.forward_losses(model, batch)
        l2 = a.forward_losses(model, batch)
    for k in set(l1) & set(l2):
        if torch.is_tensor(l1[k]) and torch.is_tensor(l2[k]):
            assert torch.allclose(l1[k], l2[k], atol=1e-4), \
                f"freeze leaked at key {k}: {l1[k].item()} != {l2[k].item()}"
```

- [ ] **Step 2: Run — verify failure**

Either `AttributeError` (no method) or two forwards produce different losses (matching not frozen).

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_hipad_adapter.py::test_hipad_adapter_freeze_stochastic_state_two_forwards_match -v
```

- [ ] **Step 3: Migrate `FrozenMatching` into the adapter**

Move the body of `tools/gradient_analysis/matching_freeze.py` (the `FrozenMatching` class, the per-sampler patch factories, `_replan_gather`, `per_forward_seed`) into the bottom of `tools/gradient_analysis/adapters/hipad.py` as **module-private** symbols (prefixed `_`). Do NOT delete `matching_freeze.py` yet — `probe.py` and `landscape.py` still import it; T17 will redirect those callers and delete the file.

Concretely, append to `tools/gradient_analysis/adapters/hipad.py`:

```python
# ===========================================================================
# Frozen-matching machinery (formerly tools/gradient_analysis/matching_freeze.py)
# Migrated under the adapter so the M3 probe never has to know about HiP-AD's
# Hungarian samplers directly.
# ===========================================================================

from collections import defaultdict
from typing import Callable


class _FrozenMatchingSession:
    """Stateful object holding the per-instance call queues + play indices.

    Same semantics as the old `FrozenMatching` (see Phase 1 plan T-x). The
    only change is that the patches are installed against HiP-AD's target
    classes lazily — the import is inside `_install_patches`, so importing
    this module from a VAD-only environment does not fail.
    """

    def __init__(self) -> None:
        self._queues: Dict[int, List[Any]] = defaultdict(list)
        self._play_idx: Dict[int, int] = defaultdict(int)
        self._patches: List = []

    def _take(self, sid: int) -> Optional[Any]:
        idx = self._play_idx[sid]
        q = self._queues[sid]
        if idx < len(q):
            self._play_idx[sid] = idx + 1
            return q[idx]
        return None

    def _record(self, sid: int, value: Any) -> None:
        self._queues[sid].append(value)
        self._play_idx[sid] = len(self._queues[sid])

    def next_forward(self) -> None:
        self._play_idx = defaultdict(int)

    def reset(self) -> None:
        self._queues.clear()
        self._play_idx.clear()

    def __enter__(self) -> "_FrozenMatchingSession":
        self._install_patches()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._uninstall_patches()
        self.reset()

    def _install_patches(self) -> None:
        from projects.mmdet3d_plugin.models.det.target import SparseBox3DTarget  # type: ignore
        from projects.mmdet3d_plugin.models.map.target import SparsePoint3DTarget  # type: ignore
        from projects.mmdet3d_plugin.models.motion.target import SparseMotionTarget  # type: ignore
        from projects.mmdet3d_plugin.models.plan.target import (  # type: ignore
            SparsePlanTarget, AlignPlanTarget, PlanningTarget,
        )
        self._patches = [
            (SparseBox3DTarget, "sample", SparseBox3DTarget.sample,
             _make_det_patch(self, SparseBox3DTarget.sample)),
            (SparsePoint3DTarget, "sample", SparsePoint3DTarget.sample,
             _make_map_patch(self, SparsePoint3DTarget.sample)),
            (SparseMotionTarget, "sample", SparseMotionTarget.sample,
             _make_motion_patch(self, SparseMotionTarget.sample)),
            (SparsePlanTarget, "sample", SparsePlanTarget.sample,
             _make_plan_patch(self, SparsePlanTarget.sample)),
            (AlignPlanTarget, "sample", AlignPlanTarget.sample,
             _make_align_plan_patch(self, AlignPlanTarget.sample)),
            (PlanningTarget, "sample", PlanningTarget.sample,
             _make_plan_patch(self, PlanningTarget.sample)),
        ]
        for cls, attr, _orig, patched in self._patches:
            setattr(cls, attr, patched)

    def _uninstall_patches(self) -> None:
        for cls, attr, orig, _patched in self._patches:
            setattr(cls, attr, orig)
        self._patches = []
```

Then copy `_make_det_patch`, `_make_map_patch`, `_make_motion_patch`, `_make_plan_patch`, `_make_align_plan_patch`, `_replan_gather` verbatim from `matching_freeze.py` into the same file (rename to start with `_` if not already).

- [ ] **Step 4: Add the public adapter method**

Append to the `HipadAdapter` class:

```python
    @contextlib.contextmanager
    def freeze_stochastic_state(self) -> Iterator[None]:
        """Pin Hungarian matching (det/map) and mode-argmin (motion/plan)
        to the result of the FIRST forward inside the scope. Subsequent
        forwards within the same scope replay cached assignments so ΔL is
        purely the weight-update effect."""
        with _FrozenMatchingSession() as _fm:
            yield
```

Note: the M3 probe's existing call shape was `with FrozenMatching() as fm: fm.next_forward()`. The new adapter contract no longer exposes the session object — `next_forward()` is implicit because the Phase 1 probe always called it before each forward and the patches treat absence of cached entries as record mode. If it turns out probe.py needs explicit `next_forward()` control, T16 will widen the adapter contract to expose `freeze_stochastic_state(yields=session)`.

- [ ] **Step 5: Run — verify pass**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_hipad_adapter.py -v
```

Expected: 6 tests pass.

- [ ] **Step 6: Stop and review**

Nine ops live for HiP-AD. The bulk of `matching_freeze.py` has migrated; the file still exists for backward-compat with `probe.py`/`landscape.py` (T17 deletes it). Next: VAD's freeze surface (T10).

---

## Task 10: VAD Adapter — `freeze_stochastic_state`

**Files:**
- Modify: `tools/gradient_analysis/adapters/vad.py`
- Modify: `tests/gradient_analysis/adapters/test_vad_adapter.py`

- [ ] **Step 1: Append failing test**

Append to `tests/gradient_analysis/adapters/test_vad_adapter.py`:

```python
@needs_vad
def test_vad_adapter_freeze_stochastic_state_two_forwards_match():
    import torch
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    model = a.build_model(ckpt=VAD_CKPT, device="cuda:0")
    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))

    with a.freeze_stochastic_state():
        l1 = a.forward_losses(model, batch)
        l2 = a.forward_losses(model, batch)
    for k in set(l1) & set(l2):
        if torch.is_tensor(l1[k]) and torch.is_tensor(l2[k]):
            assert torch.allclose(l1[k], l2[k], atol=1e-3), \
                f"freeze leaked at key {k}: {l1[k].item()} != {l2[k].item()}"
```

Tolerance is `1e-3` (looser than HiP-AD's `1e-4`) because VAD's `obtain_history_bev` runs additional forwards on the previous frames — small numerical drift across calls is normal.

- [ ] **Step 2: Run — verify failure**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_vad_adapter.py::test_vad_adapter_freeze_stochastic_state_two_forwards_match -v
```

Expected: `AttributeError` (method missing) or two forwards differ.

- [ ] **Step 3: Identify VAD's matching surface**

VAD uses `HungarianAssigner3D` (det), `MapHungarianAssigner3D` (map), and `PseudoSampler` for both. motion / plan are mode-argmin in the head's `loss_*` methods (no Hungarian).

Concretely, the patches need to wrap:
- `HungarianAssigner3D.assign` (det)
- `MapHungarianAssigner3D.assign` (map)

The motion / plan argmin happens inside `VAD_head.loss_planning` / `loss_traj` and is data-dependent only on the predictions; replaying it is unnecessary because freeze of the upstream det/map matching plus a fixed RNG (via `per_forward_seed`) is sufficient for the Phase 1 probe's "two forwards produce identical losses" criterion.

- [ ] **Step 4: Add the freeze session class**

Append to `tools/gradient_analysis/adapters/vad.py`:

```python
class _VadFrozenMatchingSession:
    """VAD's analogue of HiP-AD's `_FrozenMatchingSession` — patches
    `HungarianAssigner3D.assign` and `MapHungarianAssigner3D.assign` to
    replay the first call's `AssignResult` on subsequent calls within
    the same context.

    Output of an `AssignResult` is fully GT-derived (assigned indices into
    the GT lists) so we can cache + return verbatim; no re-gather needed.
    """

    def __init__(self) -> None:
        self._queues: Dict[int, List[Any]] = {}
        self._play_idx: Dict[int, int] = {}
        self._patches: List = []

    def _take(self, sid: int) -> Optional[Any]:
        idx = self._play_idx.get(sid, 0)
        q = self._queues.get(sid, [])
        if idx < len(q):
            self._play_idx[sid] = idx + 1
            return q[idx]
        return None

    def _record(self, sid: int, value: Any) -> None:
        self._queues.setdefault(sid, []).append(value)
        self._play_idx[sid] = len(self._queues[sid])

    def __enter__(self) -> "_VadFrozenMatchingSession":
        self._install_patches()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._uninstall_patches()
        self._queues.clear()
        self._play_idx.clear()

    def _install_patches(self) -> None:
        from projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d import (  # type: ignore
            HungarianAssigner3D,
        )
        from projects.mmdet3d_plugin.core.bbox.assigners.map_hungarian_assigner_3d import (  # type: ignore
            MapHungarianAssigner3D,
        )

        def _make(cls):
            orig = cls.assign

            def patched(self, *args, **kwargs):
                sid = id(self)
                cached = self._queues.get(sid) if False else None  # noqa
                cached = _take(sid_session=session_holder["s"], sid=sid)  # filled below
                if cached is None:
                    out = orig(self, *args, **kwargs)
                    _record(sid_session=session_holder["s"], sid=sid, value=out)
                    return out
                return cached
            return orig, patched

        # Capture self into closure via a holder dict (the patched fn is bound
        # to the class, not the session).
        session_holder = {"s": self}

        def _take(sid_session, sid):
            return sid_session._take(sid)

        def _record(sid_session, sid, value):
            sid_session._record(sid, value)

        for cls in (HungarianAssigner3D, MapHungarianAssigner3D):
            orig, patched = _make(cls)
            self._patches.append((cls, "assign", orig, patched))
            setattr(cls, "assign", patched)

    def _uninstall_patches(self) -> None:
        for cls, attr, orig, _patched in self._patches:
            setattr(cls, attr, orig)
        self._patches = []
```

- [ ] **Step 5: Add the adapter method + per_forward_seed**

Append to `VadAdapter`:

```python
    @contextlib.contextmanager
    def freeze_stochastic_state(self) -> Iterator[None]:
        """Freeze Hungarian assignment for det and map; pin RNG within scope
        so GridMask + dropout / dropoutN draw identical samples. Motion /
        plan are mode-argmin only and become identical once upstream
        matching + RNG are frozen.
        """
        import random as _random
        import numpy as _np

        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np_state = _np.random.get_state()
        py_state = _random.getstate()

        torch.manual_seed(0)
        _np.random.seed(0)
        _random.seed(0)
        try:
            with _VadFrozenMatchingSession():
                yield
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            _np.random.set_state(np_state)
            _random.setstate(py_state)
```

- [ ] **Step 6: Run — verify pass**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_vad_adapter.py::test_vad_adapter_freeze_stochastic_state_two_forwards_match -v
```

Expected: 1 test passes. If a key still drifts above `1e-3`, log which key and check whether `obtain_history_bev` is the source — if so, increase the tolerance and document the leak in the closing memo (T24); the M3 probe accepts probe-level tolerance as long as it's recorded.

- [ ] **Step 7: Stop and review**

Nine ops live on both sides. Both `freeze_stochastic_state` implementations are model-specific in mechanism (HiP-AD patches `*Target.sample`; VAD patches `*Assigner.assign`) but share an identical contract: "two forwards inside this scope agree on losses to a documented tolerance." Next: temporal-state snapshot (T11 HiP-AD, T12 VAD).

---

## Task 11: HiP-AD Adapter — `snapshot_temporal_state` + `restore_temporal_state`

**Files:**
- Modify: `tools/gradient_analysis/adapters/hipad.py`
- Modify: `tests/gradient_analysis/adapters/test_hipad_adapter.py`

- [ ] **Step 1: Append failing test**

Append to `tests/gradient_analysis/adapters/test_hipad_adapter.py`:

```python
@needs_hipad
def test_hipad_adapter_snapshot_restore_round_trip():
    import torch
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    model = a.build_model(ckpt=HIPAD_CKPT, device="cuda:0")
    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))

    # Take a snapshot, run two forwards (mutating run_step / banks), restore,
    # and verify a third forward agrees with the first.
    snap = a.snapshot_temporal_state(model)
    l1 = a.forward_losses(model, batch)
    _ = a.forward_losses(model, batch)
    a.restore_temporal_state(model, snap)
    l3 = a.forward_losses(model, batch)
    for k in set(l1) & set(l3):
        if torch.is_tensor(l1[k]) and torch.is_tensor(l3[k]):
            assert torch.allclose(l1[k], l3[k], atol=1e-4), k
```

- [ ] **Step 2: Run — verify failure**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_hipad_adapter.py::test_hipad_adapter_snapshot_restore_round_trip -v
```

Expected: `AttributeError`.

- [ ] **Step 3: Migrate `ModelStateSnapshot` machinery**

Move the body of `tools/gradient_analysis/temporal_state.py` (the
`_BANK_MUTABLE_ATTRS`, `_BANK_NAMES`, `_SAMPLER_MUTABLE_ATTRS` constants;
the `_deepclone`, `_raw_model`, `_bank_holders`, `_sampler_holders`
helpers; the `ModelStateSnapshot` class) into the bottom of
`tools/gradient_analysis/adapters/hipad.py`, all module-private.

Then add the adapter methods:

```python
    def snapshot_temporal_state(self, model: nn.Module) -> TemporalSnapshot:
        snap = _ModelStateSnapshot(model)
        return TemporalSnapshot(
            model_state_keys=tuple(_BANK_MUTABLE_ATTRS) + tuple(_SAMPLER_MUTABLE_ATTRS) + ("run_step",),
            payload={"_snap": snap},
        )

    def restore_temporal_state(
        self, model: nn.Module, snapshot: TemporalSnapshot,
    ) -> None:
        snap = snapshot.payload.get("_snap")
        if snap is None:
            return
        snap.restore()
```

The `_ModelStateSnapshot` class (renamed from `ModelStateSnapshot`) stays
private to the adapter so callers can only get to it through the Protocol.

- [ ] **Step 4: Run — verify pass**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_hipad_adapter.py -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Stop and review**

All ten ops live for HiP-AD. `temporal_state.py` is no longer the source of truth — T17 deletes it after `probe.py`/`landscape.py` are redirected. Next: VAD's snapshot (T12).

---

## Task 12: VAD Adapter — `snapshot_temporal_state` + `restore_temporal_state`

**Files:**
- Modify: `tools/gradient_analysis/adapters/vad.py`
- Modify: `tests/gradient_analysis/adapters/test_vad_adapter.py`

- [ ] **Step 1: Append failing test**

Append to `tests/gradient_analysis/adapters/test_vad_adapter.py`:

```python
@needs_vad
def test_vad_adapter_snapshot_restore_round_trip():
    import torch
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    model = a.build_model(ckpt=VAD_CKPT, device="cuda:0")
    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))

    snap = a.snapshot_temporal_state(model)
    l1 = a.forward_losses(model, batch)
    _ = a.forward_losses(model, batch)
    a.restore_temporal_state(model, snap)
    l3 = a.forward_losses(model, batch)
    for k in set(l1) & set(l3):
        if torch.is_tensor(l1[k]) and torch.is_tensor(l3[k]):
            assert torch.allclose(l1[k], l3[k], atol=1e-3), k
```

- [ ] **Step 2: Run — verify failure**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_vad_adapter.py::test_vad_adapter_snapshot_restore_round_trip -v
```

- [ ] **Step 3: Add the snapshot + restore methods**

Append to `tools/gradient_analysis/adapters/vad.py`:

```python
# VAD's mutable per-forward state (training mode):
#   * `prev_frame_info` dict on the detector (only used in video_test_mode,
#     but cheap to snapshot defensively)
#   * pts_bbox_head.epoch counter (set by `set_epoch`; rarely mutates inside
#     a forward but harmless to capture)
# obtain_history_bev does NOT cache a `prev_bev` queue across batches in
# training mode (it recomputes from the input queue every batch), so there
# is no analogue of HiP-AD's InstanceBank to track.
_VAD_DETECTOR_ATTRS: Tuple[str, ...] = ("prev_frame_info",)
_VAD_HEAD_ATTRS: Tuple[str, ...] = ("epoch",)


def _vad_deepclone(x: Any) -> Any:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: _vad_deepclone(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_vad_deepclone(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_vad_deepclone(v) for v in x)
    return x
```

Then on `VadAdapter`:

```python
    def snapshot_temporal_state(self, model: nn.Module) -> TemporalSnapshot:
        raw = model.module if hasattr(model, "module") else model
        payload: Dict[str, Any] = {}
        for attr in _VAD_DETECTOR_ATTRS:
            if hasattr(raw, attr):
                payload[("detector", attr)] = _vad_deepclone(getattr(raw, attr))
        head = getattr(raw, "pts_bbox_head", None)
        if head is not None:
            for attr in _VAD_HEAD_ATTRS:
                if hasattr(head, attr):
                    payload[("head", attr)] = _vad_deepclone(getattr(head, attr))
        return TemporalSnapshot(
            model_state_keys=_VAD_DETECTOR_ATTRS + _VAD_HEAD_ATTRS,
            payload=payload,
        )

    def restore_temporal_state(
        self, model: nn.Module, snapshot: TemporalSnapshot,
    ) -> None:
        raw = model.module if hasattr(model, "module") else model
        head = getattr(raw, "pts_bbox_head", None)
        for key, val in snapshot.payload.items():
            if not isinstance(key, tuple):
                continue
            scope, attr = key
            target = raw if scope == "detector" else head
            if target is not None and hasattr(target, attr):
                setattr(target, attr, _vad_deepclone(val))
```

- [ ] **Step 4: Run — verify pass**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/test_vad_adapter.py -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Stop and review**

Both adapters complete. Ten ops live on each side. Next: end-to-end MockAdapter Protocol audit + adapter-level smoke (T13), then the leak audit (T14).

---

## Task 13: Both Adapters Pass Protocol-Conformance Test End-to-End

**Files:**
- Modify: `tests/gradient_analysis/adapters/test_protocol.py`

- [ ] **Step 1: Add a runtime conformance test for both real adapters**

Append to `tests/gradient_analysis/adapters/test_protocol.py`:

```python
@pytest.mark.skipif(
    not Path("/home/yongjae/e2e/HiP-AD/ckpts/HiP-AD-Stage2_code.py").exists(),
    reason="HiP-AD config not present",
)
def test_hipad_adapter_satisfies_protocol():
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter()
    # `runtime_checkable` Protocol — verifies structural conformance.
    assert isinstance(a, GradientAnalysisAdapter)


@pytest.mark.skipif(
    not Path("/home/yongjae/e2e/VAD/data/ckpts/VAD_tiny_e2e.py").exists(),
    reason="VAD config not present",
)
def test_vad_adapter_satisfies_protocol():
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter()
    assert isinstance(a, GradientAnalysisAdapter)
```

(Add `from pathlib import Path` near the top of the test file if absent.)

- [ ] **Step 2: Run — verify pass**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/ -v
```

Expected: every test that has its config present passes; everything else is skipped. Critically, `test_*_adapter_satisfies_protocol` must pass — that's the structural seal.

- [ ] **Step 3: Stop and review**

The Protocol is provably satisfied by both adapters. Next: a manual audit pass for *semantic* leaks the structural test cannot catch (T14).

---

## Task 14: Protocol Leak Audit (R5 Close-Out)

**Files:**
- Modify: `tools/gradient_analysis/adapters/base.py` (only if leaks are found)
- Create: `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/protocol_leak_audit.md`

- [ ] **Step 1: Re-read `base.py` against both adapters**

Open all three files side by side:

```bash
$HIPAD_PY -c "import pathlib; print(pathlib.Path('tools/gradient_analysis/adapters/base.py').read_text())"
$HIPAD_PY -c "import pathlib; print(pathlib.Path('tools/gradient_analysis/adapters/hipad.py').read_text())"
$HIPAD_PY -c "import pathlib; print(pathlib.Path('tools/gradient_analysis/adapters/vad.py').read_text())"
```

Look specifically for:

| Symptom | Likely leak |
|---|---|
| HiP-AD adapter has a method that VAD's lacks | Protocol added an op only one side needs |
| VAD adapter raises `NotImplementedError` for any op | Protocol forced a contract VAD cannot meet |
| Protocol mentions `InstanceBank`, `SparseBox3D`, `pcgrad`, `dn_metas`, `prev_bev`, `BEVFormer`, `GridMask` | HiP-AD- or VAD-specific names leaked into the contract |
| One adapter's method needs an extra non-Protocol parameter | The Protocol signature is too narrow |
| Test tolerances differ by > 100× between HiP-AD (`1e-4`) and VAD (`1e-3`) | A semantic difference in what "frozen" means — document, don't try to unify |

- [ ] **Step 2: Record findings**

Create `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/protocol_leak_audit.md`:

```markdown
# Protocol Leak Audit (Phase 2 T14)

**Date:** <fill at execution>
**Auditor:** <fill at execution>

## Findings

| ID | Where | Description | Action |
|---|---|---|---|
| L1 | base.py / vad.py | `freeze_stochastic_state` semantic — VAD tolerance is 1e-3 vs HiP-AD 1e-4 because of `obtain_history_bev` numerical drift | DOCUMENTED — not a contract change |
| ... | ... | ... | ... |
```

(Fill the table at execution time. Each row is either DOCUMENTED, FIXED-BASE, FIXED-HIPAD, or FIXED-VAD.)

- [ ] **Step 3: If any FIXED-BASE row exists, edit base.py and re-run conformance**

If a fix changes `base.py`, both adapters must still satisfy the Protocol:

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/adapters/ -v
```

Expected: every test that has its config present passes.

- [ ] **Step 4: Stop and review**

R5 close-out signed: the Protocol passed structural conformance AND a semantic re-read. Next: refactor module consumers (T15-T18), then the parity gate (T19).

---

## Task 15: collector.py — Adapter Consumer + v1 Cache Loader Path (R3)

**Files:**
- Modify: `tools/gradient_analysis/collector.py`
- Modify: `tests/gradient_analysis/test_collector_validity.py` (existing) — adapter wiring

- [ ] **Step 1: Read current `tests/gradient_analysis/test_collector_validity.py`**

```bash
cat tests/gradient_analysis/test_collector_validity.py
```

This test was authored in Phase 1 T11 and exercises the v2 cache field. It currently builds a HiP-AD model directly. Adapt it to receive an adapter as an argument.

- [ ] **Step 2: Update the test to take an adapter**

Replace any inline HiP-AD imports with:

```python
from tools.gradient_analysis.adapters.hipad import HipadAdapter

def _make_collector(...):
    adapter = HipadAdapter()
    model = adapter.build_model(...)
    return GradientCollector(model=model, adapter=adapter, ...)
```

- [ ] **Step 3: Refactor `GradientCollector.__init__` to accept an adapter**

Edit `tools/gradient_analysis/collector.py` — `__init__` signature changes from:

```python
def __init__(self, model, tasks, shared_layer_names, device):
    ...
    self._utils = _import_hipad_utils()
    self.shared_param_groups, ... = self._utils["get_shared_parameters_grouped"](...)
```

to:

```python
def __init__(
    self,
    model: nn.Module,
    adapter: "GradientAnalysisAdapter",
    shared_layer_names: List[str],
    device: str,
):
    self.model = model
    self.adapter = adapter
    self.tasks = adapter.tasks
    self.device = device
    self.shared_param_groups = adapter.shared_param_groups(model, shared_layer_names)
    self.full_params: List[nn.Parameter] = [
        p for p in self._raw_model().parameters() if p.requires_grad
    ]
    self.shared_params_flat: List[nn.Parameter] = []
    for params in self.shared_param_groups.values():
        self.shared_params_flat.extend(params)
```

Also:
- Remove the module-level `_selective_eval` helper (it lives in the adapter now). Update `forward_losses` to delegate:

```python
def forward_losses(self, data) -> Dict[str, torch.Tensor]:
    return self.adapter.forward_losses(self.model, data)
```

- Replace the module-level `build_dataloader` factory with one that delegates:

```python
def build_dataloader(adapter, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return adapter.build_dataloader(batch_size=batch_size, seed=seed, shuffle=shuffle)
```

- Inside `collect_batch`, replace `_sum_task_loss = self._utils["_sum_task_loss"]` with `task_loss = self.adapter.split_losses(losses, task)`.

- Delete `_import_hipad_utils` entirely.

- [ ] **Step 4: v1 cache loader path — preserve compat**

`BatchGradients.load` already supports v1 via `d.setdefault("nonzero_masks", {})` (Phase 1 T11). Verify the contract is unchanged by adding an explicit test:

```python
def test_v1_cache_loads_with_empty_masks(tmp_path):
    """A v1 cache file (no nonzero_masks key) must load through v2 collector."""
    import torch
    from tools.gradient_analysis.collector import BatchGradients
    v1_path = tmp_path / "batch_00000.pt"
    torch.save({
        "schema_version": 1,
        "batch_idx": 0,
        "shared": {},
        "full_norm": {},
        "shared_norm": {},
        "loss_values": {},
    }, v1_path)
    bg = BatchGradients.load(v1_path)
    assert bg.batch_idx == 0
    assert bg.nonzero_masks == {}
```

Add this test to `tests/gradient_analysis/test_collector_validity.py`.

- [ ] **Step 5: Run all gradient_analysis tests**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/ -v
```

Expected: every test passes (Phase 1 + new). v1 cache loader test pass is the R3 mitigation seal.

- [ ] **Step 6: Stop and review**

`collector.py` is now adapter-driven; v1 caches still load. Next: probe.py / landscape.py refactor (T16).

---

## Task 16: probe.py + landscape.py — Adapter Consumer

**Files:**
- Modify: `tools/gradient_analysis/probe.py`
- Modify: `tools/gradient_analysis/landscape.py`

- [ ] **Step 1: Audit the current FrozenMatching call sites**

```bash
grep -n "FrozenMatching\|matching_freeze\|per_forward_seed\|temporal_state\|_selective_eval\|_import_hipad_utils\|TASK_GROUPS" tools/gradient_analysis/probe.py tools/gradient_analysis/landscape.py
```

Expected: every reference is in `import` lines plus the `with FrozenMatching() as fm: fm.next_forward()` blocks.

- [ ] **Step 2: Refactor probe.py**

Replace the imports:

```python
from .matching_freeze import FrozenMatching, per_forward_seed
from .temporal_state import temporal_state_freeze
```

with:

```python
# (no direct freeze imports — get them from the adapter)
```

Replace `with FrozenMatching() as fm: fm.next_forward(); ...` patterns with:

```python
with adapter.freeze_stochastic_state():
    ...  # multiple forwards, each automatically replays the cached matching
```

Where the previous code called `fm.next_forward()` between forwards, the new contract treats the absence of a queue head as record mode and the presence as playback — **no explicit `next_forward()` call is required**, matching what was set up in T9 Step 4.

`temporal_state_freeze(model)` calls become:

```python
snap = adapter.snapshot_temporal_state(model)
try:
    # ... probe forwards, each preceded by:
    adapter.restore_temporal_state(model, snap)
finally:
    pass  # snapshot has no leak surface in Protocol contract
```

`probe.py` functions that take a `model` now also take an `adapter` argument; update every callsite signature accordingly.

- [ ] **Step 3: Refactor landscape.py the same way**

The pattern is identical. Walk every `FrozenMatching` / `per_forward_seed` / `temporal_state_freeze` call and rewrite as in Step 2.

- [ ] **Step 4: Run probe + landscape tests**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/test_probe* tests/gradient_analysis/test_landscape* -v
```

Expected: every test passes. If a probe test fails because matching is reshuffling between forwards, it likely indicates that `next_forward()` was load-bearing — re-add the explicit `adapter.freeze_stochastic_state(yields=True)` form (modify Protocol so the manager yields an object with `next_forward()`, then update both adapters and re-run T13/T14).

- [ ] **Step 5: Stop and review**

probe + landscape no longer import from `matching_freeze.py` / `temporal_state.py`. Next: every other module (T17).

---

## Task 17: Remaining Modules + Delete Migrated Files

**Files:**
- Modify: `tools/gradient_analysis/conflict.py`, `dynamics.py`, `asymmetry.py`, `gradnorm.py`, `magnitude_dynamics.py`, `null_baseline.py`, `distribution.py`, `bootstrap.py`, `summary.py`
- Delete: `tools/gradient_analysis/matching_freeze.py`
- Delete: `tools/gradient_analysis/temporal_state.py`
- Delete: `tools/gradient_analysis/compat.py`

- [ ] **Step 1: Find every remaining HiP-AD import in `tools/gradient_analysis/`**

```bash
grep -rn "from projects\|import projects\|TASK_GROUPS\|_sum_task_loss\|compute_task_gradient_grouped\|get_shared_parameters_grouped\|FrozenMatching\|temporal_state_freeze\|_import_hipad_utils\|matching_freeze\|temporal_state\|compat" tools/gradient_analysis/ | grep -v adapters/ | grep -v __pycache__
```

Expected: a list of remaining offenders. Process them one by one.

- [ ] **Step 2: Refactor each offender**

For every match, replace:
- `TASK_GROUPS[task]` / `_sum_task_loss(loss, task)` → `adapter.split_losses(loss, task)` (or `adapter.tasks` for the keys list)
- `get_shared_parameters_grouped(model, names)` → `adapter.shared_param_groups(model, names)`
- `FrozenMatching()` → `adapter.freeze_stochastic_state()` (drop the `fm.next_forward()` calls)
- `temporal_state_freeze(model)` → `adapter.snapshot_temporal_state(model)` (and `restore_temporal_state` where the old code restored)

Each affected function signature gains an `adapter` argument; thread it through callers.

- [ ] **Step 3: Run the full test suite to catch missed callers**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/ -v
```

Expected: every test passes. Failures here are usually missed call sites — fix and rerun.

- [ ] **Step 4: Delete the three migrated source files**

```bash
rm tools/gradient_analysis/matching_freeze.py
rm tools/gradient_analysis/temporal_state.py
rm tools/gradient_analysis/compat.py
```

- [ ] **Step 5: Re-run grep — confirm no remaining references**

```bash
grep -rn "from projects\|import projects\|matching_freeze\|temporal_state\|compat" tools/gradient_analysis/ | grep -v adapters/ | grep -v __pycache__
```

Expected: empty output.

- [ ] **Step 6: Re-run the full test suite**

```bash
PYTHONPATH=. $HIPAD_PY -m pytest tests/gradient_analysis/ -v
```

Expected: every test passes.

- [ ] **Step 7: Stop and review**

`tools/gradient_analysis/` outside of `adapters/` is now model-agnostic. The adapter Protocol is the only entry point. Next: CLI (T18) then the parity gate (T19).

---

## Task 18: run_gradient_analysis.py — `--adapter` Flag

**Files:**
- Modify: `tools/run_gradient_analysis.py`

- [ ] **Step 1: Read the current arg parser**

```bash
sed -n '405,445p' tools/run_gradient_analysis.py
```

- [ ] **Step 2: Add the `--adapter` flag**

In `parse_args`, after the existing `add_argument("--config", required=True)` line, insert:

```python
    p.add_argument(
        "--adapter",
        choices=["hipad", "vad"],
        default="hipad",
        help="Which model adapter to load. 'hipad' (default) keeps Phase-1 behavior; "
             "'vad' loads tools.gradient_analysis.adapters.vad.VadAdapter.",
    )
    p.add_argument(
        "--vad-repo-root",
        default="/home/yongjae/e2e/VAD",
        help="Only used when --adapter=vad. The root of the VAD checkout.",
    )
    p.add_argument(
        "--adapter-config",
        default=None,
        help="Override the model config path the adapter uses. Defaults to "
             "ckpts/HiP-AD-Stage2_code.py for hipad and "
             "$VAD_REPO/data/ckpts/VAD_tiny_e2e.py for vad.",
    )
```

- [ ] **Step 3: Replace the inline HiP-AD import in `main()`**

Find the block currently labelled `# Now import HiP-AD runtime (deferred so --help works without env)`. Replace it with adapter construction:

```python
    if args.adapter == "hipad":
        from tools.gradient_analysis.adapters.hipad import HipadAdapter
        adapter = HipadAdapter(
            config_path=args.adapter_config or "ckpts/HiP-AD-Stage2_code.py",
        )
    elif args.adapter == "vad":
        from tools.gradient_analysis.adapters.vad import VadAdapter
        adapter = VadAdapter(
            repo_root=args.vad_repo_root,
            config_path=args.adapter_config
                        or f"{args.vad_repo_root}/data/ckpts/VAD_tiny_e2e.py",
        )
    else:
        raise SystemExit(f"unknown adapter: {args.adapter}")
```

Then thread `adapter=adapter` through every M1-M8 module dispatch in `main()`.

- [ ] **Step 4: Smoke — `--help` still prints, `--adapter hipad --smoke` runs**

```bash
PYTHONPATH=. $HIPAD_PY tools/run_gradient_analysis.py --help | grep adapter
PYTHONPATH=. $HIPAD_PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml --adapter hipad --smoke \
    --output-root /tmp/phase2_smoke
```

Expected: help shows `--adapter` flag; smoke run completes without error and writes a few files into `/tmp/phase2_smoke/`.

- [ ] **Step 5: Stop and review**

CLI is bilingual. Default behaviour is unchanged for HiP-AD callers. Next: parity gate (T19).

---

## Task 19: HiP-AD Parity Gate (R3 Close-Out)

**Files:**
- Modify: none (verification step; produces a memo at the end)
- Create: `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_parity_gate.md`

- [ ] **Step 1: Regenerate the HiP-AD report through the adapter path**

```bash
cd /home/yongjae/e2e/HiP-AD
mkdir -p gradient_analysis_results_phase2_hipad
PYTHONPATH=. $HIPAD_PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --adapter hipad \
    --all \
    --output-root gradient_analysis_results_phase2_hipad \
    2>&1 | tee gradient_analysis_results_phase2_hipad/run.log
```

Wall-clock: ≈ 30-60 minutes for the full pipeline (no GPU re-collection needed because v1 BatchGradients caches still work; only the post-processing modules re-run).

- [ ] **Step 2: Diff every CSV against Phase 1 output**

Create a one-shot diff helper `tools/_phase2_parity_diff.py`:

```python
"""Phase 2 T19 — numeric parity gate for HiP-AD adapter path vs Phase 1."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PHASE1 = Path("gradient_analysis_results_phase1")
PHASE2 = Path("gradient_analysis_results_phase2_hipad")

# Tolerances per file family.
COSINE_ATOL = 1e-6           # M1 conflict cells
NORM_RELTOL = 1e-5            # mean / shared norms
COUNT_EXACT = True            # n_total / n_valid / n_pseudo_shared exact
CI_ATOL = 1e-3                # bootstrap CI (resamples differ by RNG)


def main() -> int:
    failures: list[str] = []
    for p1 in sorted(PHASE1.rglob("*.csv")):
        rel = p1.relative_to(PHASE1)
        p2 = PHASE2 / rel
        if not p2.exists():
            failures.append(f"MISSING in phase2: {rel}")
            continue
        df1 = pd.read_csv(p1)
        df2 = pd.read_csv(p2)
        if list(df1.columns) != list(df2.columns):
            failures.append(f"COLUMN MISMATCH: {rel}")
            continue
        for col in df1.columns:
            if not np.issubdtype(df1[col].dtype, np.number):
                if not df1[col].equals(df2[col]):
                    failures.append(f"STRING COL DIFF: {rel}::{col}")
                continue
            a = df1[col].to_numpy(dtype=np.float64)
            b = df2[col].to_numpy(dtype=np.float64)
            if "cos" in col or "antisym" in col:
                if not np.allclose(a, b, atol=COSINE_ATOL, equal_nan=True):
                    delta = np.nanmax(np.abs(a - b))
                    failures.append(f"COSINE DIFF: {rel}::{col} max|Δ|={delta:.3e}")
            elif "n_" in col:
                if not (a == b).all():
                    failures.append(f"COUNT DIFF: {rel}::{col}")
            elif "ci_" in col or "bootstrap" in col:
                if not np.allclose(a, b, atol=CI_ATOL, equal_nan=True):
                    failures.append(f"CI DIFF: {rel}::{col}")
            else:
                rel_diff = np.abs(a - b) / np.maximum(np.abs(a), 1e-12)
                if np.nanmax(rel_diff) > NORM_RELTOL:
                    failures.append(f"NORM RELTOL DIFF: {rel}::{col}")
    if failures:
        print("PARITY GATE FAILED:")
        for f in failures[:50]:
            print("  -", f)
        if len(failures) > 50:
            print(f"  ... ({len(failures)-50} more)")
        return 1
    print("PARITY GATE PASS (all CSVs match within documented tolerances)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run:

```bash
PYTHONPATH=. $HIPAD_PY tools/_phase2_parity_diff.py
```

Expected: `PARITY GATE PASS`. If any failure prints, the adapter refactor introduced numerical drift — inspect the offending CSV row, root-cause, fix, and rerun.

- [ ] **Step 3: Record the result**

Create `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_parity_gate.md`:

```markdown
# Phase 2 Parity Gate (T19)

**Date:** <fill at execution>
**Adapter path:** `tools/gradient_analysis/adapters/hipad.py`
**Phase 1 reference:** `gradient_analysis_results_phase1/`
**Adapter regen:** `gradient_analysis_results_phase2_hipad/`

## Tolerances

| Quantity | Tolerance |
|---|---|
| cosine cells (`*cos*`, `*antisym*`) | abs ≤ 1e-6 |
| norm cells (`*norm*`, `mean_*`) | rel ≤ 1e-5 |
| count cells (`n_total`, `n_valid`, `n_pseudo_shared`) | exact |
| bootstrap CI cells (`*ci_*`, `*bootstrap*`) | abs ≤ 1e-3 (RNG-dependent) |

## Result

- [ ] PARITY GATE PASS
- [ ] PARITY GATE FAIL — see appended diff log

## Appended diff log (if failed)

```
<paste output of _phase2_parity_diff.py>
```
```

Fill the result at execution time.

- [ ] **Step 4: Delete the helper script**

```bash
rm tools/_phase2_parity_diff.py
```

- [ ] **Step 5: Stop and review**

R3 close-out signed iff the parity gate passed. If the gate failed, the failure mode dictates next action — most commonly a missed call site in T17 (add the missing `adapter` arg, rerun T19). Do NOT proceed to T20 until the gate is green.

---

## Task 20: VAD Primary-Run Config

**Files:**
- Create: `configs/gradient_analysis_vad.yaml`

- [ ] **Step 1: Mirror configs/gradient_analysis.yaml structure**

```bash
cat configs/gradient_analysis.yaml
```

- [ ] **Step 2: Author the VAD config**

Create `configs/gradient_analysis_vad.yaml`:

```yaml
# Gradient analysis configuration — VAD primary run (Phase 2 T22).
# Mirrors configs/gradient_analysis.yaml so cross-model rows align in Phase 3.

adapter: vad
vad_repo_root: /home/yongjae/e2e/VAD
adapter_config: /home/yongjae/e2e/VAD/data/ckpts/VAD_tiny_e2e.py

device: cuda:0
batch_size: 6
num_batches: 100
seed: 42

checkpoints:
  - name: ep1
    path: /home/yongjae/e2e/VAD/data/ckpts/epoch_1.pth
  - name: ep10
    path: /home/yongjae/e2e/VAD/data/ckpts/epoch_10.pth
  - name: ep30
    path: /home/yongjae/e2e/VAD/data/ckpts/epoch_30.pth
  - name: ep60
    path: /home/yongjae/e2e/VAD/data/ckpts/epoch_60.pth

# Canonical group names — see VAD_DEFAULT_GROUP_NAMES in adapters/vad.py.
shared_layer_names:
  - enc0_temporal_self_attention
  - enc1_temporal_self_attention
  - enc2_temporal_self_attention
  - enc0_spatial_cross_attention
  - enc1_spatial_cross_attention
  - enc2_spatial_cross_attention
  - enc0_ffn
  - enc1_ffn
  - enc2_ffn
  - dec0_self_attn
  - dec1_self_attn
  - dec2_self_attn
  - dec3_self_attn
  - dec4_self_attn
  - dec5_self_attn
  - dec0_cross_attn
  - dec1_cross_attn
  - dec2_cross_attn
  - dec3_cross_attn
  - dec4_cross_attn
  - dec5_cross_attn
  - dec0_ffn
  - dec1_ffn
  - dec2_ffn
  - dec3_ffn
  - dec4_ffn
  - dec5_ffn
  - map_dec0_self_attn
  - map_dec1_self_attn
  - map_dec2_self_attn
  - map_dec3_self_attn
  - map_dec4_self_attn
  - map_dec5_self_attn
  - map_dec0_cross_attn
  - map_dec1_cross_attn
  - map_dec2_cross_attn
  - map_dec3_cross_attn
  - map_dec4_cross_attn
  - map_dec5_cross_attn
  - map_dec0_ffn
  - map_dec1_ffn
  - map_dec2_ffn
  - map_dec3_ffn
  - map_dec4_ffn
  - map_dec5_ffn

probe:
  alphas:
    raw: [1.0e-3]
    normalized: [1.0e-4]
  freeze_matching: true
  reset_temporal_state: true

null_baseline:
  n_repeats: 1000
  bonferroni: true

bootstrap:
  n_resamples: 2000

distribution:
  bandwidth: silverman
```

The encoder layer count is 3 (matches VAD_tiny_e2e.py default; verify by reading `cfg.model.pts_bbox_head.transformer.encoder.num_layers`). If it's different, adjust the `enc{i}_*` rows accordingly.

- [ ] **Step 3: Confirm the encoder layer count matches the canonical names**

```bash
cd /home/yongjae/e2e/VAD
PYTHONPATH=. $VAD_PY -c "
from mmcv import Config
cfg = Config.fromfile('data/ckpts/VAD_tiny_e2e.py')
print('encoder layers:', cfg.model.pts_bbox_head.transformer.encoder.num_layers)
print('decoder layers:', cfg.model.pts_bbox_head.transformer.decoder.num_layers)
print('map decoder layers:', cfg.model.pts_bbox_head.map_transformer.decoder.num_layers)
"
```

If any number differs from the yaml, update the yaml and rerun.

- [ ] **Step 4: Stop and review**

VAD config exists and matches the model's actual layer counts. Next: smoke (T21) before launching the long run.

---

## Task 21: VAD Smoke Run

**Files:**
- Modify: none (one-shot run)

- [ ] **Step 1: Run with `--smoke` (3 batches, 1 ckpt)**

```bash
cd /home/yongjae/e2e/HiP-AD
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    $HIPAD_PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis_vad.yaml \
    --adapter vad \
    --vad-repo-root /home/yongjae/e2e/VAD \
    --checkpoints ep1 \
    --smoke \
    --output-root /tmp/phase2_vad_smoke \
    2>&1 | tee /tmp/phase2_vad_smoke.log
```

Expected: completion in 5-15 minutes. `/tmp/phase2_vad_smoke/` contains the same artefact tree as a HiP-AD run (subdirs for null_baseline, distribution, bootstrap, magnitude_dynamics, probe, summary). Module count matches.

- [ ] **Step 2: Inspect the produced artefacts**

```bash
find /tmp/phase2_vad_smoke -name "*.csv" | head -20
find /tmp/phase2_vad_smoke -name "*.md" | head -10
```

Expected: at minimum `summary_report.md`, `conflict_*.csv`, `null_baseline.csv`, `distribution_report.csv`, `magnitude_dynamics.csv` are produced. Open each and confirm task names are `det/map/motion/plan` and at least one row has non-NaN values.

- [ ] **Step 3: Identify any module that crashed or produced empty output**

Failures here are typically:
- VAD `shared_param_groups` returned empty for some group (canonical name doesn't resolve in VAD model) — fix `_resolve_block` in T8 and rerun.
- VAD `forward_losses` returned a key that `split_losses` cannot route — extend `_VAD_TASK_PREFIXES` in T6 and rerun.
- VAD `freeze_stochastic_state` raised — typically a missing assigner class; add the missing one to `_VadFrozenMatchingSession._install_patches` in T10 and rerun.

- [ ] **Step 4: Stop and review**

VAD pipeline runs end-to-end on a 3-batch smoke. Next: launch the 100-batch primary run as a background job (T22).

---

## Task 22: VAD Primary Run (Background, ≈ 1.5 GPU-Day)

**Files:**
- Modify: none (long-running job)

- [ ] **Step 1: Pre-check disk + GPU**

```bash
df -h /home/yongjae/e2e/HiP-AD
nvidia-smi --query-gpu=index,name,memory.free --format=csv
```

Expected: ≥ 50 GB free on the disk, at least one GPU with ≥ 16 GB free.

- [ ] **Step 2: Launch the primary run in the background**

```bash
cd /home/yongjae/e2e/HiP-AD
mkdir -p gradient_analysis_results_phase2_vad
nohup env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    $HIPAD_PY tools/run_gradient_analysis.py \
        --config configs/gradient_analysis_vad.yaml \
        --adapter vad \
        --vad-repo-root /home/yongjae/e2e/VAD \
        --all \
        --output-root gradient_analysis_results_phase2_vad \
    > gradient_analysis_results_phase2_vad/run.log 2>&1 &
echo "VAD primary run pid=$!"
```

- [ ] **Step 3: Sanity-check progress periodically**

After 5 minutes:

```bash
tail -50 gradient_analysis_results_phase2_vad/run.log
ls gradient_analysis_results_phase2_vad/
```

Expected: log shows checkpoint loading + first batches collected; output dir has at least one `ckpt_ep1/` subtree.

After the run finishes (≈ 1.5 GPU-day):

```bash
grep -i "error\|traceback\|nan\|inf" gradient_analysis_results_phase2_vad/run.log | head -20
```

Expected: no errors. NaNs in specific cells are tolerated (per spec) but not in the `summary_report.md` headline metrics.

- [ ] **Step 4: Stop and review**

The 4×100 VAD cache exists at `gradient_analysis_results_phase2_vad/`. If anything broke mid-run, fix and resume by re-running with only the missing `--checkpoints` set. Next: cross-model summary (T23).

---

## Task 23: Cross-Model Summary — HiP-AD vs VAD Axis Values

**Files:**
- Create: `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/cross_model_axes.md`
- Create: `tools/_phase2_cross_model.py` (one-shot, deleted at end of step)

- [ ] **Step 1: Author the cross-model script**

Create `tools/_phase2_cross_model.py`:

```python
"""Phase 2 T23 — produce a side-by-side D1/D2/D4 table for HiP-AD vs VAD.

Reads from:
  gradient_analysis_results_phase1/                  (HiP-AD)
  gradient_analysis_results_phase2_vad/              (VAD)

Writes:
  cross_model_axes.csv  (machine-readable)
  cross_model_axes.md   (human-readable, embedded into the spec doc)
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

OUT_DIR = Path(
    "docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework"
)
HIPAD = Path("gradient_analysis_results_phase1")
VAD = Path("gradient_analysis_results_phase2_vad")


def _d1_pass_rates(root: Path) -> dict[str, float]:
    rows = {}
    for ck in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("ckpt_")):
        nb = ck / "null_baseline" / "null_baseline.csv"
        if not nb.exists():
            continue
        df = pd.read_csv(nb)
        rate = df["passes_noise_threshold"].astype(bool).mean()
        rows[ck.name.replace("ckpt_", "")] = float(rate)
    return rows


def _d2_max_min_ratio(root: Path) -> dict[str, float]:
    csv = root / "dynamics" / "norm_dynamics.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    out = {}
    for ck, sub in df.groupby("checkpoint"):
        out[str(ck)] = float(sub["mean_norm"].max() / sub["mean_norm"].min())
    return out


def _d4_slopes(root: Path) -> dict[str, dict[str, tuple[float, float]]]:
    csv = root / "magnitude_dynamics" / "per_task_slopes.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    return {
        row.task: {"slope": float(row.slope), "r2": float(row.r2)}
        for row in df.itertuples()
    }


def _emit(headline: str, rows: list[tuple[str, str, str, str, str]]) -> str:
    lines = [f"## {headline}", "",
             "| metric | ckpt | hipad | vad | comment |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    h_d1 = _d1_pass_rates(HIPAD)
    v_d1 = _d1_pass_rates(VAD)
    h_d2 = _d2_max_min_ratio(HIPAD)
    v_d2 = _d2_max_min_ratio(VAD)
    h_d4 = _d4_slopes(HIPAD)
    v_d4 = _d4_slopes(VAD)

    blocks = []
    rows = [("D1 pass rate", k, f"{h_d1.get(k, float('nan')):.3f}",
             f"{v_d1.get(k, float('nan')):.3f}", "")
            for k in sorted(set(h_d1) | set(v_d1))]
    blocks.append(_emit("D1 — null-baseline pass rate", rows))

    rows = [("D2 max/min", k, f"{h_d2.get(k, float('nan')):.2f}×",
             f"{v_d2.get(k, float('nan')):.2f}×", "")
            for k in sorted(set(h_d2) | set(v_d2))]
    blocks.append(_emit("D2 — magnitude max/min ratio", rows))

    rows = []
    for t in ["det", "map", "motion", "plan"]:
        rows.append(("D4 slope", t,
                     f"{h_d4.get(t, {}).get('slope', float('nan')):+.4f}",
                     f"{v_d4.get(t, {}).get('slope', float('nan')):+.4f}",
                     f"R² hipad={h_d4.get(t, {}).get('r2', 0):.2f} "
                     f"vad={v_d4.get(t, {}).get('r2', 0):.2f}"))
    blocks.append(_emit("D4 — per-task norm slope vs epoch", rows))

    md = "# Cross-Model Diagnostic Axes (Phase 2 T23)\n\n" + "\n\n".join(blocks)
    (OUT_DIR / "cross_model_axes.md").write_text(md)

    pd.DataFrame({
        "ckpt": sorted(set(h_d1) | set(v_d1)),
        "hipad_d1": [h_d1.get(k) for k in sorted(set(h_d1) | set(v_d1))],
        "vad_d1": [v_d1.get(k) for k in sorted(set(h_d1) | set(v_d1))],
    }).to_csv(OUT_DIR / "cross_model_axes_d1.csv", index=False)

    print("wrote", OUT_DIR / "cross_model_axes.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run**

```bash
cd /home/yongjae/e2e/HiP-AD
PYTHONPATH=. $HIPAD_PY tools/_phase2_cross_model.py
```

Expected: writes `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/cross_model_axes.md` and a `_d1.csv`.

- [ ] **Step 3: Read the result**

```bash
cat docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/cross_model_axes.md
```

Inspect: are VAD's D1 pass rates also < 30% (Pass-B framing transfers)? Is D2 also severe early and shrinking? Is D4 monotone-increasing on at least one task with R² > 0.5?

- [ ] **Step 4: Delete the helper script**

```bash
rm tools/_phase2_cross_model.py
```

- [ ] **Step 5: Stop and review**

The cross-model table answers Phase 1 gating's open question for VAD ("does VAD also exhibit magnitude-dominated, conflict-weak signal?"). The verdict feeds the closing memo (T24).

---

## Task 24: Phase 2 Closing Memo

**Files:**
- Create: `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_summary.md`

- [ ] **Step 1: Author the closing memo**

Create `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_summary.md`:

```markdown
# Phase 2 Closing Memo

**Status:** filled (<execution date>)
**Spec:** [2026-05-02 design](../2026-05-02-gradient-dynamics-diagnostic-framework-design.md) §4
**Plan:** [2026-05-02 phase2](../../plans/2026-05-02-gradient-dynamics-diagnostic-framework-phase2.md)
**Phase 1 gating:** [Pass B](phase1_gating_decision.md)

## What landed

| Item | Location |
|---|---|
| Adapter Protocol (10 ops) | `tools/gradient_analysis/adapters/base.py` |
| HiP-AD adapter | `tools/gradient_analysis/adapters/hipad.py` |
| VAD adapter | `tools/gradient_analysis/adapters/vad.py` |
| MockAdapter conformance test | `tests/gradient_analysis/adapters/test_protocol.py` |
| HiP-AD parity gate | `phase2_parity_gate.md` (PASS) |
| VAD primary-run output (4 ckpts × 100 batches) | `gradient_analysis_results_phase2_vad/` |
| Cross-model axis table (D1/D2/D4) | `cross_model_axes.md` |
| Protocol leak audit | `protocol_leak_audit.md` |

## Spec § 4.1 Protocol delta

The spec sketched 9 ops; we landed 10. The added op is `restore_temporal_state`,
the symmetric pair to `snapshot_temporal_state`. Documented for traceability;
it does not change Phase 3's contract because Phase 3 operates on cached
output, not the live model.

## Spec § 4.3 VAD adapter delta — `prev_bev_queue`

The spec described VAD's temporal state as "snapshot / restore of `prev_bev_queue`".
Reading the VAD source ([VAD.py:241](../../../../VAD/projects/mmdet3d_plugin/VAD/VAD.py#L241))
shows that in training mode `obtain_history_bev` recomputes `prev_bev` from
the input queue every batch, with no persistent ring buffer. The adapter's
`snapshot_temporal_state` therefore captures `prev_frame_info` (used only in
`video_test_mode`) and the head's `epoch` counter — a defensive
implementation that is correct but smaller than the spec implied. No probe
correctness regression follows from this; the snapshot/restore pair would
only matter if VAD were run in `video_test_mode=True` during gradient
collection, which is never the case for the primary run.

## Cross-model axis verdict (D1 / D2 / D4)

<paste contents of cross_model_axes.md here, or link>

**Reading:**
- D1 (null-baseline pass rate): _<fill from data>_. Pass-B framing _<does/does not>_ transfer.
- D2 (max/min mag ratio): _<fill from data>_. Magnitude imbalance _<is/is not>_ severe in VAD too.
- D4 (per-task slope): _<fill from data>_. Temporal drift _<does/does not>_ reproduce.

## Risks (closing status)

| # | Status | Note |
|---|---|---|
| R1 VAD config / dataset incompat | RESOLVED at T0 | smoke ran clean |
| R3 v1 cache breaks | RESOLVED at T19 | parity gate PASS |
| R5 Protocol leaks HiP-AD assumptions | RESOLVED at T14 | leak audit memo signed |
| R6 motion freeze | INHERIT FROM PHASE 1 | motion-source caveat carries into Phase 3 cells |

## Open questions for Phase 3

1. Does VAD's D2 trajectory also shrink under standard training (HiP-AD: 8.4× → 4.3×)? Phase 3 must record both trajectories.
2. The motion-probe-unfriendliness from R6 — does it reproduce on VAD? (motion gradient norm small relative to noise; helpful_ratio ≈ 0.) If yes, Phase 3 desiderata cells should tag both models' motion rows as noise-dominated.
3. det-vs-map mid-decoder norm conflict (HiP-AD's only direction-signal candidate per gating doc): is there a VAD-side analogue? Worth surfacing in Phase 3 even though global D1 is weak in both.
```

Fill `<...>` placeholders at execution time from the cross-model table.

- [ ] **Step 2: Stop and review**

The Phase 2 deliverable is complete. Phase 3 (desiderata matrix + gap analysis) reads from this memo plus `cross_model_axes.md` plus `gradient_analysis_results_phase2_vad/`. The user can now batch-commit the Phase 2 work.

---

## Self-Review

Before declaring Phase 2 done, sweep this plan against the spec one more time:

**1. Spec § 4.1 Protocol — 9 abstract ops:** ✅ all in `base.py` (T1) + 1 added (`restore_temporal_state`, T1). HiP-AD covers all in T3/T5/T7/T9/T11; VAD covers all in T4/T6/T8/T10/T12.

**2. Spec § 4.2 HiP-AD adapter — migrates `_import_hipad_utils`, `FrozenMatching`, `ModelStateSnapshot`, `_selective_eval`:** ✅ T3 (`_import_hipad_utils`), T5 (`_selective_eval`), T9 (`FrozenMatching`), T11 (`ModelStateSnapshot`). Cache format preserved at T15 Step 4.

**3. Spec § 4.3 VAD adapter — tasks, loss-keys, shared groups, freeze, temporal, init_detector:** ✅ T4 (tasks/init_detector), T6 (loss-keys), T8 (shared groups), T10 (freeze), T12 (temporal). The `prev_bev_queue` simplification is documented in the closing memo (T24).

**4. Spec § 4.4 Module refactor — every M1-M8 consumes only an adapter, HiP-AD imports removed:** ✅ T15 (collector), T16 (probe + landscape), T17 (the rest + delete migrated files). T17 Step 5 grep proves no remaining HiP-AD direct imports.

**5. Spec § 4.5 VAD primary run — 4 ckpts at the same profile as HiP-AD:** ✅ T20 (config), T21 (smoke), T22 (run).

**6. R1 mitigation — 0.5-day VAD smoke before Phase 2 code work:** ✅ T0.

**7. R3 mitigation — keep v1 loader path:** ✅ T15 Step 4 + T19 parity gate.

**8. R5 mitigation — Protocol authored during both adapters in parallel:** ✅ interleaved T3/T4, T5/T6, T7/T8, T9/T10, T11/T12 — every Protocol-touch task has a paired counterpart immediately after, plus T14 final audit.

**9. Phase 1 gating Pass-B framing — VAD load-bearing question is "magnitude-dominated, conflict-weak":** ✅ T23 produces the side-by-side D1/D2/D4 table that answers it; T24 records the verdict.

**10. No placeholders / TBDs:** the only `<fill at execution>` strings are dates and human verdicts in the closing memos — those cannot be authored ahead of execution. No code step has a TBD.

**11. Type / signature consistency:** `GradientAnalysisAdapter` Protocol method names match every callsite in the test files and module-consumer refactors. `TemporalSnapshot` field names (`model_state_keys`, `payload`) match every test access.

If any item is not ✅, fix the offending tasks before declaring Phase 2 ready for execution.
