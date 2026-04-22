# HiP-AD Stage2 GradNorm Integration — Design Spec

- **Date**: 2026-04-22
- **Target config**: `projects/configs/experiments/E9_E2_E1_stage2_18ep_GN.py`
- **Base config**: `projects/configs/experiments/E2_E1_stage2_18ep.py`
- **Reference ckpt**: `ckpts/nusc_stage1_e1.pth` (from `E1_stage1_12ep.py`)
- **Data**: `data_nusc/infos/nuscenes_infos_train_1_3_seed0.pkl` (1/3 seed0 split)
- **Baseline for comparison**: 사용자 보유 1/3 seed0 E2 ckpt (재학습 없음)
- **Related work (in-repo)**: `docs/gradient_conflict_analysis.md`, `docs/gradient_conflict_results.md`

---

## 1. Overview & Scope

### 1.1 목적
HiP-AD stage2 학습에 GradNorm (Chen et al., *Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*, ICML 2018)을 적용한다. 목적은 아래의 우선순위 순이다.

1. **Primary (A)**: task별 gradient magnitude의 동적 balancing으로 학습 안정성 및 성능 (det mAP/NDS, map mAP, motion minADE, plan L2/collision, ego L1) 향상 가능성 확인.
2. **Secondary (C)**: epoch-wise task weight dynamics `w_i(t)`, shared gradient norm `||G_W^{(i)}||`, relative inverse training rate `r̃_i(t)` 를 기록하여 추후 분석 자산으로 확보.
3. **Downstream (D)**: PCGrad·distillation과 결합 가능한 깨끗한 task-weight foundation 확보.

기존 연구 배경 (`docs/gradient_conflict_*.md`): map=5.79 vs motion=43.34 수준의 magnitude imbalance가 관측되어 있으나 "conflict"는 noise 수준일 가능성. GradNorm은 본 imbalance를 magnitude 축에서 직접 다루는 방법으로 채택한다.

### 1.2 Out of scope
- Stage1 config (`E1_stage1_12ep.py`) 수정
- PCGrad 결합 (후속 실험)
- Depth loss를 GradNorm에 포함시키는 옵션
- Hyperparameter sweep 자동화

### 1.3 Final design decisions (요약)

| 항목 | 결정 |
|---|---|
| GradNorm 대상 tasks | `det`, `map`, `motion`, `plan`, `ego` (5개) |
| 제외 task | `depth` (neck에서 분기, decoder shared W에 gradient 0) |
| Shared W | 모든 6개 decoder layer (1 single + 5 temporal) 의 `AsymmetricFFN` 최종 `nn.Linear` `weight` concat |
| `w_i` parameterization | `nn.Parameter`, 별도 `Adam(lr₂)` optimizer |
| 초기 `w_i` | 기존 config 비율: det=4.25, map=11.0, motion=0.4, plan=1.5, ego=1.0 (sum=18.15) |
| α (restoring force) | 1.5 |
| `lr₂` (w_i optimizer) | 2.5e-2 |
| `update_every` | 1 (매 iter) |
| `update_after_step` | 500 (= `warmup_iters`) |
| `pivot_warmup_steps` | 50 (`L_i(0)`은 iter 500~549 평균으로 freeze) |
| Target norm 부호 | `gw_avg * r̃^α` (논문·LucasBoTang 방식) |
| Sum normalization | `T = init_weights.sum() = 18.15`로 고정 (sum-to-init-sum) |
| Positivity safety | `clamp_min=1e-4` 후 sum renormalize |
| fp16 safety | GradNorm의 gradient norm 계산만 fp32 승격 |
| DDP safety | `w` update 후 `all_reduce(mean)` |
| 학습 epoch | 18 (기존 E9 그대로) |
| 모델 optimizer | 불변 (AdamW lr=1e-4, backbone lr_mult=0.5, grad_clip max_norm=25) |
| Baseline 실험 | 사용자 보유 E2(1/3 seed0) ckpt/log 재활용 |

---

## 2. Module structure

### 2.1 파일 변경 요약
```
projects/mmdet3d_plugin/
├── core/
│   └── gradnorm/                         NEW DIR
│       ├── __init__.py                   NEW
│       └── weighter.py                   NEW (GradNormLossWeighter)
├── models/
│   ├── sparse_detector.py                MODIFY (__init__, forward_train, 2 helpers)
│   └── sparse_onedecoder.py              MODIFY (1 helper method 추가)
projects/configs/experiments/
└── E9_E2_E1_stage2_18ep_GN.py            REPLACE (placeholder → 실 config)
```

### 2.2 No-change 목록
- `tools/train.py`, `tools/dist_train.sh`
- `projects/mmdet3d_plugin/apis/train.py`, `apis/mmdet_train.py` (mmcv `Fp16OptimizerHook` / 기존 hook 흐름 유지)
- 기존 loss / head / neck / backbone 파일 전체
- `projects/configs/experiments/E1_stage1_12ep.py`

### 2.3 Responsibility split

- **`core/gradnorm/weighter.py`** — HiP-AD 독립. 순수 GradNorm 알고리즘 (state, forward, renormalization, DDP safety).
- **`models/sparse_detector.py`** — `forward_train` 내에서 task-level aggregation, rename, `GradNormLossWeighter` 호출.
- **`models/sparse_onedecoder.py`** — `collect_ffn_last_fc_params()` helper 추가 (shared W 수집). 기존 `loss` 로직은 건드리지 않음.
- **config** — GradNorm 하이퍼파라미터 주입.

---

## 3. `GradNormLossWeighter` 세부 구현

### 3.1 Class state
```python
class GradNormLossWeighter(nn.Module):
    # 생성자 상수
    task_names: list[str]
    T_tasks: int                    # = len(task_names) (logging 용도)
    alpha: float
    clamp_min: float
    update_after_step: int
    pivot_warmup_steps: int
    update_every: int

    # Parameter
    w: nn.Parameter                 # [T_tasks], init = init_weights

    # 내부 optimizer (state_dict 별도 저장)
    w_optimizer: torch.optim.Adam

    # buffers (state_dict 자동 저장)
    L0:           [T_tasks]         torch.float32
    L0_running:   [T_tasks]         torch.float32
    step_counter: scalar(int64)
    pivot_ready:  scalar(bool)
    init_w_sum:   scalar(float32)   # = sum(init_weights), sum-to-init-sum renorm target
```

### 3.2 생성자
```python
def __init__(
    self,
    task_names: list[str],
    init_weights: list[float],
    alpha: float = 1.5,
    lr: float = 2.5e-2,
    update_after_step: int = 500,
    pivot_warmup_steps: int = 50,
    update_every: int = 1,
    clamp_min: float = 1e-4,
):
    assert len(task_names) == len(init_weights) >= 2
    assert alpha >= 0 and lr > 0 and update_after_step >= 0
    assert pivot_warmup_steps >= 1 and update_every >= 1
    assert all(w > 0 for w in init_weights)

    self.task_names = list(task_names)
    self.T_tasks = len(task_names)
    self.alpha = alpha
    self.clamp_min = clamp_min
    self.update_after_step = update_after_step
    self.pivot_warmup_steps = pivot_warmup_steps
    self.update_every = update_every

    self.w = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
    self.w_optimizer = torch.optim.Adam([self.w], lr=lr)

    self.register_buffer("L0",           torch.zeros(self.T_tasks))
    self.register_buffer("L0_running",   torch.zeros(self.T_tasks))
    self.register_buffer("step_counter", torch.zeros((), dtype=torch.long))
    self.register_buffer("pivot_ready",  torch.zeros((), dtype=torch.bool))
    self.register_buffer(
        "init_w_sum",
        torch.tensor(float(sum(init_weights)), dtype=torch.float32),
    )
```

### 3.3 Forward (매 iter 호출)
```python
def forward(
    self,
    task_losses: dict[str, Tensor],      # {name: scalar, requires_grad=True, fp16 OK}
    shared_params: list[nn.Parameter],   # decoder FFN 최종 Linear weight 들
) -> tuple[Tensor, dict[str, float]]:

    device = next(iter(task_losses.values())).device
    step = int(self.step_counter.item())
    self.step_counter += 1

    # 1) 고정 순서로 stack
    L = torch.stack([task_losses[n] for n in self.task_names])     # [T_tasks]
    w_det = self.w.detach()

    # 2) 실제 backward 대상 (depth 와는 별도 합산)
    weighted_loss = (w_det * L).sum()

    # 3) logging base
    log = {f"w_{n}": float(w_det[i].item())
           for i, n in enumerate(self.task_names)}
    log["step"] = step

    # ── Safety 2 gate 1: warmup 구간 ──
    if step < self.update_after_step:
        log["phase_id"] = 0         # "warmup"
        return weighted_loss, log

    # ── Safety 2 gate 2: pivot 누적 구간 ──
    if not self.pivot_ready:
        self.L0_running += L.detach().float()
        pivot_step = step - self.update_after_step + 1
        if pivot_step >= self.pivot_warmup_steps:
            self.L0.copy_(self.L0_running / float(self.pivot_warmup_steps))
            # pivot을 freeze하는 시점에 rank 간 mean 동기화 (Section 3.6 참조)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(self.L0, op=torch.distributed.ReduceOp.SUM)
                self.L0.mul_(1.0 / torch.distributed.get_world_size())
            self.pivot_ready.fill_(True)
            log["phase_id"] = 2     # "pivot_set"
        else:
            log["phase_id"] = 1     # "pivot_accum"
        return weighted_loss, log

    # ── update_every gate ──
    if (step - self.update_after_step - self.pivot_warmup_steps) % self.update_every != 0:
        log["phase_id"] = 4         # "gn_skip_every"
        return weighted_loss, log

    # ── GradNorm main ──
    grad_norms = []
    for i, Li in enumerate(L):
        grads = torch.autograd.grad(
            outputs=self.w[i] * Li,
            inputs=shared_params,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )
        # Safety 1: fp32 승격 후 norm
        g_flat = torch.cat([g.float().flatten() for g in grads])
        grad_norms.append(g_flat.norm(p=2))
    gw = torch.stack(grad_norms)

    # relative inverse training rate r̃_i
    loss_ratio = L.detach().float() / self.L0.clamp_min(1e-8)
    rt = loss_ratio / loss_ratio.mean().clamp_min(1e-8)

    gw_avg = gw.mean().detach()
    target = (gw_avg * (rt ** self.alpha)).detach()

    gn_loss = (gw - target).abs().sum()

    self.w_optimizer.zero_grad(set_to_none=True)
    gn_loss.backward()
    self.w_optimizer.step()

    # Safety 3: clamp + sum-to-init-sum renorm
    with torch.no_grad():
        self.w.data.clamp_(min=self.clamp_min)
        self.w.data.mul_(self.init_w_sum / self.w.data.sum().clamp_min(1e-8))
        # DDP weight drift 제거
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.w.data, op=torch.distributed.ReduceOp.SUM)
            self.w.data.mul_(1.0 / torch.distributed.get_world_size())

    # logging 확장
    for i, n in enumerate(self.task_names):
        log[f"grad_norm_{n}"]  = float(gw[i].item())
        log[f"rt_{n}"]         = float(rt[i].item())
        log[f"L_{n}"]          = float(L[i].item())
        log[f"L0_{n}"]         = float(self.L0[i].item())
        log[f"weighted_L_{n}"] = float((w_det[i] * L[i]).item())
    log["gn_loss"]  = float(gn_loss.item())
    log["phase_id"] = 3             # "gn_active"
    return weighted_loss, log
```

### 3.4 Phase ID encoding (wandb numeric용)

| ID | Phase | 조건 |
|---|---|---|
| 0 | `warmup`        | `step < update_after_step` |
| 1 | `pivot_accum`   | `update_after_step ≤ step < update_after_step + pivot_warmup_steps` |
| 2 | `pivot_set`     | pivot freeze된 iter (1회만 찍힘) |
| 3 | `gn_active`     | GradNorm main이 실제 실행된 iter |
| 4 | `gn_skip_every` | `update_every>1`일 때 skip한 iter |

### 3.5 Checkpoint 처리
- `w` 는 `nn.Parameter` → mmcv save/load 사이클에 자동 포함.
- `L0`, `L0_running`, `step_counter`, `pivot_ready`, `init_w_sum` 는 buffer → 자동 포함.
- `w_optimizer.state_dict()` 는 **별도 필드**로 checkpoint hook에서 저장. Stage1→Stage2 초기 load 시에는 기존 ckpt에 없으므로 mmcv `strict=False`(기본)로 warning만 발생하고 새 state로 초기화.
- 구현 세부: SparseDetector에 `get_gradnorm_state()` / `load_gradnorm_state()` helper 추가. Checkpoint hook에서 `runner.model.module.get_gradnorm_state()` 호출하여 `gradnorm_aux.pth`를 함께 저장. Resume 시 `load_gradnorm_state(torch.load(...))` 호출 (implementation plan 단계에서 확정).

### 3.6 Distributed training 정확성
- 각 rank는 local batch 기준으로 GradNorm forward를 돌리므로 `w.grad` 가 rank별로 달라진다 → 위 3.3 말미의 `all_reduce` 로 `w` 를 동기화.
- `L0_running` / `L0` 는 각 rank별 고유 값으로 남아도 큰 문제는 없지만, 엄밀성을 위해 `L0 freeze` 직전에도 `all_reduce(mean)` 한 번 수행 (implementation에 포함).

---

## 4. SparseDetector integration

### 4.1 Head output → task scalar aggregation (prefix rule)

`SparseOneDecoder.loss` 는 `combine_layer_loss=True` (default) 하에서 per-task **aggregated key**만 출력한다. Stage2 non-distill 세팅에서 생성되는 key:

| Task | Key 예 |
|---|---|
| det | `det_loss_cls`, `det_loss_box`, `det_loss_cns`, `det_loss_yns` |
| map | `map_loss_cls`, `map_loss_line` |
| motion | `motion_loss_cls`, `motion_loss_reg` |
| plan | `plan_loss_temp_cls`, `plan_loss_temp_reg` (anchor_types=[("temp","2hz")]) |
| ego | `ego_loss_status` |
| depth | `loss_dense_depth` (SparseDetector가 추가) |

Aggregation:
```python
TASK_PREFIXES = {"det":"det_loss_", "map":"map_loss_", "motion":"motion_loss_",
                 "plan":"plan_loss_", "ego":"ego_loss_"}
# L_i = Σ v for k,v in output.items() if k.startswith(TASK_PREFIXES[i])
```

Depth와 기타(`scenes_loss_*` 등)는 aggregation에서 제외되어 `_parse_losses`에 그대로 전달.

### 4.2 Rename rule (이중 집계 차단)

mmcv `_parse_losses` 는 `'loss' in key` 인 항목만 합산한다. GradNorm path에서는 원래 per-task loss 가 `loss_gradnorm_total`로 흡수되므로 **원본 key를 monitor 로 rename** 하여 sum에서 배제한다.

```python
# "det_loss_cls" → "monitor_det_cls"
new_key = "monitor_" + key.replace("_loss_", "_", 1)
```
- `.detach()` 로 graph를 끊어 이중 backward 방지.
- `loss_dense_depth` 등 prefix 미매치 항목은 원본 그대로 유지.

### 4.3 SparseDetector diff (의사코드)

```python
# NEW kw: gradnorm (Optional[dict])
class SparseDetector(BaseModule):
    def __init__(self, ..., gradnorm: Optional[dict] = None, **kwargs):
        super().__init__(...)
        ...
        if gradnorm is not None:
            from projects.mmdet3d_plugin.core.gradnorm import GradNormLossWeighter
            self.gradnorm = GradNormLossWeighter(**gradnorm)
        else:
            self.gradnorm = None

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        if "fut_img" in data and self.training:
            data = self.extract_fut_feat(img, feature_maps, data)
        model_outs = self.head(img, feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(depths, data["gt_depth"])

        if self.gradnorm is None:
            return output                              # baseline bit-identical

        task_names = self.gradnorm.task_names
        task_prefixes = {t + "_loss_" for t in task_names}

        task_losses = self._aggregate_task_losses(output, task_names)
        shared_params = self.head.onedecoder_head.collect_ffn_last_fc_params()
        weighted, gn_log = self.gradnorm(task_losses, shared_params)

        output = self._rename_as_monitor(output, task_prefixes)
        output["loss_gradnorm_total"] = weighted
        device = weighted.device
        for k, v in gn_log.items():
            output[f"gn_{k}"] = torch.as_tensor(v, device=device, dtype=torch.float32)

        # optional: rank0 CSV dump (Section 6.3)
        self._maybe_dump_gn_csv(gn_log)
        return output

    @staticmethod
    def _aggregate_task_losses(output, task_names):
        task_losses = {}
        for t in task_names:
            prefix = f"{t}_loss_"
            parts = [v for k, v in output.items() if k.startswith(prefix)]
            assert len(parts) > 0, f"No loss key with prefix '{prefix}'"
            task_losses[t] = torch.stack(parts).sum()
        return task_losses

    @staticmethod
    def _rename_as_monitor(output, task_prefixes):
        new = {}
        for k, v in output.items():
            matched = next((p for p in task_prefixes if k.startswith(p)), None)
            if matched is not None:
                nk = "monitor_" + k.replace("_loss_", "_", 1)
                new[nk] = v.detach() if torch.is_tensor(v) else v
            else:
                new[k] = v
        return new
```

### 4.4 `SparseOneDecoder.collect_ffn_last_fc_params()`

```python
def collect_ffn_last_fc_params(self) -> list[nn.Parameter]:
    """operation_order 의 'ffn' 위치 AsymmetricFFN 에서
       마지막 nn.Linear.weight 를 순서대로 반환."""
    from projects.mmdet3d_plugin.models.blocks import AsymmetricFFN
    params = []
    for op, module in zip(self.operation_order, self.layers):
        if op != "ffn":
            continue
        assert isinstance(module, AsymmetricFFN)
        last_linear = None
        for sub in reversed(list(module.layers.children())):
            if isinstance(sub, nn.Linear):
                last_linear = sub
                break
        assert last_linear is not None
        params.append(last_linear.weight)
    assert len(params) == self.operation_order.count("ffn")
    return params
```

Stage2 config 기준 결과: **6개 Parameter tensor** (shape 각 `[256, 1024]`).

### 4.5 최종 total loss 검산
GradNorm path 활성 시 `_parse_losses` 가 sum하는 항목:
```
total = loss_gradnorm_total  +  loss_dense_depth
      = Σ_{i in 5 tasks} w_i.detach() * L_i  +  L_depth
```

Baseline(`gradnorm=None`) 의 sum:
```
total = det_loss_cls + det_loss_box + det_loss_cns + det_loss_yns
      + map_loss_cls + map_loss_line
      + motion_loss_cls + motion_loss_reg
      + plan_loss_temp_cls + plan_loss_temp_reg
      + ego_loss_status
      + loss_dense_depth
```
→ iter 0 에서 `w_i = {4.25, 11.0, 0.4, 1.5, 1.0}` 이고 각 `L_i`가 이미 내부 weight를 포함한 sum이므로, **초기 gradient 방향은 baseline과 동일**.

---

## 5. E9 config delta

### 5.1 파일 헤더 주석
```python
# ──────────────────────────────────────────────────────────────
# E9: Stage2 학습에 GradNorm 적용 (E1 ckpt 기반, 1/3 seed0 data)
# Base : E2_E1_stage2_18ep.py
# Delta: `model.gradnorm` 추가 외 동일
# Tasks (5): det, map, motion, plan, ego   (depth 제외)
# Shared W : 6개 decoder FFN 최종 FC weight concat
# α=1.5, lr_w=2.5e-2, warmup=500 iter, pivot N=50 iter
# ──────────────────────────────────────────────────────────────
```

### 5.2 `model` dict 변경 (유일한 기능적 변경)
```python
model = dict(
    type="SparseDetector",
    ...
    head=dict(...),                               # 불변

    # NEW
    gradnorm=dict(
        task_names=["det", "map", "motion", "plan", "ego"],
        init_weights=[4.25, 11.0, 0.4, 1.5, 1.0],
        alpha=1.5,
        lr=2.5e-2,
        update_after_step=500,
        pivot_warmup_steps=50,
        update_every=1,
        clamp_min=1e-4,
    ),
)
```

### 5.3 불변 항목 (baseline 공정성)

| 항목 | 값 |
|---|---|
| `load_from` | `/home/yongjae/e2e/HiP-AD/ckpts/nusc_stage1_e1.pth` |
| `ann_file` | `data_nusc/infos/nuscenes_infos_train_1_3_seed0.pkl` |
| `num_epochs` / `batch_size` / `num_gpus` | 18 / 6 / 2 |
| 모델 `optimizer` | `AdamW(lr=1e-4, weight_decay=0.001, paramwise_cfg{img_backbone.lr_mult=0.5})` |
| `optimizer_config` | `grad_clip=dict(max_norm=25, norm_type=2)` |
| `lr_config` | CosineAnnealing, `warmup_iters=500`, `warmup_ratio=1/3`, `min_lr_ratio=1e-3` |
| `fp16` | `loss_scale=32.0` |
| `work_dir` | `work_dirs/exp/E9_E2_E1_stage2_18ep_GN` |
| `wandb_name` | `E9_E2_E1_stage2_18ep_GN` |
| `evaluation.eval_mode` | `with_det=True, with_map=True, with_motion=True, with_planning=True` |

---

## 6. Logging & diagnostics

### 6.1 Wandb metric schema

**학습 loss (`_parse_losses` 대상)**
- `loss_gradnorm_total` — `Σ w_i.detach() * L_i` (5 tasks)
- `loss_dense_depth` — depth aux

**Per-loss monitor (backward 제외, log_vars에만)**
- `monitor_det_{cls,box,cns,yns}`
- `monitor_map_{cls,line}`
- `monitor_motion_{cls,reg}`
- `monitor_plan_temp_{cls,reg}` (+ 존재 시 col/dir/bound/status)
- `monitor_ego_status`

**GradNorm diagnostics (`gn_*`)**
- `gn_step`, `gn_phase_id` (0–4)
- `gn_w_{task}` — 현재 task weight (매 iter)
- `gn_grad_norm_{task}` — `||G_W^{(i)}||` (gn_active 시만)
- `gn_rt_{task}` — relative inverse training rate r̃_i
- `gn_L_{task}` — 당회 iter raw task-scalar loss
- `gn_L0_{task}` — pivot (pivot_ready 이후)
- `gn_weighted_L_{task}` — `w_i * L_i` (task별 effective contribution to total)
- `gn_gn_loss` — GradNorm 내부 loss `Σ|gw − target|`

**영향 분석용 파생 metric (wandb에 바로 push, SparseDetector 또는 후처리에서 계산)**
- `gn_w_ratio_{task}` = `w_i(t) / w_i(0)` — 시간에 따른 weight 증감 비율 (축적 다이나믹스)
- `gn_contribution_frac_{task}` = `w_i * L_i / Σ_j (w_j * L_j)` — 총 weighted loss 중 task i 차지 비율
- `gn_grad_norm_share_{task}` = `||G_W^{(i)}|| / Σ_j ||G_W^{(j)}||` — 공유 W에 대한 task별 gradient 기여 비율
- `gn_loss_progress_{task}` = `L_i / L0_i` — pivot 대비 학습 진행도 (r̃ normalize 전 raw 값)

### 6.2 log interval
- 기존 `log_config.interval = 50` iter 유지.
- GradNorm 은 매 iter 업데이트되지만 wandb push는 50 iter 단위 sampling → 기존 부하와 동일.

### 6.3 CSV dump (rank 0 only, **ON**)

- 파일: `{work_dir}/gradnorm_log.csv` (header 포함)
- Append mode. Training 재시작 시 기존 파일이 있으면 `gradnorm_log_YYYYMMDD_HHMMSS.csv` 로 rename 후 새 파일 생성 (resume 시 역사 보존).
- Columns (순서 고정):
  ```
  step, epoch, iter_in_epoch, phase_id, lr_model,
  w_det, w_map, w_motion, w_plan, w_ego,
  grad_norm_det, grad_norm_map, grad_norm_motion, grad_norm_plan, grad_norm_ego,
  rt_det, rt_map, rt_motion, rt_plan, rt_ego,
  L_det, L_map, L_motion, L_plan, L_ego,
  L0_det, L0_map, L0_motion, L0_plan, L0_ego,
  weighted_L_det, weighted_L_map, weighted_L_motion, weighted_L_plan, weighted_L_ego,
  contribution_frac_det, ..., contribution_frac_ego,
  grad_norm_share_det, ..., grad_norm_share_ego,
  w_ratio_det, ..., w_ratio_ego,
  gn_loss, timestamp_iso
  ```
- 쓰기 빈도: **매 iter 기록** (wandb는 50 iter sampling, CSV는 full resolution).
  - 1 row ≈ 500 bytes · 1000 iter · 18 epoch ≈ 9 MB/epoch. 18 epoch 전체 ≈ 180 MB (허용 범위).
- `phase_id` 가 0 / 1 일 때는 grad_norm_*, rt_*, L0_* 컬럼을 `NaN`으로 기록.
- **계산 주체**: `SparseDetector._maybe_dump_gn_csv(gn_log)` 가 `GradNormLossWeighter` 의 원시 `gn_log`에 더해
  derived column (`contribution_frac_*`, `grad_norm_share_*`, `w_ratio_*`) 과 runner 메타데이터 (`epoch`, `iter_in_epoch`, `lr_model`, `timestamp_iso`) 를
  이 helper 내부에서 계산·조합한 뒤 rank 0 에서만 append. `init_w_sum` 은 `w_ratio` 계산에 쓰이므로
  `SparseDetector` 가 `self.gradnorm.w.detach()` 를 매 iter 참조.

### 6.4 Wandb 패널 권장 구성 (논문 figure 소스)
1. Total loss (`loss_gradnorm_total` + `loss_dense_depth`)
2. **Task weight dynamics** (`gn_w_*` 5-line plot) ← 논문의 핵심 figure
3. Per-task magnitude (`monitor_*` 별 prefix)
4. Gradient norm share & r̃ (`gn_grad_norm_share_*`, `gn_rt_*`)
5. Contribution fraction (`gn_contribution_frac_*` stacked area)
6. Phase marker (`gn_phase_id` step plot)

### 6.5 Validation metric (epoch-wise, 기존 eval hook 재사용)
기존 `evaluation.eval_mode` 가 모든 task에 대해 ON 이므로 epoch × `checkpoint_epoch_interval * 2` 마다 자동 기록. GradNorm 영향 분석용:
- det: mAP, NDS, per-class AP
- map: mAP
- motion: minADE, minFDE
- planning: L2 (1s/2s/3s), collision rate
- ego: L1 (config의 status)
추가 스크립트로 baseline E2(1/3 seed0) 의 동일 metric과 epoch-wise delta 계산 (post-hoc, 본 design scope 밖).

---

## 7. Risks, failure modes, rollback

### 7.1 Risk matrix

| # | 위험 | 실패 양상 | 완화책 |
|---|---|---|---|
| R1 | DDP rank 간 `w` drift | 느리게 분산 성능 차이 누적 | Section 3.3의 `all_reduce(mean)` 매 update 직후 |
| R2 | fp16 gradient underflow → per-task norm 0 | r̃ 분모 0, `w` 발산 | Safety 1 (fp32 승격) + `clamp_min(1e-8)` 여러 곳 |
| R3 | pivot `L0` 오염 (너무 이른 시점) | 학습 내내 r̃ 편향 | Safety 2 (warmup+50 iter 평균) |
| R4 | `w_i` 음수/0 | 부호 반전, 학습 붕괴 | Safety 3 (clamp + sum-to-init-sum) |
| R5 | `update_every=1` 추가 backward로 iter time 과증가 | 학습 시간 1.3× 초과 | 관측 후 `update_every=5`로 완화 |
| R6 | shared W 선택이 부적절하여 signal 거의 0 | r̃ 의미 상실 | 6 layer concat smoothing; `gn_grad_norm_*` 모니터 |
| R7 | `AsymmetricFFN` 구조 가정 불일치 | 초기화 시 Assert | `collect_ffn_last_fc_params` 의 assert |
| R8 | `_parse_losses`가 기대한 dict 구조 변경 | unexpected loss 합산 | Rename rule로 `'loss'` 문자 제거 |
| R9 | stage1 ckpt load 시 `gradnorm.*` key 없음 | load warning | mmcv `strict=False` (기본) → 신규 state 초기화 |
| R10 | GradNorm 학습 발산 | `w` 특정 값으로 폭증/0 수렴 | clamp+renorm; 발산 시 `lr_w=5e-3`으로 완화 |

### 7.2 Rollback (3-tier)
- **Tier A (즉시 off)**: config의 `model.gradnorm = None` → E2 baseline과 bit-identical.
- **Tier B (구조 유지, GN 동결)**: `update_after_step = 10**9` → warmup gate 영구 유지, `w = init_weights` 고정.
- **Tier C (전면 rollback)**: E9 config, `core/gradnorm/`, SparseDetector/SparseOneDecoder의 신규 diff를 git revert.

### 7.3 Go / no-go 검증 (학습 시작 후 첫 1k iter)
1. `iter < 500`: `gn_phase_id=0`, `loss_gradnorm_total` ≈ baseline 5 task sum.
2. `500 ≤ iter < 550`: `phase_id` 1→2 전환, `gn_L0_*` 모두 유한·양수.
3. `iter ≥ 550`: `gn_w_*` 초기값 근처에서 smooth drift (단일 iter 변화 < 5%), `gn_loss` 폭발 없음.
4. 수치 안정성: `loss_gradnorm_total`, `weighted_loss` 에 NaN/Inf 없음 (`grad_clip=25` 로 1차 방어).

모두 pass → full 18 epoch 진행. 실패 → Tier B 로 일시 동결 후 진단.

---

## Appendix A — 파일/심볼 cross-reference

| Target | 파일 | 섹션 |
|---|---|---|
| `SparseDetector.__init__` / `forward_train` | `projects/mmdet3d_plugin/models/sparse_detector.py` | 4.3 |
| `_aggregate_task_losses`, `_rename_as_monitor`, `_maybe_dump_gn_csv` | 동 파일 | 4.3, 6.3 |
| `SparseOneDecoder.collect_ffn_last_fc_params` | `projects/mmdet3d_plugin/models/sparse_onedecoder.py` | 4.4 |
| `GradNormLossWeighter` | `projects/mmdet3d_plugin/core/gradnorm/weighter.py` | 3 |
| E9 config | `projects/configs/experiments/E9_E2_E1_stage2_18ep_GN.py` | 5 |
| `AsymmetricFFN.layers` | `projects/mmdet3d_plugin/models/blocks.py:329` | 4.4 |
| mmcv `_parse_losses` | `mmdet/models/detectors/base.py:176` | 4.2 |
| 기존 E2 loss collection | `projects/mmdet3d_plugin/models/sparse_onedecoder.py:1123` (`loss`) | 4.1 |

## Appendix B — 참고 구현

- Chen et al. 2018, ICML (원 논문): `target = gw_avg * r̃^α`, `w_i` Parameter + 별도 Adam, `T = num_tasks`로 sum renormalize.
- `LucasBoTang/GradNorm` (`rideflux_etc/GradNorm/gradnorm.py`): 논문과 부호 일치, iter 0 pivot, `T = num_tasks` sum-to-T.
- `lucidrains/gradnorm-pytorch` (`rideflux_etc/gradnorm-pytorch/gradnorm_pytorch/gradnorm_pytorch.py`): `register_buffer` 기반, `update_after_step`/`update_every`/`initial_losses_decay`, L1 normalize × init_sum.

본 설계는 **LucasBoTang 뼈대 + lucidrains의 `update_after_step`/`update_every`/sum-to-init-sum** 를 조합했다.
