# release → revision: HiP-AD §3.2 Collaborative Interaction 정리

출발점: `main` (= release, paper 공개 코드)
도착점: `revision` branch
중간 단계 (v2/v3) 는 생략. 최종 누적 변경만 정리.

---

## 1. Paper §3.2 sub-claim 매핑 (release vs revision)

| paper §3.2 sub-claim | release (main) | **revision** |
|---|---|---|
| ① per-task self-attn (perception) | `[det], [map]` — plan/ego 박스 없음 | `[det], [map], [plan, ego]` — figure에 그려진 3 박스 매핑 (plan/ego는 release의 concat 그룹 convention 그대로) |
| ② unified self-attn (cross-task) | asymmetric `Q=plan/ego, K=det/map` (`InteractiveAttention`) — perception 끼리 cross 자체가 없음 | symmetric `[det, map, plan, ego]` self-attn (`SeparateAttention`) **+ structured mask** (perception 행 × plan/ego 열 = −∞) — paper text "plan accesses all tasks" 단방향 + figure 회색 cell 둘 다 매핑 |
| ③ Geometric Attention `Softmax(QK^T/√C − τD)V` | **dead code**: `MultiheadFlashAttention.forward`의 `assert attn_mask is None` 으로 mask 차단 + 모든 config에서 `with_distance_attn_mask=False` default | **active**: SDPA 수동 구현 path + config flag on. unified self-attn 안에서만 적용 |
| ④ τ MLP from Q | `distance_tau = nn.Linear(embed_dims, 8)` 정의만 있고 forward 호출 안 됨 | forward에서 `tau = distance_tau(sep_query)` 호출됨 |
| ⑤ D = pairwise distance matrix (paper: map-agent, map-map, agent-map) | `get_distance_attn_mask` 함수만 있고 호출 안 됨 | 호출됨. unified self-attn 안의 perception×perception 페어 (det-det, det-map, map-det, map-map) 에 적용. plan/ego 행은 τ=0 면제 |
| ⑥ plan exempt from τ·D | moot (τ·D 자체가 dead) | **plan AND ego** 둘 다 면제 (release plan/ego concat convention 일치) |
| Temporal Interaction (paper figure: 4 blocks) | 3 pairs (plan↔plan_hist 누락) | 4 pairs — plan↔plan_hist 추가 |

---

## 2. 변경 파일 (코드 + config)

### A. [`projects/mmdet3d_plugin/models/attention.py`](../projects/mmdet3d_plugin/models/attention.py)

- `MultiheadFlashAttention.forward`의 `assert attn_mask is None` 제거
- `FlashMHA` inner attention 호출을 **수동 scaled dot-product attention** 으로 교체
  ```python
  scale = 1.0 / math.sqrt(q.shape[-1])
  attn = torch.matmul(q, k.transpose(-2, -1)) * scale
  if final_mask is not None:
      attn = attn + final_mask
  attn = attn.softmax(dim=-1)
  ...
  context = torch.matmul(attn, v)
  ```
- 이유: PyTorch 1.13 환경. `F.scaled_dot_product_attention` 없음 + flash-attn은 임의 additive bias 못 받음. mask 받으려면 수동 SDPA가 유일한 단일-path 선택지 (no silent fallback)

### B. [`projects/mmdet3d_plugin/models/separate_attn.py`](../projects/mmdet3d_plugin/models/separate_attn.py)

#### `SeparateAttention` — 3가지 신규 기능 추가

1. **`with_distance_attn_mask` / `with_velocity_attn_mask` 플래그 추가** — `__init__` 에 받아서 forward에서 `InteractiveAttention.get_distance_attn_mask` 호출. release에서는 InteractiveAttention 클래스에만 있던 거리 attention 코드가 SeparateAttention에서도 호출됨 (graph_model이 SeparateAttention이라 per-task self-attn에도 같은 코드 재사용 가능). 단 revision config 에서는 per-task self-attn에는 적용 안 함 (paper §3.2가 unified self-attn 안의 mechanism으로만 명시)

2. **`with_structured_mask` 플래그 + `_make_structured_mask` 메서드 (revision의 핵심)** — perception_row × planning_column = −∞ 의 additive mask 생성. shape `[Nq, Nq]`, batch/head broadcast. config로 perception/planning modality 분류 override 가능
   ```python
   def _make_structured_mask(self, separate, num_query_list, dtype, device):
       # build [Nq, Nq] mask: -inf at perception_row × planning_column, 0 elsewhere
       ...
   ```

3. **forward의 self-attn 경로 (`key is None` branch)** — distance mask + velocity mask + structured mask 차례로 sep_attn_mask에 누적 더해서 attention block에 전달

#### `InteractiveAttention.get_distance_attn_mask` — plan/ego 둘 다 면제

```python
# Paper §3.2 + release convention (plan/ego concat 그룹)
for q_type, q_len in zip(sep_query_list, q_lens):
    if q_type in ("plan", "ego"):     # release는 호출 안 했고, 호출되더라도 면제 로직 자체 없었음
        tau = tau.clone()
        tau[:, q_offset:q_offset + q_len, :] = 0.0
    q_offset += q_len
```

또 return shape `[B*H, Nq, Nk]` → `[B, H, Nq, Nk]` 로 변경 (SDPA path 호환).

### C. [`projects/mmdet3d_plugin/models/sparse_onedecoder.py`](../projects/mmdet3d_plugin/models/sparse_onedecoder.py)

nuScenes 학습 지원 + multi-granularity 인프라 + distill hook 등 대규모 추가 (release 대비 +724 라인). §3.2 관련 핵심 변경은:

- decoder 레벨 `with_distance_attn_mask` flag 처리. `True`면 `self.distance_tau = nn.Linear(embed_dims, 8)` instantiate
- `inter_gnn` op 호출부에 `det_anchor`, `map_anchor`, `plan_anchor`, `distance_tau` 인자 전달 (unified self-attn에서 거리 gating 입력으로 사용)
- `gnn` op 호출부에는 위 인자 전달 안 함 (per-task self-attn에 τ·D 안 씀)

### D. [`projects/configs/experiments/E1_stage1_12ep_new.py`](../projects/configs/experiments/E1_stage1_12ep_new.py) + [`E2_E1_stage2_18ep_new.py`](../projects/configs/experiments/E2_E1_stage2_18ep_new.py) (신규 config, release에 없던 파일)

#### 학습 setup
| | release b2d | **revision _new (nuScenes)** |
|---|---|---|
| `num_gpus` | 8 | **4** |
| `batch_size` (per-GPU) | 4 | 4 (paper §4.2와 동일) |
| total batch | 32 | 16 (paper의 1/2) |
| `lr` | 2e-4 | **1e-4** (total batch 1/2 → linear scaling) |
| `weight_decay` | 0.001 | 0.001 |
| stage1 epochs | 12 | 12 |
| stage2 epochs | 6 | **18** (nuScenes baseline 그대로 — 데이터 분포 차이로 길게 학습) |
| iters/epoch (stage1) | 879 | 1758 |
| 학습량 (= epochs × dataset) | paper b2d 기준 | nuScenes 기준 |

#### Decoder onedecoder_head
- `with_distance_attn_mask = True` (decoder 레벨, distance_tau Linear instantiate)

#### graph_model (per-task self-attn)
- `separate_list = [["det"], ["map"], ["plan", "ego"]]` — release `[["det"],["map"]]` 에 figure에 그려진 3번째 박스 (plan/ego concat) 추가
- `decouple_list = [True, False, False]`
- 3개 MultiheadFlashAttention 인스턴스 (det는 decouple_attn=True로 embed*2, map/plan-ego는 embed_dims)
- **`with_distance_attn_mask` 미설정 (= False)** — per-task self-attn에 τ·D 적용 안 함

#### inter_graph_model (unified self-attn = collaborative attention)
- `type = "SeparateAttention"` — release의 `InteractiveAttention` 대체
- `separate_list = [["det", "map", "plan", "ego"]]` — 4-task unified self-attn
- `decouple_list = [False]`
- `with_distance_attn_mask = True` — τ·D 활성
- **`with_structured_mask = True`** — perception → plan/ego 차단 mask 활성 (revision의 핵심)

#### temp_graph_model (temporal interaction)
- `query_list = [["det"], ["map"], ["plan", "ego"], ["plan", "ego"]]`
- `key_list = [["det"], ["map"], ["plan", "ego"], ["det", "map"]]`
- 4번째 pair (plan↔plan_hist) 추가 — release 3 pairs에 paper figure의 plan-plan temporal cross-attn 추가
- 4번째 attention block (`MultiheadFlashAttention`) 추가

---

## 3. revision의 unified self-attn 실제 동작 (structured mask 그림)

`separate_list=[["det","map","plan","ego"]]` 한 그룹에서 SeparateAttention.forward가 만드는 attention matrix (예시: det=3, map=2, plan=4, ego=1 query):

```
      d d d  m m  p p p p  e        ← key 측
  d:  0 0 0  0 0  X X X X  X        ← det 행, plan/ego 열 차단
  d:  0 0 0  0 0  X X X X  X
  d:  0 0 0  0 0  X X X X  X
  m:  0 0 0  0 0  X X X X  X        ← map 행, plan/ego 열 차단
  m:  0 0 0  0 0  X X X X  X
  p:  0 0 0  0 0  0 0 0 0  0        ← plan 행, 모든 task 다 attend (paper "plan accesses all")
  p:  0 0 0  0 0  0 0 0 0  0
  p:  0 0 0  0 0  0 0 0 0  0
  p:  0 0 0  0 0  0 0 0 0  0
  e:  0 0 0  0 0  0 0 0 0  0        ← ego 행, plan과 동일하게 자유 (release convention)
```
- `X` = −∞ (softmax 후 attention weight 0)
- `0` = 자유 attend
- 추가로 perception×perception 페어 (det-det/det-map/map-det/map-map) 에는 τ·D 거리 gating도 곱해짐
- plan/ego 행은 τ·D 면제 (paper 명시)

---

## 4. revision branch 학습 명령

```bash
conda activate hipad && \
cd /home/kyungmin/min_ws/rideflux/HiP-AD && \
git checkout revision && \
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    bash tools/dist_train.sh projects/configs/experiments/E1_stage1_12ep_new.py 4 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    bash tools/dist_train.sh projects/configs/experiments/E2_E1_stage2_18ep_new.py 4
```

- work_dir: `work_dirs/exp/E1_stage1_12ep_new/`, `work_dirs/exp/E2_E1_stage2_18ep_new/`
- stage2 `load_from`: `./work_dirs/exp/E1_stage1_12ep_new/latest.pth`
- wandb run: `E1_stage1_12ep_new`, `E2_E1_stage2_18ep_new`

---

## 5. revision이 release 대비 paper와 가까워진 부분

1. **§3.2 Eq. 1 τ·D 거리 gating**: release에선 dead → revision에서 unified self-attn 안에 active
2. **§3.2 Collaborative Attention Map (figure 회색 cell)**: release에선 cell 자체가 없음 (asymmetric 직사각 matrix) → revision에서 정사각 matrix + structured mask로 회색 cell 표현
3. **§3.2 "map-agent, map-map, agent-map" cross-task 거리 gating**: release에선 적용처 없음 (perception 끼리 cross 자체가 없으니) → revision에서 적용
4. **§3.2 "plan accesses information from all tasks"**: release는 plan/ego → det/map 단방향이라 충족하지만 perception 끼리 차단 → revision은 plan/ego 자유 + perception 끼리 자유 (mask는 perception→plan/ego만 차단)
5. **§3.2 plan exempt from τ·D**: release moot → revision active (plan + ego 모두 면제)
6. **paper figure Temporal Interaction 4 blocks**: release 3 → revision 4 (plan↔plan_hist 추가)
7. **paper figure per-task self-attn 3 boxes**: release 2 → revision 3 (plan/ego 박스 추가)

---

## 6. 의도된 deviation (paper와 다르지만 nuScenes 환경/release 호환 위해 유지)

- **nuScenes-specific**: `ego_fut_mode=6`, `plan_anchor_types=[("temp","2hz")]` — paper App B "disable driving-style for nuScenes" 따름. §3.3 Multi-granularity Planning 자체가 nuScenes 학습에서는 비활성 (b2d에서만 활성)
- **lr, batch, schedule**: 4 GPU 환경 제약. paper의 8 GPU × 4 = 32 total → 4 × 4 = 16 (linear scaling), lr 1e-4
- **stage2 epochs = 18**: paper b2d=6. nuScenes baseline 그대로 이어받음 (데이터 분포 차이로 더 길게 학습)
- **`adapt_status=False`**: closed-loop fallback 로직 비활성. nuScenes는 open-loop라 무관
- **ego query 자체가 attention 참여**: paper figure에는 ego 없음. release가 1-query 모듈로 추가. revision은 release convention 그대로 유지

---

## 7. paper 명시 없는 곳에서 한 결정 (필요 시 ablation 가능)

| 결정 | 대안 | 영향 |
|---|---|---|
| `SeparateAttention.forward`의 `else` branch (cross-attn 경로)에는 distance/structured mask 코드 미적용 | 동일 logic 복제 | 현재 use case에서 호출 안 됨. 향후 사용 시 silent disable 위험 |
| `temp_gnn` 호출부에 τ·D 미전달 | TemporalSeparateAttention에도 distance code 추가 | paper §3.2가 temporal에 Eq.1 적용 명시 안 함. 보수적 결정 |
| ego도 plan과 함께 τ·D 면제 | plan만 면제 | release convention과 일치 (plan/ego가 같은 attn 블록 concat 그룹) |
| structured mask 기본 분류: `perception=("det","map"), planning=("plan","ego")` | motion 추가 또는 다른 분류 | nuScenes (motion 없음) stage1 + b2d (motion 있음) stage2 모두 호환. motion은 paper §3.2 cross-task 명시 안 됨 |
| `distance_tau.bias = uniform(0, 2.0)` | bias = 0 또는 다른 range | release 저자가 코드에 박아둔 값 그대로. paper 명시 없음. 학습 첫 epoch 모니터 필수 |
| flash-attn → 수동 SDPA path | xformers `memory_efficient_attention` | PyTorch 1.13 호환. 학습 속도 1.5~2x 감소 trade-off |

---

## 8. 모니터링 포인트

| 시그널 | 정상 | 경고 |
|---|---|---|
| `loss_det_cls` 첫 epoch 곡선 | 매끈하게 하강 | 평탄 또는 NaN → `distance_tau.bias` 너무 강함 |
| `loss_map_*` 곡선 | release 비슷한 추세 | 한참 안 떨어짐 → structured mask 강도 검토 |
| Attention entropy | 분포 적당 | 0 가까이 → τ·D + structured mask 결합이 너무 좁음 |
| GPU memory | < 22 GiB/GPU | OOM → batch 축소 또는 gradient checkpointing |
| Wall-clock per iter | release 1.5~2x | 3x 이상 → SDPA + mask overhead profile |
