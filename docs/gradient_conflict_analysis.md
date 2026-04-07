# HiP-AD Gradient Conflict 분석 설계

## 1. 핵심 아이디어

Multi-task 모델에서 각 task loss를 **개별적으로 backward**하면, 같은 파라미터에 대해 task마다 다른 gradient가 생긴다.
이 gradient들의 방향이 서로 반대면 → **gradient conflict** → 학습이 비효율적이거나 특정 task가 희생됨.

```
                    forward (공유)
  Input ──→ [Shared Parameters θ] ──→ det_loss
                                  ──→ map_loss
                                  ──→ plan_loss
                                  ──→ ego_loss
                                  ──→ motion_loss

  각각 따로 backward하면:
    ∂(det_loss)/∂θ    = g_det
    ∂(map_loss)/∂θ    = g_map
    ∂(plan_loss)/∂θ   = g_plan
    ∂(ego_loss)/∂θ    = g_ego
    ∂(motion_loss)/∂θ = g_motion

  cosine_similarity(g_plan, g_det) < 0  →  conflict!
```

**OneDecoder 구조이기 때문에 이게 가능한 것이다.** 파라미터가 완전히 분리되어 있었다면 비교할 shared θ 자체가 없다.

---

## 2. HiP-AD Stage2 모델 구조

### 2.1 전체 구조 (위에서 아래로)

```
┌─────────────────────────────────────────────────────────────┐
│                     6x Camera Images                         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ★ SHARED: img_backbone (ResNet50) + img_neck (FPN)         │
│            + depth_branch                                    │
│                                                              │
│  모든 task의 gradient가 여기를 통과함                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ multi-scale image features
                           ▼
┌─════════════════════════════════════════════════════════════─┐
│              OneDecoder × 6 layers                           │
│                                                              │
│  4개의 query stream:                                         │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐                   │
│  │det:900│ │map:100│ │plan: 6│ │ego:  1│                    │
│  └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘                   │
│      │         │         │         │                         │
│      ▼         ▼         ▼         ▼                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ temp_gnn (Temporal Attention) — temporal layer에서만  │   │
│  │                                                       │   │
│  │  attn[0]: det ← det(prev)           det gradient만   │   │
│  │  attn[1]: map ← map(prev)           map gradient만   │   │
│  │  attn[2]: [plan,ego] ← [det,map]    plan+ego grad    │   │
│  │           └→ det,map에도 key를 통해 plan grad 전파    │   │
│  └──────────────────────────────────────────────────────┘   │
│      │         │         │         │                         │
│      ▼         ▼         ▼         ▼                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ gnn (Self-Attention)                                  │   │
│  │                                                       │   │
│  │  attn[0]: det ← det                 det gradient만   │   │
│  │  attn[1]: map ← map                 map gradient만   │   │
│  │  (plan, ego는 여기 안 거침)                           │   │
│  └──────────────────────────────────────────────────────┘   │
│      │         │         │         │                         │
│      ▼         ▼         ▼         ▼                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ inter_gnn (Cross-Attention)                           │   │
│  │                                                       │   │
│  │  attn[0]: [plan,ego] Q ← [det,map] K                 │   │
│  │           plan/ego가 det/map 정보를 가져옴            │   │
│  │           plan+ego gradient만 (attn 파라미터)         │   │
│  │           det/map에도 key grad 전파 (representation)  │   │
│  └──────────────────────────────────────────────────────┘   │
│      │         │         │         │                         │
│      ▼         ▼         ▼         ▼                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ★ SHARED: norm (LayerNorm 256d)                       │   │
│  │   모든 query에 동일 파라미터 적용                     │   │
│  └──────────────────────────────────────────────────────┘   │
│      │         │         │         │                         │
│      ▼         ▼         ▼         ▼                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ deformable (Image Feature Aggregation) — task별 분리  │   │
│  │                                                       │   │
│  │  det_deformable   → det만    (task-specific 파라미터) │   │
│  │  map_deformable   → map만    (task-specific 파라미터) │   │
│  │  ego_deformable   → ego만    (task-specific 파라미터) │   │
│  │  plan_deformable  → plan만   (task-specific 파라미터) │   │
│  │                                                       │   │
│  │  ※ 파라미터는 각자이지만, image feature는 공유       │   │
│  │    → backbone으로의 gradient는 모든 task에서 흐름     │   │
│  └──────────────────────────────────────────────────────┘   │
│      │         │         │         │                         │
│      ▼         ▼         ▼         ▼                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ★ SHARED: FFN (256d→1024d→256d) + norm               │   │
│  │   concat된 모든 query에 동일 파라미터 적용            │   │
│  └──────────────────────────────────────────────────────┘   │
│      │         │         │         │                         │
│      ▼         ▼         ▼         ▼                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ refine (Prediction Heads) — task별 분리               │   │
│  │                                                       │   │
│  │  det_refine    → bbox cls/reg     (task-specific)     │   │
│  │  map_refine    → line cls/reg     (task-specific)     │   │
│  │  ego_refine    → ego status       (task-specific)     │   │
│  │  plan_refine   → plan cls/reg     (task-specific)     │   │
│  │  motion_refine → traj cls/reg     (task-specific)     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  (위 과정을 layer 1~6까지 반복, 매 layer마다 loss 합산)     │
└─════════════════════════════════════════════════════════════─┘
```

### 2.2 Motion의 특수한 위치

Motion은 `query_select`에 포함되지 않아 자체 query stream이 없다.
대신 **det의 output(instance_feature + anchor_embed)을 입력**으로 받아 motion_refine을 통해 예측한다.

```
det_instance_feature ─┐
                      ├─→ motion_query ─→ motion_refine ─→ motion_loss
det_anchor_embed ─────┘
                      ↑
              motion_anchor_encoder (static anchor의 position embedding)
```

**따라서 motion_loss의 gradient는 det feature를 거쳐 backbone까지 흐른다.**

---

## 3. Gradient Conflict 분석이 가능한 이유

### 3.1 "각 task loss를 따로 backward하면 안 되나?"에 대한 답: **된다.**

PyTorch에서 `loss.backward(retain_graph=True)`를 사용하면 각 task loss에 대해 독립적으로 gradient를 계산할 수 있다.

```python
# 1) Forward pass (1번만)
losses = model.forward_train(img, **data)

# 2) task별 loss 분리
det_loss = sum(v for k, v in losses.items() if k.startswith('det_'))
map_loss = sum(v for k, v in losses.items() if k.startswith('map_'))
plan_loss = sum(v for k, v in losses.items() if k.startswith('plan_'))
ego_loss = sum(v for k, v in losses.items() if k.startswith('ego_'))
motion_loss = sum(v for k, v in losses.items() if k.startswith('motion_'))

# 3) 각각 backward해서 gradient 수집
model.zero_grad()
det_loss.backward(retain_graph=True)
g_det = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

model.zero_grad()
map_loss.backward(retain_graph=True)
g_map = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

# ... plan, ego, motion도 동일
```

### 3.2 OneDecoder 구조가 분석을 막지 않는다

OneDecoder는 여러 task의 query를 **하나의 decoder에서 같이 처리**하는 구조다.
이것은 gradient conflict 분석을 **막는 게 아니라, 오히려 conflict가 발생하는 이유**다.

- 만약 task마다 완전히 별도의 decoder를 썼다면, shared parameter가 backbone뿐이므로 conflict 포인트가 적다.
- OneDecoder에서는 FFN, norm 등이 모든 task query에 대해 공유되므로, conflict가 발생할 수 있는 지점이 많다.

---

## 4. Shared 파라미터 영역별 Gradient 흐름

| 파라미터 영역 | 어떤 task의 gradient가 흐르는가 | Conflict 가능성 |
|---|---|---|
| **backbone + FPN** | det, map, plan, ego, motion (ALL) | ★★★ 최대 |
| **FFN** (layer당 1개) | det, map, plan, ego (query가 있는 4개) | ★★★ |
| **norm** (layer당 2개) | det, map, plan, ego | ★★ |
| **temp_gnn.attn[2]** | plan, ego (+ det/map via key grad) | ★★ |
| **inter_gnn.attn[0]** | plan, ego (파라미터), det/map (key grad) | ★★ |
| temp_gnn.attn[0] | det only | 없음 |
| temp_gnn.attn[1] | map only | 없음 |
| gnn.attn[0] | det only | 없음 |
| gnn.attn[1] | map only | 없음 |
| *_deformable | 각자 task only | 없음 |
| *_refine, *_decoder | 각자 task only | 없음 |

---

## 5. 분석 계획

### 5.1 분석 대상

- **체크포인트**: iter_7032, iter_14064, iter_21096 (3개)
- **데이터**: training set에서 batch_size=6으로 N개 batch 샘플링
- **Task 그룹**: det, map, plan, ego, motion (5개)

### 5.2 분석할 Metric

각 task pair (i, j)에 대해 shared 파라미터 θ에서:

1. **Cosine Similarity**: `cos(g_i, g_j)` — 방향 일치도
   - \> 0: 같은 방향 (cooperate)
   - < 0: 반대 방향 (conflict)

2. **Gradient Magnitude Ratio**: `||g_i|| / ||g_j||` — 크기 불균형
   - 한 task의 gradient가 지배적이면 다른 task가 무시됨

3. **Conflict Ratio**: batch 중 conflict가 발생하는 비율

### 5.3 파라미터 영역별 분리 분석

전체 shared를 하나로 flatten하는 것 외에, 영역별로 나눠서 분석:
- **backbone**: img_backbone.* + img_neck.*
- **decoder_ffn**: head.*.ffn.*
- **decoder_norm**: head.*.norm.*
- **decoder_attn**: temp_gnn.attn[2] + inter_gnn.attn[0]

### 5.4 체크포인트 간 비교

- 학습 초기(7032) → 중기(14064) → 후기(21096)로 갈수록 conflict가 줄어드는지/커지는지
- 특히 plan vs det, plan vs map conflict가 학습에 따라 어떻게 변화하는지

---

## 6. Loss 키 참조 (Stage2 config 기준)

```
det_loss_*     : det_loss_cls, det_loss_box, det_loss_cns, det_loss_yns (+ dn variants)
map_loss_*     : map_loss_cls, map_loss_line
ego_loss_*     : ego_loss_status
plan_loss_*    : plan_loss_temp_cls, plan_loss_temp_reg
                 (optional: plan_loss_col, plan_loss_dir, plan_loss_bound, plan_loss_status)
motion_loss_*  : motion_loss_cls, motion_loss_reg
depth_loss     : depth_loss (from depth_branch, not a "task" per se)
scenes_loss_*  : scenes_loss_reg (future frame prediction)
```

### Loss Weight 참조

| Loss | Weight |
|------|--------|
| det_cls | 2.0 |
| det_box | 0.25 |
| map_cls | 1.0 |
| map_line | 10.0 |
| ego_status | 1.0 |
| plan_cls | 0.5 |
| plan_reg | 1.0 |
| motion_cls | 0.2 |
| motion_reg | 0.2 |
