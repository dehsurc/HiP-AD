# HiP-AD Gradient Conflict 분석 결과

---

## 1. 모델 아키텍처와 분석 영역

### 전체 구조

```
6x Camera Images
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  BACKBONE (ResNet50 + FPN)                                    │
│  ─ 이미지에서 multi-scale feature 추출                        │
│  ─ 모든 task의 gradient가 여기를 통과                         │
│  ─ 파라미터 수: 매우 많음 → gradient가 dilute됨               │
│                                                               │
│  📊 분석 그룹: "backbone"                                     │
└──────────────────────┬───────────────────────────────────────┘
                       │ image features
                       ▼
┌══════════════════════════════════════════════════════════════─┐
│  ONEDECODER (6 decoder layers 반복)                           │
│                                                               │
│  4개의 query가 동시에 처리됨:                                 │
│  [det: 900개] [map: 100개] [plan: 6개] [ego: 1개]            │
│                                                               │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│  ┃  SHARED 파라미터 (gradient conflict 분석 대상)          ┃  │
│  ┃                                                         ┃  │
│  ┃  ┌───────────────────────────────────────────────────┐  ┃  │
│  ┃  │ temp_gnn (Temporal Attention)                     │  ┃  │
│  ┃  │  attn[0]: det ← det(prev)     det만 통과         │  ┃  │
│  ┃  │  attn[1]: map ← map(prev)     map만 통과         │  ┃  │
│  ┃  │  attn[2]: [plan,ego] ← [det,map]  plan+ego 통과  │  ┃  │
│  ┃  │                                                   │  ┃  │
│  ┃  │  📊 분석 그룹: "temp_gnn"                         │  ┃  │
│  ┃  └───────────────────────────────────────────────────┘  ┃  │
│  ┃                       │                                 ┃  │
│  ┃                       ▼                                 ┃  │
│  ┃  ┌───────────────────────────────────────────────────┐  ┃  │
│  ┃  │ gnn (Self-Attention)                              │  ┃  │
│  ┃  │  attn[0]: det ← det           det만 통과         │  ┃  │
│  ┃  │  attn[1]: map ← map           map만 통과         │  ┃  │
│  ┃  │  (plan, ego는 안 거침)                            │  ┃  │
│  ┃  │                                                   │  ┃  │
│  ┃  │  📊 분석 그룹: "gnn"                              │  ┃  │
│  ┃  └───────────────────────────────────────────────────┘  ┃  │
│  ┃                       │                                 ┃  │
│  ┃                       ▼                                 ┃  │
│  ┃  ┌───────────────────────────────────────────────────┐  ┃  │
│  ┃  │ inter_gnn (Cross-Attention)                       │  ┃  │
│  ┃  │  [plan,ego] Q ← [det,map] K                      │  ┃  │
│  ┃  │  plan/ego가 det/map 정보를 cross-attend           │  ┃  │
│  ┃  │                                                   │  ┃  │
│  ┃  │  📊 분석 그룹: "inter_gnn"                        │  ┃  │
│  ┃  └───────────────────────────────────────────────────┘  ┃  │
│  ┃                       │                                 ┃  │
│  ┃                       ▼                                 ┃  │
│  ┃  ┌───────────────────────────────────────────────────┐  ┃  │
│  ┃  │ norm (LayerNorm)  ★ 모든 task query 공유          │  ┃  │
│  ┃  │  파라미터 수 적음, task query가 직접 통과          │  ┃  │
│  ┃  │                                                   │  ┃  │
│  ┃  │  📊 분석 그룹: "decoder_norm"                     │  ┃  │
│  ┃  └───────────────────────────────────────────────────┘  ┃  │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
│                          │                                    │
│  ┌───────────────────────▼───────────────────────────────┐   │
│  │ deformable (Image Feature Aggregation)  TASK-SPECIFIC  │   │
│  │  det_deformable  → det만                               │   │
│  │  map_deformable  → map만                               │   │
│  │  plan_deformable → plan만                              │   │
│  │  ego_deformable  → ego만                               │   │
│  │                                                        │   │
│  │  ⚠ 각 task 전용 → conflict 분석 대상 아님             │   │
│  └────────────────────────┬──────────────────────────────┘   │
│                           │                                   │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━▼━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│  ┃  FFN + norm  ★ 모든 task query 공유                    ┃  │
│  ┃  파라미터 수 적음, task query가 직접 통과               ┃  │
│  ┃                                                         ┃  │
│  ┃  📊 분석 그룹: "decoder_ffn", "decoder_norm"           ┃  │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
│                           │                                   │
│  ┌────────────────────────▼──────────────────────────────┐   │
│  │ refine (Prediction Heads)               TASK-SPECIFIC  │   │
│  │  det_refine    → bbox cls/reg                          │   │
│  │  map_refine    → line cls/reg                          │   │
│  │  plan_refine   → plan cls/reg                          │   │
│  │  ego_refine    → ego status                            │   │
│  │  motion_refine → traj cls/reg (det feature 입력)       │   │
│  │                                                        │   │
│  │  ⚠ 각 task 전용 → conflict 분석 대상 아님             │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  (위 과정을 6번 반복)                                         │
└══════════════════════════════════════════════════════════════─┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  LOSS 계산                                                    │
│  det_loss = Σ(det_loss_cls + det_loss_box + ...)              │
│  map_loss = Σ(map_loss_cls + map_loss_line + ...)             │
│  plan_loss = Σ(plan_loss_temp_cls + plan_loss_temp_reg + ...) │
│  ego_loss = Σ(ego_loss_status)                                │
│  motion_loss = Σ(motion_loss_cls + motion_loss_reg)           │
└──────────────────────────────────────────────────────────────┘
```

### 분석 그룹 요약

| 그룹 | 위치 | 어떤 task가 통과 | 왜 분석하는가 |
|------|------|-----------------|--------------|
| **backbone** | 이미지 인코더 | 전부 | 모든 task가 공유하지만 파라미터가 너무 많아 conflict가 dilute됨 |
| **decoder_ffn** | decoder 내 FFN | det, map, plan, ego | 파라미터 적고 모든 query가 직접 통과 → conflict 집중 가능 |
| **decoder_norm** | decoder 내 LayerNorm | det, map, plan, ego | scale/bias 소수 파라미터, 모든 query 공유 → conflict 민감 |
| **gnn** | self-attention | det(attn0), map(attn1) | det/map 각각 분리된 attention |
| **inter_gnn** | cross-attention | plan+ego ← det+map | plan이 det/map 정보를 가져오는 핵심 접점 |
| **temp_gnn** | temporal attention | det, map, plan+ego | 시간축 정보 교환 |
| **decouple_fc** | attention 전후 FC | det (decouple 구조) | det의 512d↔256d 변환 |

### Gradient가 0인 영역 (분석 의미 없음)

| 파라미터 그룹 | gradient = 0인 task | 이유 |
|---|---|---|
| **전체** | ego | ego query 1개, fp16에서 gradient flush 추정 |
| inter_gnn | det↔map 쌍 | inter_gnn은 plan+ego가 attend하는 구조, det/map 자체는 안 거침 |
| temp_gnn | det↔map 쌍 | 위와 동일 |
| decouple_fc | map 관련 | decouple은 det 전용 |

---

## 2. 분석 방법

### 과정

```
각 체크포인트에서:
  1. forward pass → loss dict (det_loss_*, map_loss_*, plan_loss_*, ...)
  2. task별로 loss 합산
  3. 각 task loss를 개별 backward (retain_graph=True)
  4. shared 파라미터에 대한 gradient 수집
  5. 파라미터 그룹별로 gradient를 1D vector로 flatten
  6. task pair 간 cosine similarity 계산
  7. 12개 batch에 대해 평균
```

### Metric

| 값 | 의미 |
|---|---|
| **cosine > +0.05** | cooperate (같은 방향) |
| **cosine ≈ 0** | 직교 (무관) |
| **cosine < -0.05** | conflict (반대 방향) |
| **CR (conflict ratio)** | 12개 batch 중 cosine < 0인 비율 |

PCGrad도 동일한 방식(전체 파라미터 flatten → cosine)으로 conflict를 판별하므로, 이 분석 결과가 PCGrad 적용 여부의 직접적 근거가 된다.

---

## 3. 결과

### 주요 task pair 결과 (cosine / CR)

#### BACKBONE

| Task Pair | iter_7032 | iter_14064 | iter_21096 |
|-----------|-----------|------------|------------|
| det vs map | **+0.050 / 0%** | **+0.078 / 0%** | +0.041 / 8% |
| det vs motion | +0.030 / 50% | +0.006 / 58% | **+0.066 / 25%** |
| det vs plan | +0.015 / 33% | -0.005 / 58% | +0.009 / 33% |
| map vs plan | -0.007 / 42% | +0.005 / 42% | -0.009 / 58% |
| motion vs plan | +0.004 / 50% | -0.000 / 25% | -0.014 / **75%** |

#### DECODER_FFN

| Task Pair | iter_7032 | iter_14064 | iter_21096 |
|-----------|-----------|------------|------------|
| det vs map | +0.002 / 25% | -0.001 / 50% | +0.001 / 50% |
| det vs motion | -0.002 / 50% | **-0.043 / 67%** | +0.005 / 42% |
| det vs plan | +0.008 / 8% | -0.002 / 50% | -0.007 / **67%** |
| map vs plan | +0.004 / 33% | +0.007 / 33% | +0.011 / 33% |
| motion vs plan | +0.005 / 25% | -0.006 / **67%** | -0.001 / 42% |

#### DECODER_NORM

| Task Pair | iter_7032 | iter_14064 | iter_21096 |
|-----------|-----------|------------|------------|
| det vs map | -0.005 / **83%** | -0.012 / **75%** | -0.002 / 58% |
| det vs motion | +0.022 / 58% | -0.037 / **67%** | **+0.069 / 25%** |
| det vs plan | +0.035 / 25% | +0.001 / 50% | +0.002 / 67% |
| map vs plan | +0.016 / 42% | +0.018 / 25% | +0.033 / 33% |
| motion vs plan | -0.007 / 58% | -0.008 / 58% | +0.002 / 50% |

#### INTER_GNN

| Task Pair | iter_7032 | iter_14064 | iter_21096 |
|-----------|-----------|------------|------------|
| det vs plan | -0.001 / 58% | -0.000 / 33% | -0.002 / 58% |
| map vs plan | +0.001 / 50% | -0.003 / **67%** | -0.001 / 58% |
| motion vs plan | +0.002 / 33% | -0.004 / **67%** | -0.001 / 50% |

#### DECOUPLE_FC

| Task Pair | iter_7032 | iter_14064 | iter_21096 |
|-----------|-----------|------------|------------|
| det vs motion | +0.018 / 67% | +0.015 / 42% | +0.052 / 17% |
| det vs plan | +0.008 / 50% | +0.013 / 33% | +0.005 / 50% |
| motion vs plan | +0.025 / 25% | +0.005 / 42% | **-0.030 / 83%** |

---

## 4. 분석

### 전반적 결과

대부분의 cosine 값이 **-0.05 ~ +0.08 범위**이다. 심한 conflict (cos < -0.1)는 어떤 파라미터 그룹에서도 관찰되지 않았다.

### 주목할 패턴

**(a) det vs map: backbone에서 cooperate, decoder_norm에서 conflict**

같은 task pair인데 파라미터 위치에 따라 반대 결과:
- backbone: cos +0.04~0.08, CR 0~8% → cooperate
- decoder_norm: cos -0.005~-0.012, CR 58~83% → conflict

backbone은 두 task에 모두 유용한 feature를 뽑고 있지만, LayerNorm의 scale/bias 조정 방향은 서로 다름.

**(b) motion vs plan: 학습 후반에 conflict 악화**

- backbone: CR 50% → 75%
- decouple_fc: CR 25% → **83%**, cos +0.025 → **-0.030**

학습이 진행될수록 motion과 plan의 gradient 방향이 점점 달라짐. motion이 det feature 기반이라 간접적으로 plan과 경쟁.

**(c) det vs motion: 학습 중 alignment 개선**

- backbone: cos +0.03 → +0.07
- gnn: cos -0.004 → +0.057

motion이 det feature에서 파생되므로 학습이 진행되면서 자연스럽게 align.

---

## 5. 결론

1. **HiP-AD Stage2에서 심각한 gradient conflict는 관찰되지 않는다.** cosine이 대부분 -0.05~+0.08 범위로 직교에 가까움
2. **PCGrad 같은 gradient surgery를 적용할 강한 근거가 없다.** PCGrad는 cosine < 0일 때만 수정하는데, 현재 cosine ≈ 0이므로 projection할 것이 거의 없음
3. **직교 ≠ "문제 없음"** — task 간 gradient가 직교한다는 것은 multi-task synergy가 없다는 의미이기도 함. 이는 gradient conflict와는 다른 관점의 문제
4. 가장 눈에 띄는 패턴은 **motion vs plan의 학습 후반 conflict 악화** (decouple_fc에서 cos -0.030, CR 83%)이나, 절대적 크기는 여전히 작음
