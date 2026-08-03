# ATTITTUD 실험 종합 비교분석 (E10–E13)

**작성일:** 2026-07-14
**분석 노트북:** [`analysis/comprehensive_analysis.ipynb`](../analysis/comprehensive_analysis.ipynb) · [`analysis/attittud_surgery_analysis.ipynb`](../analysis/attittud_surgery_analysis.ipynb)
**그림:** [`analysis/figs2/`](../analysis/figs2/) · **surgery 그림:** [`analysis/figs/`](../analysis/figs/)

---

## 0. TL;DR

> **gradient surgery(ATTITTUD)는 이 세팅에서 planning에 도움이 되지 않으며, 오히려 미세하게 해롭다.** surgery를 완전히 제거한 순수 batch8 finetune(E13)이 planning L2·map mAP·NDS·안정성에서 모두 1위다. surgery의 개입은 aux⊥plan 구조상 원천적으로 미미(neutral 99.9%)했고, batch-1을 요구하는 대가로 심각한 ego 발산(NaN skip 수천 회)을 유발했다.

핵심 순위 (planning L2, 낮을수록 좋음):

| 순위 | 실험 | plan L2 | 비고 |
|---|---|---|---|
| 🥇 | **E13 batch8 (surgery 없음)** | **0.6224** | planning·map·NDS·안정성 최고 |
| 2 | E12 ATTITTUD det-primary | 0.6594 | |
| 3 | E10 ATTITTUD plan-primary | 0.6632 | |
| 4 | baseline 6ep (batch16 straight) | 0.7473 | |
| 5 | E11 aligned-aux (plan freeze) | 0.9630 | 별개 축(기생 planning) |
| — | floor (base 3ep, 시작점) | 1.0337 | |

- **surgery 제거 효과:** E10 0.6632 → E13 0.6224 = **planning −6.1% (surgery를 뺐더니 좋아짐)**
- **순수 finetune 효과:** floor 1.0337 → E13 0.6224 = **−39.8%** (surgery 없이 finetune만으로 40% 개선)

---

## 1. 실험 셋업

모든 finetune 실험은 **동일한 3ep 체크포인트**(`70+stage2_3ep.pth`)에서 **+3ep, 유효배치 8, lr 7e-5, fp32, BN frozen**으로 학습.

| id | 실험 | surgery | forward 방식 | 총 iter |
|---|---|---|---|---|
| base_3ep | floor (시작점) | — | — | — |
| base_6ep | straight 6ep | ❌ | batch16 | (별도) |
| **E10** | ATTITTUD plan-primary | ✅ SVD(k≈6) | batch1 × accum8 | 84,390 |
| **E11** | aligned-aux (plan head 동결) | 정렬 성분만 | batch1 × accum8 | 84,390 |
| **E12** | ATTITTUD det-primary | ✅ SVD(k≈6) | batch1 × accum8 | 84,390 |
| **E13** | 순수 finetune | ❌ 없음 | batch8 | 10,548 |

> **공정성 한계:** E13(batch8 forward)와 ATTITTUD(batch1 forward + surgery)는 유효배치는 같지만 **(a) forward 방식**과 **(b) surgery 유무**가 함께 다르다. 둘을 완전 격리하지는 못한다. 단, 아래 결과는 **두 요인 모두 ATTITTUD에 불리**함을 보여준다(surgery는 개입 미미, batch-1은 발산 유발).

---

## 2. Eval 완전 비교

### 2.1 헤드라인 (그림 [`11_planning_safety.png`](../analysis/figs2/11_planning_safety.png))

| 실험 | plan L2↓ | map mAP↑ | det mAP↑ | NDS↑ | car_EPA↑ | car minADE↓ | car minFDE↓ | obj_col↓ | obj_box_col↓ |
|---|---|---|---|---|---|---|---|---|---|
| floor 3ep | 1.0337 | 0.4988 | 0.3723 | 0.4905 | 0.4418 | 0.7529 | 1.1869 | 0.89% | 0.28% |
| baseline 6ep | 0.7473 | 0.5161 | 0.3747 | 0.4963 | 0.4312 | 0.6885 | 1.0931 | 0.61% | 0.13% |
| **E13** | **0.6224** 🥇 | **0.5471** 🥇 | 0.4054 | **0.5186** 🥇 | 0.4681 | 0.6524 | **1.0312** 🥇 | 0.66% | **0.08%** 🥇 |
| E10 | 0.6632 | 0.5416 | **0.4084** 🥇 | 0.5155 | 0.4695 | 0.6521 | 1.0413 | 0.69% | 0.10% |
| E12 | 0.6594 | 0.5426 | 0.4074 | 0.5143 | **0.4703** 🥇 | **0.6501** 🥇 | 1.0329 | 0.67% | 0.08% |
| E11 | 0.9630 | 0.5296 | 0.3959 | 0.5115 | 0.4695 | 0.6554 | 1.0501 | **0.55%** 🥇 | 0.23% |

**지표별 최고:** E13이 9개 중 **5개**(planL2, map_mAP, NDS, car_minFDE, obj_box_col)에서 1위. E10은 det_mAP만, E12는 car_EPA·minADE, E11은 obj_col만.

- det_mAP는 E10이 근소 1위(0.4084)지만 E13(0.4054)과 차이 −0.7%로 미미.
- obj_col은 E11이 최저지만 E11은 planning L2가 크게 뒤짐(0.9630) → trade-off (보수적 궤적).

### 2.2 Detection 세부 (그림 [`13_det_perclass.png`](../analysis/figs2/13_det_perclass.png))

**mAP / NDS:** det mAP는 ATTITTUD(E10 0.4084, E12 0.4074)가 E13(0.4054)보다 근소 우위. 하지만 **NDS는 E13(0.5186)이 최고** — TP error가 좋기 때문.

**TP errors (낮을수록 좋음):**

| | mATE | mASE | mAOE | mAVE | mAAE |
|---|---|---|---|---|---|
| baseline 6ep | 0.6175 | 0.2841 | **0.5511** | 0.2642 | 0.1935 |
| **E13** | 0.5787 | **0.2772** | 0.5654 | **0.2391** | **0.1809** |
| E10 | **0.5771** | 0.2774 | 0.5964 | 0.2548 | 0.1811 |
| E12 | 0.5883 | 0.2803 | 0.5904 | 0.2477 | 0.1873 |
| E11 | 0.5982 | 0.2778 | 0.5559 | 0.2484 | 0.1840 |

> **주목: ATTITTUD가 orientation error(mAOE)를 악화시킨다.** E10 0.5964·E12 0.5904 vs E13 0.5654·baseline 0.5511. surgery의 gradient 왜곡이 방향 추정에 부정적 부작용을 낸 것으로 보인다. 반대로 E13은 velocity(mAVE)·attribute(mAAE)에서 최고.

### 2.3 Map / Motion (그림 [`14_map_motion.png`](../analysis/figs2/14_map_motion.png))

**Map (mAP_normal, 높을수록 좋음):** E13 0.5471 > E12 0.5426 > E10 0.5416 > E11 0.5296 > baseline 0.5161. 세부(ped_crossing/divider/boundary) 모두 E13이 baseline 대비 큰 폭 우위, boundary는 E13이 전체 1위(0.5736).

**Motion:** car_EPA·minADE는 E12가 근소 최고(0.4703 / 0.6501), car_minFDE·miss_rate는 E13 최고. 네 finetune 실험이 서로 근소 차이 — motion은 surgery에 거의 무관.

### 2.4 Collision vs horizon (그림 [`12_collision_ts.png`](../analysis/figs2/12_collision_ts.png))

6개 미래 시점(0.5–3.0s)별 obj_col/obj_box_col을 파싱. 원거리(3.0s)로 갈수록 모든 실험에서 충돌률 상승. E11이 근거리 obj_col 최저(보수적 planning), E13이 obj_box_col 전 구간 최저.

---

## 3. 학습 동역학 (그림 [`15_train_dynamics.png`](../analysis/figs2/15_train_dynamics.png))

### 3.1 종료 시점 loss (마지막 5% iter 평균)

| 실험 | total loss | plan_loss_temp_reg | det_loss_box | ego_loss_status |
|---|---|---|---|---|
| **E13** | **14.05** | **1.211** | **3.602** | **1.07** |
| E10 | 600.66 | 1.237 | 3.691 | 587.39 |
| E12 | 260.69 | 1.267 | 3.678 | 247.31 |

> **planning loss도 E13이 최저(1.211)** — planning이 실제로 더 잘 학습됐다. E10/E12의 거대한 total loss(600, 260)는 전적으로 **ego_loss_status 폭주**가 오염시킨 것.

### 3.2 ego_loss_status 폭주 — batch-1의 근본 문제

| 실험 | mean | median | **max** | n(>10) | n(>1e3) |
|---|---|---|---|---|---|
| **E13 (batch8)** | 1.29 | 1.279 | **2.0** | 0 | 0 |
| E10 (batch1, plan-primary) | 458.78 | 72.97 | 35,613 | 1,521 | 178 |
| E12 (batch1, det-primary) | 1,483.36 | 72.95 | **838,722** | 1,513 | 163 |

- **batch8(E13)은 ego가 완전히 안정**(max 2.0). batch1은 median조차 73으로 폭주 상태.
- **det-primary(E12)가 plan-primary(E10)보다 24배 더 폭주**(max 838,722 vs 35,613). 이는 surgery 진단과 정확히 일치한다: **plan-primary는 ego의 충돌 성분을 α=−1로 반전시켜 *우연히* 발산을 억제**하지만, det-primary는 그 억제가 없어 훨씬 심하게 폭주.

### 3.3 NaN / skip 이벤트

| 실험 | NaN/skip 관련 로그 라인 |
|---|---|
| E10 (batch1) | **2,043** |
| E12 (batch1) | **3,943** |
| **E13 (batch8)** | **0** |

batch-1 per-sample forward는 ego 발산으로 SafeOptimizerHook의 step-skip을 수천 회 유발. batch8은 미니배치 평균이 발산을 희석해 **한 번도 skip하지 않음**. 이것이 E13가 더 나은 실용적 이유 중 하나다.

---

## 4. Surgery 진단 요약 (jsonl, 별도 노트북)

per-sample surgery 로그(`attittud_persample.jsonl`) 분석 결과 ([`attittud_surgery_analysis.ipynb`](../analysis/attittud_surgery_analysis.ipynb)):

- **개입 강도 미미:** aux gradient 중 primary와 충돌하는 성분(`frac_bad`)이 det/map/motion에서 **0.03–0.1%**. surgery가 실제로 바꾸는 비율(`mod_ratio`)도 2–5%.
- **직교성이 근본 원인:** aux와 plan gradient의 cos ≈ 0 (std 0.015–0.028, 랜덤의 15–28배라 결합은 *실재*하나 sign-cancelled). 에너지의 99.9%가 primary subspace 밖(neutral).
- **유일한 실질 개입은 ego:** ego만 plan과 유의미하게 충돌(frac_bad 5.5%, cos² 12.9%). 이 ego 반전이 §3.2의 우연한 안정화 부수효과를 냄.
- **compass 무관성:** primary를 plan(E10)→det(E12)로 바꿔도 surgery 흔적이 대칭(swap-pair cos 0.0004 vs 0.0003). compass 선택은 약한 lever.

---

## 5. 핵심 결론

1. **surgery는 무익하거나 미세하게 유해하다.** aux⊥plan 구조상 개입 여지 자체가 없고(neutral 99.9%), 실제로 뺐더니(E13) planning이 −6.1% 개선됐다. 게다가 orientation error(mAOE)를 악화시키는 부작용까지 있었다.

2. **batch-1 요구가 결정적 손해다.** surgery는 per-sample gradient를 필요로 해 batch-1을 강제하는데, 이것이 ego head를 발산시켜(NaN skip 2,000–4,000회) 학습을 오염시킨다. batch8은 발산 0회.

3. **det-primary가 plan-primary보다 위험하다.** ego의 우연한 안정화 효과가 사라져 ego가 24배 더 폭주(E12 max 838,722). 성능은 비슷하나 학습 안정성은 더 나쁨.

4. **planning의 진짜 lever는 surgery가 아니다.** 순수 finetune만으로 floor 대비 −39.8%. planning 개선은 gradient 조작이 아니라 학습 예산/프로토콜(및 표현 수준)에서 나온다.

---

## 6. 다음 방향

- **실전 경로:** E13 레시피(batch8/16 순수 finetune)를 더 길게 또는 lr 스케줄 조정으로 밀기. 가장 안전하고 이미 최고 성능.
- **연구 경로:** aux⊥plan은 gradient가 아니라 **공유 표현의 구조적 사실**. E11(aligned-aux)이 표현 정렬을 시도했으나 실패(L2 0.9630). 결합을 lever화하려면 aux와 plan이 표현을 실제로 공유하도록 하는 **아키텍처/보조손실 재설계**가 필요.
- **X1(rank1-PCGrad) 보류:** compass 차원(1 vs 6)만 바꾸는 실험. 진단상 개입이 오히려 더 미미(neutral↑)하고 batch-1 발산을 상속하므로, E13를 이길 가능성이 없어 34시간 학습은 비권장. config는 재현용으로 보존([`X1_rank1_pcgrad_b1_ft3ep.py`](../projects/configs/experiments/X1_rank1_pcgrad_b1_ft3ep.py)).

---

## 7. 재현 정보

| 자원 | 경로 |
|---|---|
| Eval 로그 | `HiP-AD/eval_{base_3ep,base_6ep,attittud_6ep_final}.log`, `HiP-AD-aligned/eval_e1{1,2,3}_*.log` |
| 학습 로그 | `HiP-AD-aligned/work_dirs/exp/E1{0,2,3}_*/[date].log.json` |
| surgery 로그 | `HiP-AD-aligned/work_dirs/exp/E1{0,2}_*/attittud_persample.jsonl` |
| 종합 노트북 빌더 | `analysis/build_comprehensive_notebook.py` |
| surgery 노트북 빌더 | `analysis/build_attittud_notebook.py` |
