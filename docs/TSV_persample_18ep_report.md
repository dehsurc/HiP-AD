# Per-Sample TSV-Gradient 분석 보고서 (18ep, N=100)

> 대상: HiP-AD stage-2 18ep, 공유 `inter_gnn` 어텐션 가중치
> 도구: [tools/gradient_analysis/tsv_gradient_persample.py](../tools/gradient_analysis/tsv_gradient_persample.py)
> 산출물: `gradient_analysis_results/tsv_gradient/persample_18ep/`
> 관련: [TSV_RUNBOOK.md](TSV_RUNBOOK.md) §5(per-sample 분석), [TSV_gradient_analysis_report.md](TSV_gradient_analysis_report.md)

---

## 1. 목적

기존 `tsv_gradient.py`는 **N개 샘플의 gradient를 평균**해 하나의 `Gbar_task`를 만들고 그것 하나를 SVD → task쌍당 **결과 1개**를 냈다. 평균 코사인 `cos(Gbar_plan, Gbar_aux) ≈ 0`(직교처럼 보임).

이 보고서는 반대로 **샘플 100개를 개별 계산**(평균 없음)해 **결과 100개의 분포**를 본다. 질문: 그 ~0 직교는 (a) 모든 샘플이 진짜 직교라서인가, 아니면 (b) 샘플마다 강하게 결합하지만 부호가 상쇄돼서인가? 평균은 이 둘을 구분할 수 없고, per-sample 분포만이 구분한다.

---

## 2. 방법

### 2.1 샘플당 연산 흐름
1. `build_dataloader(batch_size=1, shuffle=True, seed=42)` — nuScenes train scene 1개.
2. `restore_temporal_state` — InstanceBank cache·`run_step`·DN sampler·RNG를 **매 샘플 동일 스냅샷으로 리셋** → 각 샘플은 동일 확률조건의 cold-start frame, 변동은 오직 scene 내용.
3. `forward_losses` — `model.train()` + Dropout/BN/DeformableFeatureAggregation은 eval 고정.
4. `split_losses(task)` — task 접두사 loss 합산 → 스칼라 `L_det/map/motion/plan`.
5. `torch.autograd.grad(L_task, params)` — task loss를 **공유 inter_gnn 가중치**로 미분. forward 1번 → backward 4번(retain_graph).

### 2.2 SVD 입력 = "gradient 행렬" 그 자체
추적 대상: inter_gnn 어텐션 **6개 층**(operation_order idx 2/14/26/38/50/62) × {`in_proj_weight`(768×256, fused QKV), `out_proj.weight`(256×256)} = raw 12개. `split_qkv`가 `in_proj_weight`를 Q/K/V(각 256×256)로 분리 → **최종 24개 행렬**.

SVD 대상은 가중치가 아니라 **`G_plan = dL_plan/dW` (256×256) gradient 행렬**. `U,S,Vh = svd(G_plan)`.

### 2.3 지표 (샘플·행렬별)
| 지표 | 수식 | 의미 |
|---|---|---|
| **flat_cos** | `⟨G_plan,G_aux⟩_F / (‖G_plan‖‖G_aux‖)` | 전체 mode 정렬 (rank 무관) |
| **mode0_align** | `u0ᵀG_aux v0 / ‖G_aux‖` ∈[−1,1] | plan **최강 descent 방향** `u0v0ᵀ`에 aux 정렬 = `cos(G_aux, plan top mode)` |
| **effrank** | `(Σσ)²/Σσ²` | 참여율 유효 rank — **단일샘플 confounded**(rank≤토큰수), 참고용 |
| **global cos** | 24행렬 concat 후 `cos(plan,aux)` | **샘플당 1개** 대표값 (M2 conflict와 동일 스코프) |
| cos_of_means (기준) | `cos(mean G_plan, mean G_aux)` | 기존 평균 파이프라인 지표 (~0) |

`mode0_align`과 `flat_cos` 관계: mode0의 flat_cos 기여 = `(σ0/‖G_plan‖)·mode0_align`. plan이 mode0-지배면 둘이 일치, 아니면 괴리.

### 2.4 셋업
- ckpt `ckpts/rev_nusc/70+stage2_18ep.pth`, config `E2_E1_stage2_18ep_from70ep.py`, target `inter_gnn`, N=100.
- env conda `hipad` (torch 1.13.0+cu117 / mmcv 1.7.1), GPU 0 (`cuda:0`), fp32.
- 런타임 **8.1분, 에러 0** (counts det/map/motion/plan = 100/100/100/100).

---

## 3. 결과

### 3.1 헤드라인 — 전역(concatenated) per-sample 코사인: **샘플당으로도 진짜 직교**
| aux | cos_mean | cos_std | median | q05 | q95 | frac_neg | frac_strong(\|cos\|>0.1) | cos_of_means(기존) |
|---|---|---|---|---|---|---|---|---|
| det | +0.0006 | 0.017 | +0.0012 | −0.028 | +0.026 | 0.46 | **0.00** | +0.0136 |
| map | +0.0033 | 0.017 | +0.0037 | −0.023 | +0.037 | 0.41 | **0.00** | +0.0036 |
| motion | +0.0014 | 0.020 | +0.0016 | −0.028 | +0.034 | 0.44 | **0.00** | +0.0012 |

inter_gnn 전체를 묶으면 **100샘플 중 단 하나도 \|cos\|>0.1이 없음**. 전역 스코프에선 평균뿐 아니라 per-sample로도 **genuine 직교** (부호-상쇄 아님).

### 3.2 국소화 — 결합은 **첫 층(dec0=ig2)의 K/Q**에 집중
**층별 \|cos\| (역할·aux 평균):**
| layer | ig2(dec0) | ig14 | ig26 | ig38 | ig50 | ig62 |
|---|---|---|---|---|---|---|
| abs_cos | **0.094** | 0.021 | 0.027 | 0.029 | 0.024 | 0.015 |

**역할별 \|cos\|:** K **0.045** > Q 0.036 ≈ V 0.036 > out 0.023.

→ 결합은 **첫 inter_gnn 층에 집중**(타 층 대비 3~4×), 그 안에서도 **Key/Query 투영**이 최다.

### 3.3 부호-상쇄 정량 (top 결합 weight, 100샘플)
잔존율 = `|signed mean| / mean|cos|` = 부호평균 후 **살아남는 비율** (상쇄율 = 1−잔존율).
| weight × aux | signed mean | mean\|cos\| | frac_neg | 잔존율(=\|mean\|/mean\|cos\|) |
|---|---|---|---|---|
| ig2.in_proj_k × det | +0.009 | **0.192** | 0.45 | 0.05 |
| ig2.in_proj_k × motion | −0.005 | **0.172** | 0.51 | 0.03 |
| ig2.in_proj_q × det | +0.009 | **0.162** | 0.44 | 0.06 |
| ig2.in_proj_q × motion | −0.001 | **0.137** | 0.50 | 0.01 |
| ig2.in_proj_k × map | +0.005 | **0.130** | 0.49 | 0.04 |
| ig2.in_proj_q × map | +0.007 | **0.117** | 0.45 | 0.06 |

**ig2 K/Q에서 샘플당 \|cos\|~0.13–0.19의 실재 결합**이 있으나, 부호가 동전던지기(frac_neg≈0.5, 잔존율 0.01–0.06)라 **평균의 94–99%가 상쇄**되어 ~0이 된다. → 이것이 "직교=부호-상쇄"의 정체이며, 위치는 **첫 층 K/Q**로 특정된다.

### 3.4 mode0_align (plan 주 하강축 정렬)
| weight × aux | \|mode0_align\| | \|flat_cos\| | 해석 |
|---|---|---|---|
| ig2.in_proj_k × det | 0.196 | 0.192 | 거의 일치 → plan **mode0-지배**(σ0/‖Gp‖~0.74, 에너지 55%) |
| ig2.in_proj_q × det | 0.078 | 0.162 | 괴리 → aux가 plan의 **비-top mode**로 결합 |

전역 `corr(mode0_align, flat_cos)=0.837`. `mode0_align`도 **부호-상쇄**(frac_neg 0.48~0.58) — plan의 주 하강축에서조차 aux 정렬 부호가 샘플마다 반전.

### 3.5 gradient norm 비대칭
ig2.in_proj_q 100샘플 median: plan 0.286 / **det 2.10 (≈7× plan)** / map 0.072 / motion 0.113. det가 gradient 크기를 지배하나 배율은 **약 7배(median)**. (초기 단일샘플에서 관찰된 630×는 그 샘플의 plan_fro가 유난히 작았던 outlier.)

### 3.6 effrank (confounded, 참고)
plan 단일샘플 effrank ~7.6 / aux ~11–13 (of 256). 단일샘플 gradient는 저랭크 → 스펙트럼 해석 안 함, top-mode(`u0,v0`)만 사용 (RUNBOOK §5).

### 3.7 3ep 대조 — 위치·크기는 epoch 불변, **부호-상쇄는 학습하며 심화**
동일 셋업으로 3ep ckpt(`70+stage2_3ep.pth`) 100샘플 추가 실행(5.7분, err 0). config 동일.

**변하지 않는 것 (크기·위치):**
- 전역 per-sample: 3ep도 genuine 직교(frac_strong=0, std~0.02).
- 층별 \|cos\|: dec0(ig2) 0.092 vs 나머지 0.017–0.028 — 18ep(0.094)와 **거의 동일**.
- 역할별 \|cos\|: K 0.044 / Q 0.036 / V 0.036 / out 0.023 — 18ep와 **동일**.
- ig2 K/Q의 mean\|cos\|: det 0.192 / motion 0.13–0.17 — **양 epoch 동일**. → 결합의 **위치·강도는 3ep에 이미 완성**, 18ep까지 불변.

**변하는 것 (부호 잔존율 = 평균에서 살아남는 정도):**
| weight × aux | 3ep signed / frac_neg / 잔존율 | 18ep signed / frac_neg / 잔존율 |
|---|---|---|
| ig2.k × det | +0.047 / 0.40 / **24%** | +0.009 / 0.45 / **5%** |
| ig2.q × det | +0.043 / 0.39 / **28%** | +0.009 / 0.44 / **6%** |
| ig2.k × motion | −0.031 / 0.52 / **18%** | −0.005 / 0.51 / **3%** |
| ig2.q × motion | −0.023 / 0.51 / **17%** | −0.001 / 0.50 / **0.6%** |
| ig2.q × map | +0.016 / 0.47 / 16% | +0.007 / 0.45 / 6% |

→ **부호 방향은 두 epoch 공통** (det=plan 강화 +, motion=plan 반대 −, map=중립), 하지만 **일관 성분(잔존율)이 3ep 24–28% → 18ep 5–6%(det), 17–18% → ~1–3%(motion)로 축소**. 즉 per-sample 결합 크기(\|cos\|)는 그대로인데 **학습이 부호를 점점 balance시켜 더 완전히 상쇄**시킴. 이는 메모리 발견 5의 "plastic 3ep가 결합을 더 키움"과 정합하며, 왜 18ep 평균 지표가 3ep보다 더 깨끗이 ~0인지 설명한다.

---

## 4. 해석

**직교는 두 층위로 나뉜다:**
1. **전역(inter_gnn 전체)**: "aux gradient step이 plan 전체를 움직이나?" 수준에선 per-sample로도 genuine 직교 (frac_strong=0).
2. **국소(첫 층 K/Q)**: 샘플당 \|cos\|~0.15의 실재 결합이 존재하나 **부호-상쇄**로 평균 ~0. 평균/concat 파이프라인은 (norm-가중 희석 + 부호 선평균으로) **원리적으로 이걸 못 본다** — per-sample \|cos\|만이 드러낸다. 이 실험의 핵심 가치.

**기존 발견과의 정합:**
- 발견 1(부호-상쇄, mean-abs cos 0.02~0.03 @ dec0/1/2)을 **dec0 K/Q로 위치특정·정량화**(0.13~0.19, 상쇄율 0.03~0.06).
- 결합이 **Key** 투영에 집중된 것은 발견 4/5의 NTK 함수결합·"K 조작" 맥락과 일치. 단 발견 5(Stage A)에서 K를 키워도 planning 개선 X → 이 결합은 **진단적/구조적 사실**이지 training lever가 아님. 본 관찰(부호-상쇄 국소성)은 그 결론을 **관측 층위에서 재확인**한다(부호가 상쇄되므로 평균 gradient 위 method는 잡을 게 없다).

---

## 5. 한계
- 단일 ckpt(18ep, saturated), target=inter_gnn 한정. 다른 epoch/backbone·neck에서 국소화 패턴은 미확인.
- \|cos\|~0.15는 **중간 강도** 결합(강하지 않음). "결합 실재"는 random 대조(발견 1의 ~25×) 기준의 상대적 진술.
- per-sample effrank·spectrum은 confounded라 해석 제외.
- 관측적 분석 — 인과 주장 아님(인과는 발견 5 Stage A 참조).

---

## 6. 재현
```bash
cd /home/yongjae/e2e/HiP-AD-pcgrad
CUDA_VISIBLE_DEVICES=0 /home/yongjae/miniconda3/envs/hipad/bin/python \
  tools/gradient_analysis/tsv_gradient_persample.py --n 100 \
  --ckpt-path ckpts/rev_nusc/70+stage2_18ep.pth \
  --out gradient_analysis_results/tsv_gradient/persample_18ep
```
옵션: `--target {inter_gnn,backbone,neck,backbone_neck}`, `--n`, `--ckpt-path`, `--config`, `--out`.

## 7. 산출물
`gradient_analysis_results/tsv_gradient/persample_18ep/`
| 파일 | 내용 |
|---|---|
| `per_sample_global.csv` | 100×3 — 샘플당 전역 코사인 (헤드라인) |
| `per_sample_by_matrix.csv` | 행렬×aux — cos_mean/abs_cos_mean/frac_neg/align_mean (§3.2–3.3 원자료) |
| `per_sample_metrics.csv` | 100×24×3 — flat_cos·mode0_align·mode0_proj·effrank 전부 |
| `per_sample_summary.csv` | aux별 전역 분포 요약 |
| `per_sample.pt` | raw rows + mean_grads + counts |
