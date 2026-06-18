# Scene-feature (data-feature-level) gradient 분석 — 종합 결론 + 참고자료 (2026-06-17)

> aux(det/map/motion) ↔ plan gradient 상호작용이 **데이터 장면(scene)의 특징에 따라 구조화되어
> 있는가**를 묻는 분석의 최종 종합. 결론: **부호(방향) 기준 scene-조건부 구조는 1·2세대 모두 NULL.**
> 장면이 좌우하는 것은 상호작용의 *크기*뿐이고, *부호*는 모델 상태(checkpoint)가 결정한다.

## 0. 공통 동기 — 무엇을 풀려고 했나

per-batch aux→plan gradient 신호 `grad_dot = g_aux·g_plan` 는 **크기는 실재하지만(med|Δ|≈2.5e-5,
ULP의 ×200) 부호가 ~50/50으로 갈려 평균이 0**으로 사라진다(부호 상쇄). 핵심 질문:

> 이 50/50 부호 분할은 **(a) 순수 노이즈**(→ per-scene 가중치 불가, 적분 net bias만 레버)인가,
> **(b) scene 특징으로 구조화**(→ "이런 장면에선 det가 plan을 돕는다"는 조건부 task-weighting knob 존재)인가?

문헌 근거(검증 완료): **ForkMerge Finding 1**(gradient cosine 충돌은 negative transfer를 예측하지
못함 → validation 기반 측정 필요)이 직접 근거. PCGrad Def.1은 *per-step cosine을 충돌로 정의*하는
출처(측정 대상), TAG는 *task affinity가 학습 내내 변하므로 시간 적분 필요*의 근거. (§참고자료 D)

---

## 1. 종합 결론표 (2026-06-17 갱신)

| # | 분석 | scene 특징(독립변수) | 비교 대상(종속변수) | 결론 | 상태 |
|---|---|---|---|---|---|
| 1 | 1세대 부호 일관성 | per-scene 부호(layer/ckpt) | grad_dot 부호 | 부호는 **상태-의존**, scene-고정 아님 → static 가중치 기각, online만 가능 | ✅ SOLID |
| 2 | 1세대 난이도 조건부 | plan baseline_loss | grad_dot / \|grad_dot\| | **크기**는 plan-hard에 집중, **부호**는 난이도 무관 | ✅ SOLID |
| 3 | 1세대 cross-aux | task 간 장면 일치 | grad_dot (aux쌍) | det~motion만 약한 협력(ρ=0.20), map 독립 | ✅ SOLID |
| 4 | 1세대 cosine tail / conflict | 반정렬 장면 군집 | conflict cosine | plan과는 near-null, 숨은 충돌집단 없음(aux끼리만 구조) | ✅ SOLID |
| 5 | 2세대 persample coupling | 밀도·회전·곡률·속도·관련성 | **cos(g_aux,g_plan)** | 어떤 물리 scene feature도 부호를 못 푼다 → 조건부 knob **NULL** | ✅ 완료 |
| 6 | 2세대 persample **perf** | 위 feature + plan 난이도 | **cos ↔ loss_plan** | cos는 planning 성능과 무관; 유일 후기 신호는 **norm 기하 confound** → knob **NULL** | ✅ **완료(갱신)** |
| 7 | trajectory probe | (파라미터공간, 물리 아님) | k-step L_plan 궤적 | 결론 없음 — NTK 라인으로 대체 | ⚠️ smoke만, 중단 |

**한 줄 결론**: scene 특징으로 aux↔plan **부호(방향)를 예측·조건화하려는 시도는 1·2세대 모두 NULL**.
부호는 장면이 아니라 **모델 상태**가 결정하고, 장면이 좌우하는 건 상호작용의 **크기**(plan-hard 집중)뿐.
정적 per-scene weighting은 기각, 남은 가능성은 **online/단계별 weighting**과 **비-gradient(rescore/표현) 채널**.

---

## 2. 행별 상세

### 행 1 — 부호는 scene-고정이 아니라 상태(checkpoint)-의존
- **방법**: `inter_gnn_b1000_pcgrad`의 4 ckpt(1/3/9/18ep) × 1000 batch × 3 decoder layer.
  같은 (scene) 부호가 **layer 간** / **checkpoint 간** 일치하는 비율을 shuffled null과 비교(binomtest).
- **전제 검증(SOLID)**: 고정 seed로 batch_idx가 ckpt 간 동일 장면을 가리킴 — plan baseline_loss의
  ckpt 간 Spearman **0.59~0.91**(p<1e-94)로 실증.
- **핵심 수치**(`sign_consistency.csv`):
  | 축 | det | map | motion |
  |---|---|---|---|
  | cross-layer(3) 동일 ckpt | +7.0%p (p=1.1e-23) | +9.7%p (p=5.6e-43) | **+14.3%p (p=9.6e-88)** |
  | cross-ckpt(4) 동일 scene | +1.3%p (p=0.020) | +1.8%p (p=0.0024) | +0.4%p (**n.s.**) |
- **해석**: 한 시점의 부호는 layer 간 강하게 일관(=실재하는 per-scene 방향)이나, 학습 단계를 넘으면
  거의 사라짐(cross-ckpt excess가 5~30배 작음, motion은 n.s.). → 부호는 **상태가 결정** → 정적
  per-scene 가중치 기각, online(상태 인지) weighting만 가능.

### 행 2 — 상호작용 크기는 plan-hard에 집중, 부호는 난이도 무관
- **방법**: plan 난이도 = per-batch plan baseline_loss. `Spearman(grad_dot, plan_loss)`(부호)와
  `Spearman(|grad_dot|, plan_loss)`(크기)를 source별로(`conditioning.csv`).
- **핵심 수치**:
  - **크기** `|grad_dot|~plan_loss`: det **0.196**(p=7.8e-104), map **0.233**(p=5.1e-148), motion **0.156**(p=4.3e-66) — 전부 강함.
  - **부호** `grad_dot~plan_loss`: det −0.025(p=0.006, 효과 무시), map +0.010(n.s.), motion −0.003(n.s.).
- **해석**: 어려운 장면일수록 상호작용이 **크지만**, 그 **방향은 난이도로 예측 불가**. → 50/50 부호
  분할이 "크기가 큰 곳"의 문제가 아니라 진짜 상태-구동 효과임을 보강.

### 행 3 — aux task끼리 같은 장면을 돕는가 (부분적)
- **방법**: `Spearman` between aux sources' grad_dot, n=12000 (`cross_aux_corr.csv`).
- **핵심 수치**: **det~motion ρ=0.199 (p=4.4e-107)** / det~map ρ=0.017 (n.s.) / map~motion ρ=−0.004 (n.s.).
- **해석**: det이 plan을 돕는 장면에서 motion도 돕는 경향(약함). **map은 독립** — 모든 aux가 공유하는
  단일 "plan-friendly 장면"은 없음. (motion query가 det feature로 구성되는 구조·TG의 det+motion 결합과 정합.)

### 행 4 — 숨은 destructive-conflict 하위집단 없음; 구조는 aux끼리만
- **핵심 수치**:
  - plan-pair cosine 꼬리: **P(cos<−0.3) < 0.06%**, 음/양 꼬리 대칭 (`cosine_tails.csv`).
  - conflict ratio **task→plan 전부 0.47~0.54**(0.5 랜덤선 근처) ↔ **aux-aux는 강한 구조**(det~motion 0.14~0.17 = 매우 협력) (노트북 §4, `inter_gnn_b1000_pcgrad/ckpt_*/conflict/`).
- **해석**: ~0 평균 아래 강한 반정렬 장면 군집은 **없음**. plan과의 장면별 net 충돌 구조도 없음.
  구조는 plan이 아니라 **aux끼리**(특히 det~motion)에 존재.

### 행 5 — per-sample coupling: 물리 scene feature가 부호를 푸는가 (NULL)
- **방법**(`persample_coupling.py`+`persample_analyze.py`): bs=1 단일 장면마다 cos(g_aux,g_plan)와
  **norm-matched 랜덤방향 대조 cos_rand**(거짓말탐지기)를 기록. 6개 strata(밀집 `n_near15≥6`/회전/
  고관련 `n_relevant≥1`/고곡률 `map_curv≥med`/고속 `ego_speed≥med`/`ctrl_random`)에서
  `mean(cos|Z)−mean(cos|~Z)` + 2000-perm test. 연속 feature는 Spearman vs perm floor.
- **판정 규칙**: det·map·motion 모두 유의 + **aux=rand엔 없음** + **ctrl_random엔 없음** → 구조적.
- **결과**: 60개 strata 검정 중 p<0.05는 4개뿐, 모두 |cos| 노이즈 스케일·ckpt 비재현. **랜덤 대조도
  같은 크기의 "유의" hit을 냄**(18ep turn p=0.048; 9ep rand n_near15 ρ=−0.136) → 진짜 aux도 노이즈 floor.
- **해석**: **어떤 물리 scene feature도 cos 부호를 예측 못 함** = 조건부 weighting knob NULL.

### 행 6 — per-sample performance: cos가 planning 성능을 예측하는가 (✅ 완료, NULL)
- **이번에 실행**: `persample_perf.py`(fp32, bs=1, n=1500) → `persample_perf_{1,3,9,18}ep.csv`
  (valid 1221/1245/1253/958). cos(g_aux,g_plan)**와 그 장면의 loss_plan을 동시 기록**.
- **축 A (상쇄, 재확인)**: mean(cos)≈0, frac_neg 0.47~0.52, |cos|/랜덤 **22~39배**(map 최대). fp32로
  NaN 편향 제거 후에도 부호 상쇄 유지.
- **축 B (cos↔성능)**: `Spearman(cos_aux, loss_plan)` 12개 검정 → **BH-FDR 생존은 9ep/map, 18ep/map 둘뿐**.
  그러나 셋 다 knob 실격:
  1. **부호 ckpt 간 뒤집힘**: cos_map↔loss_plan = +0.048/+0.054/−0.085/−0.108 (1/3/9/18ep).
  2. **18ep는 gplan_norm 기하 confound**: `gplan_norm~loss_plan`이 0.34→0.55→0.64→**0.67**로 상승.
     gplan_norm 통제 시 18ep map −0.107 → **−0.031 붕괴**(랜덤 대조는 −0.086→−0.095 유지) → aux-특이
     정렬이 아니라 기하(어려운 장면 = 큰 ‖g_plan‖ = 어떤 방향이든 cos 저하).
  3. **효과크기 무시 가능**: 생존 \|ρ\|≤0.11 → 분산 1.3% 미만.
- **유일한 약한 실마리**: 9ep map만 (랜덤 대조 통과 + gplan_norm 부분 생존 partial −0.046 + 사분위
  frac 0.59→0.48 감소) → "9ep에서 plan-hard 장면일수록 map이 plan과 더 충돌"로 **TG의 map-해로움을
  방향적으로 약하게 보강**. 단 3개 ep 중 1개에서만, 1/3ep는 부호 반대 → 일반화 불가.
- **해석**: **per-scene gradient 정렬은 planning 성능과 사실상 무관** → 성능 기반 knob도 NULL.
- **시각화**: `figures/feature_level_null.png`, `figures/no_performance_knob.png` (§참고자료 C).

### 행 7 — trajectory probe (smoke만, 중단)
- "trajectory"는 **물리 주행궤적이 아니라 파라미터공간 가상 최적화 경로**(aux gradient 방향 k-step
  가상 이동하며 L_plan·곡률 추적, 단일스텝 null↔1ep TG non-null 간극을 2차/곡률로 설명 시도).
- 실제 4 ckpt run은 batch 2/16에서 중단, CSV 0개. 유일 데이터는 2-batch `traj_smoke.csv`(결론 없음).
- **권고: 재개하지 말 것.** WHETHER는 TG가 이미 답했고, 2차 메커니즘은 가상궤적보다 **NTK 라인
  (`couple/`, `ntk_*`, `ntk_viz.ipynb`)**이 더 원리적. (프로젝트 메모리 "Q4(NTK) 진행 중"과 정합.)

---

## 3. 열린 항목

- ✅ (해결) persample_perf 축 B — NULL로 종결.
- ▶ 남은 productive 방향: **online/단계별(early-training) weighting**, 그리고 **비-gradient
  채널**(rescore/표현). gradient-level per-scene weighting은 1·2세대 종합으로 **기각**.
- ▶ 2차 결합 메커니즘: trajectory probe가 아니라 **NTK 라인**으로.

---

## 참고자료 (Reference materials)

### A. 1세대 sign-structure (분석 코드 + 결과)
- 코드: `tools/gradient_analysis/analyze_plan_sign_structure.py`
- 결과: `gradient_analysis_results/plan_sign_structure/` —
  `report.md`, `sign_consistency.csv`(행1), `conditioning.csv`(행2), `cross_aux_corr.csv`(행3),
  `cosine_tails.csv`(행4), `net_bias.csv`
- conflict 원천: `gradient_analysis_results/inter_gnn_b1000_pcgrad/ckpt_{1,3,9,18}ep/conflict/*_summary.csv`
- 시각화 노트북: `2026-06-14-tg-sign-structure-analysis.ipynb` (§3 Sign Structure, §4 Conflict)

### B. 2세대 per-sample (코드 + 데이터)
- coupling(행5): `tools/gradient_analysis/persample_coupling.py`, `persample_analyze.py` →
  `gradient_analysis_results/persample_{1,3,9,18}ep.csv` (cos-only)
- performance(행6): `tools/gradient_analysis/persample_perf.py`, `persample_perf_analyze.py` →
  `gradient_analysis_results/persample_perf_{1,3,9,18}ep.csv` (cos + loss_plan + scene feats)
- scene feature 정의: `tools/viewer/strata_analysis.py`(scene_features), `tools/viewer/relevance_analysis.py`(relevance_feats)
- trajectory(행7): `tools/gradient_analysis/trajectory_probe.py`, `trajectory_analyze.py`, `traj_smoke.csv`

### C. 시각화 (이번 생성)
- `figures/feature_level_null.png` — (A) 부호 상쇄 히스토그램, (B) scene feature Spearman vs 노이즈 floor
- `figures/no_performance_knob.png` — (A) cos↔loss_plan ckpt별 부호 뒤집힘, (B) gplan_norm 기하 confound
- 생성 스크립트: `tools/gradient_analysis/plot_feature_null.py`

### D. 외부 문헌 (검증 완료 — arXiv 원문 확인)
- **ForkMerge** (직접 근거): Jiang et al., *Mitigating Negative Transfer in Auxiliary-Task Learning*,
  NeurIPS 2023, arXiv:2301.12618. **Finding 1**: "Negative transfer is not necessarily caused by
  gradient conflicts and gradient conflicts do not necessarily lead to negative transfer." → GCS(gradient
  cosine)는 transfer 예측 불가, validation 기반 fork-merge 측정으로 대체.
- **PCGrad** (측정 대상의 정의): Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020,
  arXiv:2001.06782. **Definition 1**: cos φᵢⱼ<0 ⇔ conflicting gradients(per-step cosine을 충돌로 정의·투영).
- **TAG** (시간 적분 필요): Fifty et al., *Efficiently Identifying Task Groupings for MTL*, NeurIPS 2021,
  arXiv:2109.04617. inter-task affinity = 1-step lookahead loss-ratio를 **학습 전체에 걸쳐 평균**;
  affinity가 학습 단계마다 변함을 보임(본 프로젝트의 "부호=상태 의존"과 정합).
- (cosine 회의론, 보강) Kurin et al., NeurIPS 2022, arXiv:2201.04122; Xin et al., NeurIPS 2022, arXiv:2209.11379.

### E. 설계/배경 문서
- `2026-06-12-plan-relation-sign-structure-and-tg-design.md` (1세대 §A + TG 설계)
- 본 문서: `2026-06-17-scene-feature-analysis-conclusions.md`
