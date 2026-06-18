# Plan↔aux 관계: 부호-구조 분석 결과 + Transfer-Gain 실험 설계 (2026-06-12)

## 배경 — 왜 이 분석인가

기존 §4/§5 진단의 한계가 데이터로 확정됨:

1. **aux→plan per-batch 신호는 float floor 위에 있음** (med|Δ|≈2.5e-5 vs ULP 1.2e-7,
   ×200) — "floor에 깔려 안 보인다"는 이전 해석은 과장이었음.
2. **진짜 문제는 부호 상쇄**: batch별 부호가 49.5/50.3으로 갈려 평균이 0으로 수렴.
3. 문헌 (PCGrad Def.1, ForkMerge Finding 1, TAG): cosine/1-step 신호는
   "누가 plan을 돕는가"의 verdict가 될 수 없음 → **적분(학습) 기반 측정 필요**.

이에 두 갈래 실행: (A) 부호 상쇄의 구조성 판정(기존 데이터, 무비용),
(B) ForkMerge식 leave-one-aux-out Transfer Gain(실제 1ep 재학습).

## (A) 부호-구조 분석 결과

스크립트: `tools/gradient_analysis/analyze_plan_sign_structure.py`
출력: `gradient_analysis_results/plan_sign_structure/`
지표: analytic `grad_dot`(g_aux·g_plan) — floor·곡률 오염 없음. raw variant, steps=1.

### A1. Net bias — pooled로는 0, 그러나 시간가변 구조 존재

| source | pooled(ALL) | 1ep | 3ep | 9ep | 18ep |
|---|---|---|---|---|---|
| det | n.s. | **OPPOSED** (−3.4e-3, CI<0) | **ALIGNED** (+2.3e-3, CI>0) | n.s. | n.s. |
| map | n.s. | ALIGNED (약) | n.s. | ALIGNED (약) | n.s. |
| motion | n.s. | n.s. | n.s. | n.s. | n.s. |

- 이전에 본 "미세 net bias −2.4e-7 (helps)"는 **pooled에서 유의하지 않음**.
- 대신 **det↔plan 관계가 학습 단계에 따라 뒤집힘**: 초기(1ep) 반대 → 3ep 정렬 →
  수렴기(9/18ep) 소멸. delta 교차검증도 동일 패턴 (1ep HURTS → 3ep HELPS → n.s.).

**다중비교 보정 후 확정 등급 (검증 워크플로 재계산, sign test + Bonferroni m=15):**

| 플래그 | 보정 후 | 등급 |
|---|---|---|
| det 1ep OPPOSED/HURTS | p_sign=1.5e-5 → 2.2e-4 (grad_dot), 9.0e-5 (delta) | **SOLID** — 양 지표 Bonferroni 생존 |
| det 3ep ALIGNED/HELPS | corrected 0.10; BH-FDR q=0.05 통과 | 경계 (FDR만 생존) |
| map 1ep ALIGNED | corrected 0.45, Wilcoxon p=0.79 모순 | **기각** |
| map 9ep ALIGNED | corrected 0.14–0.23 | **기각** |

- ⚠ net_bias.csv의 delta CI 끝값은 float32 격자(atom) 인공물 — 판정은 sign test
  기준으로 할 것 (위 표). pooled "ALL n.s."는 "효과 없음"의 증거가 아니라 1ep/3ep
  부호 반전이 상쇄된 결과로 읽어야 함.

### A2. 부호는 scene-고정이 아니라 상태(checkpoint)-의존

| 축 | 일관성 excess vs null | 해석 |
|---|---|---|
| cross-layer(3) 동일 ckpt | **+7~14%p, p≤8e-23** (fair null에도 불변) | 한 시점의 부호는 layer 간 일관 = 실재하는 per-scene 방향 — **rock solid** |
| cross-ckpt(4) 동일 scene | map +1.7%p (perm p=0.0026, 가족보정 생존), det +1.4%p (perm p=0.014, 보정 후 경계 0.086), motion n.s. | 그 방향이 학습 단계를 넘어 거의 유지되지 않음 |

검증 노트: 더 공정한 per-(layer,ckpt) null·cluster-robust permutation(20k)에서도
결론 유지(pooled null이 오히려 보수적). **batch_idx↔scene 정렬은 실증 검증됨**
(동일 batch_idx의 plan baseline_loss가 ckpt 간 Spearman 0.59~0.91, shift-1 대조군
−0.01; skip된 batch 0건 확인).

→ aux→plan 상호작용은 **순간적(state-dependent)** 관계. scene에 고정된 정적
가중치(static per-scene weighting)는 학습 시점 간 전이가 안 됨; 쓰려면
**online(상태 인지) weighting**이어야 함.

### A3. 기타 구조

- **det~motion의 plan 효과 상관** ρ=0.20 (p~1e-107): det이 plan을 돕는 scene에서
  motion도 돕는 경향 — det-motion 협력 구조가 plan 효과까지 연장됨. map은 독립.
- **|grad_dot| ~ plan 난이도** ρ=0.16~0.23 (모두 유의): 상호작용의 *크기*는
  plan-hard scene에 집중, 단 *부호*는 난이도로 예측 불가.
- **cosine 꼬리 질량 ≈ 0** (P(|cos|>0.3) < 0.06%): 숨은 destructive-conflict
  하위집단 없음 — 직교 결론 재확인.

## (B) Transfer-Gain 실험 (실행 인프라)

설계: ForkMerge TG = P(leave-one-out 1ep fine-tune) − P(tg_full 1ep fine-tune),
P = planning val **L2** (1/2/3s 평균), guardrail = obj_box_col. 4 variant:
tg_full / tg_no_det / tg_no_map / tg_no_motion + pretrain_ref(미세조정 전) drift 체크.

### 구현 (이번에 추가/수정된 것)

| 파일 | 내용 |
|---|---|
| `projects/mmdet3d_plugin/models/sparse_detector.py` | `ablate_tasks` 파라미터 + `_cut_ablated_grads` — 해당 task의 `<task>_loss_*`를 detach (gradient 차단, 로그 값 유지 → ablation 중 그 task의 퇴화도 관찰 가능) |
| `projects/configs/stage2_tg/_base_tg_1ep.py` | iter_31644 geometry(4GPU×bs4, 1758 it/ep), load_from(가중치만), lr 2e-5 cosine(min_lr_ratio 0.1, warmup 100), wandb 제거 |
| `projects/configs/stage2_tg/tg_{full,no_det,no_map,no_motion}.py` | variant 4종 |
| `tools/anal_tg.sh` | 드라이버: 학습→eval→집계, SMOKE 모드(별도 work_dir), hipad env PATH, pipefail |
| `tools/gradient_analysis/aggregate_tg.py` | eval 로그에서 L2/obj_col/obj_box_col/mAP/NDS 파싱 → `tg_summary.csv` + `tg_report.md` |

실행: `CUDA_VISIBLE_DEVICES=0,1,2,3 CKPT=<ckpt> CKPT_TAG=<tag> LR=<lr> bash tools/anal_tg.sh`
(4 GPU, 약 4–5h). 스모크: `SMOKE=1 bash tools/anal_tg.sh`.

### 설계 결정 (2026-06-13, 검증 워크플로 + 실험 점검 결과)

**(Q1) 출발 checkpoint = 수렴(18ep) ❌ → 중기(9ep) ✓.** 18ep 스모크에서 `drift(tg_full−pretrain_ref)=−0.0079` > `TG_L2(det)=+0.0026` — 수렴점은 포화라 신호가 미세조정 drift보다 작아 *구성상 null*. 분석 A도 aux↔plan 동역학이 초기/중기에 활성임을 보임. → **본런은 9ep(iter_15822) + LR=5e-5(9ep 자연 cosine LR)** 로 가소성 확보. 18ep은 "수렴 marginal" 참고용. 이상적 후속은 {3,9,18ep} sweep으로 A의 gradient 시간축에 인과 TG 시간지도 중첩.

**(Q2) det/motion 결합 → joint ablation 추가.** [sparse_onedecoder.py:986](../../../projects/mmdet3d_plugin/models/sparse_onedecoder.py)에서 `motion_query = motion_mode + (det_instance_feature + det_anchor_embed)` — motion이 det feature로 만들어지므로 `no_det`만 하면 motion loss가 det feature를 계속 supervise(과소평가). leave-one-out 단독은 *marginal-given-others*이므로 `tg_no_det_motion`(joint) 추가; `TG(det)+TG(motion)` vs `TG(det&motion)`로 중복/시너지 판별. 최종 5 variant.

**(eval) planning-focused.** `planning_eval()`은 자급자족(예측 궤적=모델 출력, GT·collision 박스=자체 dataset)이라 det/map/motion 메트릭과 무관. map mAP(Chamfer)가 ~40분 병목이라 `with_map=False, with_motion=False`, `with_det=True`(75초 ablation 새너티)+`with_planning=True`로 eval/variant ~40분→~5분.

### 검증 워크플로가 잡은 버그 5종 (모두 수정)

1. **DDP unused-param 크래시** — detach가 ablated task 전용 param의 autograd hook을 끊어 iter-2 크래시. → `find_unused_parameters=True`는 backbone gradient checkpointing(`with_cp`)과 "mark ready only once" 충돌. **최종: `L*0+L.detach()` zero-multiply** — param을 graph에 유지(grad 0), default DDP 경로로 동작. (smoke 검증 완료)
2. eval 산출물 충돌(pretrain_ref가 tg_full 덮어씀) → per-tag work_dir 격리.
3. 재학습 후 stale eval 캐시 재사용 → 학습 시 eval 로그 무효화.
4. distill loss key 누수(`det_kd_loss_*` 등) → prefix-leak assert 가드.
5. 통계 등급 정정 — det-1ep-OPPOSED만 SOLID(Bonferroni), map ALIGNED 기각, cross-layer 부호구조 rock-solid.

**파이프라인 end-to-end 검증 완료** (smoke: 학습+eval+집계 `tg_report.md` 정상 생성).

### 해석 규약

- `TG_L2(task) > 0` ⇒ task 제거로 plan L2 악화 ⇒ **task가 plan을 돕고 있었음**.
- 판정은 drift 스케일(tg_full − pretrain_ref) 대비로 읽음.
- 단일 seed 한계: 차이가 작으면 seed 2–3개 반복으로 noise floor 확인 필요
  (`SEED=1 bash tools/anal_tg.sh` 재실행, work_dir 분리).

## 부수 복구 (중요)

`HiP-AD/data`가 B2D 레이아웃으로 스왑되면서(6/12) pcgrad repo의 `data` 심볼릭이
깨져 **anal.sh 포함 모든 nuScenes 작업이 깨진 상태**였음. `HiP-AD-pcgrad/data`를
nuScenes(`/data/nuscenes`, `HiP-AD/data_nusc/*`) + B2D 파일링크의 superset 실제
디렉토리로 재구성해 양쪽 모두 동작하도록 복구. `ckpts` → `HiP-AD/ckpts` 링크 추가
(resnet50 pretrained).

## 다음 판단 기준

1. TG 결과가 나오면: A1의 시간가변 구조(1ep/3ep)와 대조 — 수렴 ckpt에서 TG≈0이면
   "수렴 후 aux는 plan에 중립"이 결론, TG>0이면 per-step 직교에도 불구하고
   적분 효과 존재(ForkMerge의 regularization 해석 지지).
2. A2의 결론(상태-의존)에 따라 per-scene 정적 가중치 방향은 기각, 후속은
   online weighting 또는 단계별(early-training) 개입으로 좁힘.
