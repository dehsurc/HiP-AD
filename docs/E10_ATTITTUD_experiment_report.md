# E10 ATTITTUD — Planning-aware per-sample gradient surgery 실험 보고서

작성 2026-07-09 · HiP-AD nuScenes · 상태: ATTITTUD run 학습 중(iter ~80.5k/84.4k), 대조군 완주, 중간 eval 확보

---

## 0. 요약 (TL;DR)

- **가설**: task 간 gradient 관계가 sample마다 다르므로 **batch=1**로 학습하고, ATTITTUD(Dery et al. 2021) 아이디어로 **planning gradient를 나침반 삼아 aux(det/map/motion/ego) gradient를 planning-aware하게 수정**하면 planning이 개선된다.
- **구현**: `ATTITTUDOptimizerHook` — PCGrad식 "1 forward → task별 retain_graph backward"로 공유부 task-gradient 추출 → 최근 plan gradient의 SVD subspace(k≈4–6) 기준 good/bad/neutral 분해 → **good×1, bad×(−1), neutral×1** 재조합. batch1의 발산·노이즈는 **grad accumulation K=8 + NaN 가드 + BN freeze**로 대응.
- **안정성 결과 (헤드라인)**: 동일 seed·동일 데이터에서 **수술 유무만** 차이. 대조군은 ego head 발산으로 nan-guard **21회 skip군집·ego_status 1e10 thrashing**, ATTITTUD는 **skip 0회·ego_status ~25로 수렴**. → 수술이 batch1 학습을 *가능하게* 만듦.
- **성능 결과 (중간, iter_56260=5ep 상당)**: ATTITTUD가 baseline 6ep를 **plan L2(0.7395<0.7483)·map mAP(0.5318>0.5162) 모두 근소 우위**(1 epoch 덜 학습하고도). 단 **map 개선(+3%)이 plan 개선(+1.2%)보다 큼** → 이득의 주원천이 plan-특이 정렬이 아니라 일반 feature 학습(neutral 채널)일 가능성.

---

## 1. 동기 & 가설

- 자율주행 E2E(HiP-AD)의 unified decoder는 det/map/motion/ego/plan을 공유 파라미터로 학습 → task 간 gradient 충돌.
- 선행 분석([[gradient_conflict_analysis]], TSV persample): planning과 aux는 **평균적으로 거의 직교(sign-상쇄)**, ego만 planning과 유의미 충돌.
- 가설: **(1)** 배치를 1로 낮춰 per-sample gradient 관계를 노출, **(2)** ATTITTUD로 planning-aware하게 aux gradient를 수정, **(3)** batch1 전환에 맞춘 lr·optimizer·안정화 재설계.
- 모티브 논문: Dery, Dauphin, Grangier, *Auxiliary Task Update Decomposition: The Good, The Bad and The Neutral*, ICLR 2021 (arXiv:2108.11346).

---

## 2. 실험 설정

### 2.1 데이터·시작점
- nuScenes trainval (28,130 train / 6,019 val). 시작 checkpoint `ckpts/rev_nusc/70+stage2_3ep.pth` 에서 **finetune** (stage2 3ep 지점).
- ATTITTUD 학습 = 3 epoch (max_iters 84,390 = 28,130×3). 총 stage2 노출 ≈ **6ep 상당**(3ep base + 3ep ATTITTUD).

### 2.2 batch1 전환 셋업
| 항목 | 값 | 이유 |
|---|---|---|
| batch_size × num_gpus | 1 × 1 | per-sample gradient 관계 보존(가설1) |
| grad accumulation | **K=8** | per-sample 수술 유지 + batch 평균완충 복원(유효배치 8) |
| lr | 7e-5 | 1e-4×√(8/16), 유효배치 8 sqrt 스케일 (finetune 보수적) |
| fp16 | **None (fp32)** | multi-backward 안전성 + 기존 fp16 NaN 이력 회피 |
| BN | **freeze (FreezeBNHook)** | batch1은 BN 통계 무의미(1 scene 6뷰) |
| img_backbone with_cp | False | activation ckpt는 multi-backward와 충돌 |
| warmup(lr) / warmup(surgery) | 1000 / 50 | 초기 노이즈 완화 / 조기 수술 활성 |

### 2.3 수술 설정 (config `attittud=dict(...)`)
- `primary_task="plan"` — **plan만 primary(나침반)**, det/map/motion/ego는 aux(수술 대상)
- `shared_layers=["inter_gnn","ffn","norm"]` — **decoder 공유부만**(42그룹·7.1M). backbone 제외로 subspace가 backbone에 지배되는 문제 회피
- `mode="svd_buffer"`, `buffer_size=16`, `svd_energy=0.90`, `k_max=8` — 최근 plan gradient 16개 Gram-PCA로 누적에너지 90% rank 자동결정(실측 k≈4–6)
- `alpha_good=1.0, alpha_bad=-1.0, alpha_neutral=1.0`

**⚠️ α 해석(중요)**: neutral(=plan과 직교, aux의 ~95–98%)을 α=1로 **그대로 유지**. 반전되는 건 plan-충돌 성분(~2%)뿐. 즉 현재 설정은 "planning-only 집중"이 아니라 **"aux 거의 전부 학습 + plan 충돌만 반전"의 온건한 버전**. (진짜 planning-집중은 α_neutral=0 → §7 후속 아이디어)

---

## 3. 코드 흐름 (PCGrad식 multi-backward)

한 sample = 한 micro-step. hook: `projects/mmdet3d_plugin/core/hooks/attittud_optimizer_hook.py`

1. **Forward 1번** → `SparseDetector.train_step`가 loss를 prefix로 묶어 `task_losses={det,map,motion,ego,plan}` + `aux_loss`(dense_depth)로 분리. 그래프는 하나.
2. **task별 backward (retain_graph=True)** — PCGrad와 동일. 각 backward 직전 `_zero_shared_grads()`로 **공유부 grad만 격리**해 `flat[task]` 벡터로 추출. task-private 파라미터는 zero 안 함 → 5회 backward 동안 자연 누적(= total loss private grad, 수술 안 함).
3. **수술(공유부만)**: ring buffer→SVD로 plan subspace U → 각 aux grad를 U 기준 good/bad/neutral 분해 → α 재조합 → `merged = g_plan + Σ aux_modified`.
4. **accumulation(K=8)**: `merged`를 `_accum_shared`에 누적, private는 `.grad` 누적. 8 sample 후 공유부 할당 → 전체 /8 평균 → clip(max_norm 25) → `optimizer.step()`.

비용: forward 1 + **backward 6회**(5 task + aux) → batch1 ~1.7s/iter (대조군 순수학습 ~0.6s).

---

## 4. 안정성 인프라 (batch1 필수)

- **FreezeBNHook**: 매 train iter 전 BN eval 고정.
- **SafeOptimizerHook / NaN 가드**(`nan_guard.py`): non-finite loss/grad 감지 시 스텝 skip + **모든 InstanceBank.reset()**. (ego temporal bank가 `cached_feature=instance_feature.detach()`로 값 재귀 → NaN이 캐시 오염 → step-skip만으론 못 지움, reset 필수)
- **근본 원인 규명**: batch1 NaN은 fp16이 아니라(attention weight 실측 fp32) **무경계 ego-status MLP + temporal 값 재귀**가 평균완충 없는 batch1에서 발산 → fp32에서도 inf. warmup 무효(lr≈4e-6에서도 발생).

---

## 5. 로깅 (세겹 로컬 저장)

- `attittud_persample.jsonl` — **매 sample** task별 grad_norm·cos_plan·frac_bad·mod_ratio·k·plan_grad_norm·ego_status·loss·timestamp
- `<ts>.log.json` — 집계 스칼라 44키: 충돌행렬(task쌍 cos 10개), subspace 스펙트럼(eff_rank·eig0_frac)·drift, plan_coherence, 수술강도, **task별 방향수렴성**(coherence=‖E[g]‖/E‖g‖, align_ema, self_coherence, dirmean)
- `<ts>.log` 사람용 + wandb(본실행 시)

방향수렴성 관측: **coherence≈0.12(노이즈플로어 근처)**, self_coherence≈0.15–0.25, ego self_coherence −0.51(연속스텝 방향 뒤집힘). → batch1 per-sample gradient는 강한 공통 수렴방향 없음(가설 "sample마다 관계 다르다" 지지).

---

## 6. 결과

### 6.1 안정성 (500iter smoke, 동일 seed)
| | nan-guard skip | ego_status | 학습 |
|---|---|---|---|
| ATTITTUD | **0** | ~25 (간헐 스파이크도 흡수) | 정상 |
| 대조군(수술 X) | **~1685 윈도우** | ~1e10 | thrashing·under-train |

ego bad성분 반전(cos_plan −0.35, α_bad=−1)이 ego 발산을 실제 감쇠 = planning-aware 수술의 **안정화 부수효과**.

### 6.2 성능 (val 6019, motion 제외 재평가, iter_56260=5ep 상당)
| checkpoint | 상당 | plan L2 ↓ | map mAP ↑ | obj_col ↓ |
|---|---|---|---|---|
| baseline 3ep(시작점) | 3ep | 1.0395 | 0.4987 | 0.0089 |
| baseline 6ep | 6ep | 0.7483 | 0.5162 | 0.0060 |
| **ATTITTUD(iter_56260)** | **5ep** | **0.7395** | **0.5318** | 0.0073 |

- ATTITTUD가 **5ep 상당인데 6ep baseline을 두 지표 모두 근소 우위**(1 epoch 덜 학습).
- 단 **plan 개선 −0.0088(~1.2%) < map 개선 +0.0156(~3%)** → 이득 원천이 plan-특이 정렬(2% aligned)보다 **일반 aux feature(neutral 95%)**일 가능성.

### 6.3 주의(확정 아님)
1. iter_56260은 **5ep 상당(2ep 수술)** — full 3ep 완주본이 진짜 6ep 등가 비교. 현재 preliminary.
2. 단일 seed·단일 eval, error bar 없음. plan 1.2% 차이는 노이즈 가능. map 3%가 견고.
3. baseline 6ep는 처음부터 6ep라 "3ep+3ep 일반학습" 완벽 등가 아님. motion minADE는 eval DB 경로 문제로 미측정.

---

## 7. 해석 (steelman / devil)

**Steelman**: 수술이 planning뿐 아니라 **학습 안정성**이라는 강한 부수효과를 냄(대조군 붕괴 대비). map/plan 모두 근소 우위. det/map은 plan과 거의 직교(neutral 유지되어 정상 학습)라 충돌 없이 개선.

**Devil**: **이득의 주원천이 neutral 채널(일반 feature 학습)**일 개연 — map 개선 > plan 개선이 이를 시사. plan-특이 정렬 성분은 2%뿐이라 "정렬이 planning의 causal lever"라는 강한 주장은 이 데이터로 지지 안 됨. 이전 NTK 결론([[gradient_conflict_results]] "결합=진단적, lever 아님, 9ep+3ep negative")과 정합.

**미해결 ablation**: 안정화가 (a)ego-plan 충돌제거 때문인지 (b)수술의 전체 grad 축소 때문인지 → **α_bad=0(drop) vs −1(flip)** 비교로 분리 필요.

---

## 8. 상태 & 다음 스텝

- ATTITTUD full 3ep 완주 대기(현재 ~95%, resume 이력 있음: iter_56260에서 재개, ckpt 간격 5000). 완주본으로 §6.2 재확인.
- **후속 아이디어**(별도 노트 `.omc/research/idea_planning_parasitic_aligned_aux.md`): "plan을 update 안 하고 aligned-aux(α_neutral=0)만으로 planning이 학습되는가" — 정렬 채널을 단독 격리한 인과 실험. go/no-go는 per-sample jsonl로 정렬에너지 정량화 먼저.
- 공정 planning 비교를 위한 대안: 동일 3ep 시작점 + **일반 batch16 3ep** baseline, 또는 조기 checkpoint(stage2-1ep)에서 finetune해 희석 완화.

---

## 부록 — 산출물
- config: `projects/configs/experiments/E10_attittud_b1_ft3ep.py`, 대조군 `E10_ctrl_plain_b1_ft3ep.py`
- hook: `core/hooks/attittud_optimizer_hook.py`, `nan_guard.py`, `freeze_bn_hook.py`
- checkpoint: `work_dirs/exp/E10_attittud_b1_ft3ep/iter_{28130,56260}.pth`
- eval: `eval_{attittud_5ep,base_6ep,base_3ep}.pkl` / `.log`, 재평가 `/tmp/reeval_*.log`
- 로깅: `work_dirs/exp/E10_attittud_b1_ft3ep/attittud_persample.jsonl`, `*.log.json`
