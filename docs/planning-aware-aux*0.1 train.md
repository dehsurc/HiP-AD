# Planning-oriented 재가중(aux×0.1) 학습의 검출 능력 재배분 — 종합 보고서

작성: HiP-AD nuScenes val(6019) 분석 · baseline=`ckpts/rev_nusc/70+stage2.pth` vs reweighted=stage2에서 planning 제외 aux loss ×0.1 (`work_dirs/exp/E2_E1_stage2_18ep_planoriented_aux0p1/latest.pth`)

---

## 0. 요약 (TL;DR)

- 재가중 후 **전체 det mAP는 −0.007로 거의 불변**이지만, 내부적으로 검출 용량이 **planning-relevance 축을 따라 재배분**됨: 근거리(+1.3%)↔원거리(−7.4% rel), ego경로 위(+)↔경로 밖(−), 동적 agent(보존)↔정적/주변·희귀(희생).
- **단, 거리를 통제하면 "경로 위만 골라 남긴다"는 순수 relevance 신호는 약함** — 실효 선택축은 **자차거리/검출난이도**이고, 경로근접 효과는 대부분 거리에 매개됨.
- **인과 검증(실험 B, counterfactual Δplan, 충돌채널)**: reweighted가 버린 검출은 **100%가 plan에 영향 0**, plan을 좌우하는 검출(전부 자차 차로 내 근거리 차량)은 **100% 유지**. 방향성은 주장과 정확히 일치하나, 이 채널이 sparse해 표본이 얇음(n≈7).
- **결론**: "planning-aware 학습으로 planning에 중요한 검출만 남았다"는 **방향적으로 지지**되지만, 거리 교란과 채널 sparsity 때문에 **단독 확정은 불가**. 확정하려면 (i) neural inter_gnn 채널의 dense한 per-object 인과측정, (ii) planning-detach 통제 학습이 필요.

---

## 1. 실험 설정 & 검증

- reweighted = stage2(18ep)에서 **planning을 제외한 모든 aux loss(det/map/motion/depth) ×0.1**, planning ×1.0 → loss에서 planning 비중을 상대적으로 키운 "planning-aware" 재가중.
- 두 체크포인트를 동일 config·seed·test pipeline으로 재추론하여 예측(det box·map vector·plan·motion) 추출.
- **재현 검증**: 분석용 재구현 nuScenes AP가 공식 per-class AP와 **오차 ≤0.01**, stratum별 ΔmAP의 nGT 가중합이 공식 전체 ΔmAP(−0.007)와 일치. 재추론 reweighted 메트릭이 원본과 일치(det 0.3897≈0.3900, map 0.4762≈0.4765, L2 0.5533≈0.5538).
- 좌표: ego/lidar frame, 전방=+y. `results.pkl[i] ↔ timestamp-sorted infos[i] ↔ npz[i]`.

---

## 2. 전체 성능 (전후)

| 지표 | baseline | reweighted | Δ |
|---|---|---|---|
| det mAP | 0.3985 | 0.3900 | −0.0085 |
| map mAP | 0.5510 | 0.4765 | **−0.0745** |
| minADE (car·ped 평균) | 0.7025 | 0.6837 | −0.0188 (↓좋음) |
| plan L2 | 0.5972 | 0.5538 | **−0.0434 (↓좋음)** |
| box_col % | 0.067 | 0.073 | +0.006 (소폭 악화) |

→ planning(L2)·motion(minADE)은 향상, det은 소폭 하락, **map은 크게 희생**. 충돌은 미세 증가.

---

## 3. 검출(detection) 분석  (AP = 0.5/1/2/4m 평균)

### 3.1 거리 기반 — 근거리 보존/향상, 원거리 희생
| 거리대 | nGT | base | rewt | Δ(전체) | Δ(dynamic) | Δ(static) |
|---|---|---|---|---|---|---|
| 0–15m | 33,229 | 0.6150 | 0.6232 | **+0.0082** | +0.0015 | **+0.0183** |
| 15–30m | 53,811 | 0.3904 | 0.3777 | −0.0127 | −0.0066 | −0.0218 |
| 30–55m | 34,991 | 0.1408 | 0.1304 | −0.0104(−7.4% rel) | −0.0041 | **−0.0292** |

### 3.2 ego경로(planned path) 근접 기반 — 경로 위 보존/향상, 경로 밖 희생
| 경로까지 거리 | nGT | base | rewt | Δ(전체) | Δ(dynamic) |
|---|---|---|---|---|---|
| on (<2m) | 803 | 0.5410 | 0.5593 | **+0.0184** | **+0.0439** |
| near (2–5m) | 7,257 | 0.5840 | 0.5851 | +0.0011 | −0.0008 |
| mid (5–10m) | 18,572 | 0.5822 | 0.5708 | −0.0114 | −0.0066 |
| off (>10m) | 95,399 | 0.3466 | 0.3384 | −0.0082 | −0.0005 |

### 3.3 ⚠️ 거리 통제(교란 제거) — 핵심 한계
경로효과가 순수 relevance인지 거리 탓인지 보려고 **같은 거리대 안에서** 비교.

AP(15–30m): onpath −0.0163 vs offpath −0.0123 (거의 동일)

recall(score>0.3, moving-ego 4,713 scene):
| ring | Δrecall(in-corridor) | Δrecall(out) | gap |
|---|---|---|---|
| 0–15m | −0.014 | −0.010 | −0.004 |
| 15–30m | −0.024 | −0.043 | **+0.019 (약하게 지지)** |
| 30–55m | −0.250 (n=12, 노이즈) | −0.052 | — |

신규 drop 객체의 경로상 비중: 검출 1.2% → drop 0.3% (drop이 경로 4배 회피, 방향 일치하나 절대량 작음).
→ **거리를 통제하면 경로효과 약함. 지배축은 거리.**

### 3.4 클래스 기반 — 상호작용 agent 보존, 주변/희귀 희생
| 보존/향상 | car +0.000, ped +0.007, bike +0.035, truck +0.015, moto −0.003 |
|---|---|
| 희생 | barrier −0.009, cv −0.026, trailer −0.036, **bus −0.058** |

**mean Δ dynamic = −0.0006(보존)** vs **static = −0.0178(희생)**. 흔한 동적 agent는 지키고, 주변 정적·희귀 대형차는 버림.

---

## 4. 맵(map) 분석 — 정밀도의 광범위한 희생
- recall@0.5m(엄격): ped_crossing −0.112, divider −0.061, boundary −0.086 → **map mAP 하락의 주원인**
- recall@1.5m(느슨)은 소폭 상승, 평균 Chamfer는 전 거리대 +0.08~0.12m 악화 → **요소 존재는 잡되 정밀 위치가 나빠짐**
- 거리 그라디언트(recall@1m): 0–15m +0.022, 15–30m −0.026, 30m+ −0.024 (검출보다 약함)
→ 맵은 planning head와 결합이 약해 "통째 양보"에 가까움.

---

## 5. 인과 분석 — 실험 B (counterfactual Δplan)

**방법**: HiP-AD single-decoder의 planning은 두 경로로 검출에 의존 — (a) **충돌-rescore 채널**(plan_decoder.decode가 각 agent의 motion box와 ego plan 모드를 충돌검사해 충돌 모드를 −999로 억제), (b) **neural inter_gnn 채널**(plan 쿼리가 det feature에 cross-attend). 실험 B는 (a)를 측정: baseline planner에서 객체 k를 충돌검사에서 제거 → 최종 `plan_temp_2hz` 변화 Δplan. (1,500 scene·47,248 검출, plan 재현오차 0.002m.)

**결과**
1. plan은 개별 객체 제거에 극도로 robust: **47,248개 중 8개(0.02%)** 만 Δplan>0.1m.
2. 그 8개는 **전부 자차 차로 내(|x|<1.3m) 10–19m 고신뢰 car** (선행/후행 차량) → Δ가 진짜 planning-relevance를 잡음을 검증.
3. **주장 검증**: reweighted가 **버린 8,131개 검출 → 100% Δplan=0.0000**; **plan-critical 검출(n=7) → 100% 유지** (keep율 1.00 vs 전체 0.752, lift +0.25).

**한계**: (a) 충돌채널만 측정. 이 기준으론 kept·dropped 막론 99.98%가 "무관"이라 변별 신호는 꼬리(7개)뿐 — 방향성 완벽하나 표본 얇음. (b) neural 채널은 plan_output 모드에 이미 녹아 decode-ablation으로 못 잡음 → 더 dense한 의존성 미측정.

---

## 6. 종합 해석

재가중은 한정된 aux 용량을 **planning 관련성 축(거리 · 경로근접 · agent 상호작용성)을 따라 재배분**한다. 근거리/경로상/흔한 동적 agent는 보존·향상, 원거리/경로밖/주변·희귀는 희생. 인과적으로도 (충돌채널 한정) **버린 검출은 plan에 영향 0, 결정적 검출은 전부 유지**.

그러나 **"planning에 중요한 검출만 남았다"는 강한 주장은 아직 확정 불가**:
- 기하 분석에서 **거리 교란**을 통제하면 순수 relevance 신호가 약함(거리가 지배).
- 인과 분석(실험 B)은 방향성은 완벽하나 **충돌채널이 sparse(n=7)**, neural 채널 미측정.

---

## 7. 주장을 확정하기 위한 추가 실험 설계

주장을 3개 반증가능 명제로 분해: **C1**(남은 검출=planner가 의존) · **C2**(버린 검출=plan 무영향) · **C3**(원인이 planning 목적함수).

| 실험 | 내용 | 검증 | 비용 |
|---|---|---|---|
| **A. 객체별 planning 중요도** | A1 counterfactual Δplan(충돌+neural) / A2 planner attention·gradient saliency | relevance 축 정의(거리 독립) | 일부 done |
| **B. 반사실 keep vs drop** | 충돌채널: ✅done(방향 일치, sparse) → **B-neural**: inter_gnn에서 plan→det_k attention 마스킹+디코더 재실행 | C1·C2 | B-neural ~15–20분(subsample) |
| **C-ctrl. planning-detach 통제학습** | aux×0.1·planning×1 그대로 두되 **planning loss가 공유 det feature로 역류하는 gradient만 stop-grad**(V2). V1(현재) vs V2 비교 | C3 | 재학습 1회 |
| **D. gradient 분해** | det 쿼리 도달 gradient를 det-loss분 vs planning-head 역류분으로 분해 → planning-gradient가 고importance 객체에 집중·det 다운웨이트 상쇄 | 기전 | 재학습 X (1 backward/scene) |
| **E. 대안배제** | 거리·크기·가시성 통제 로지스틱 회귀(importance 계수 유의) + 비-planning 퇴화(에폭단축 등)는 비선택적임을 확인 | not-distance | 일부 재학습 |
| **F. 기능적** | TTC/경로기반 safety-critical 부분집합 recall 보존 + collision 유지 | 안전성 | 재학습 X |

### C-ctrl 정정 (논의에서 수정한 부분)
"aux×0.1 + **planning도 ×0.1**"은 통제로 **무의미**: 모든 loss를 똑같이 스케일하면 task 간 상대비중이 baseline과 동일하고, **Adam은 gradient 크기에 거의 불변**이라 사실상 baseline과 같아짐. 올바른 통제는 **상대비중(aux×0.1, plan×1)은 유지하되 planning→det gradient만 detach**(V2). V1만 importance-선택적 보존을 보이고 V2는 비선택적 퇴화면 → "원인은 planning gradient의 검출 protect" 확정.

### 실험 B 정확한 의미 (논의 정리)
두 full 모델 plan을 직접 비교하면 검출변화·planner변화가 교란됨 → **planner 하나(baseline)를 고정**하고 검출 입력만 빼서 그 검출의 인과기여만 분리. S_drop 제거 시 Δplan≈0이면 "버린 검출은 plan 무관", S_keep(또는 중요객체) 제거 시 Δplan 큼이면 "남긴 게 중요" → 같은 모델 안에서 인과 증명.

---

## 8. 결론 / 다음 단계

- **방향적 결론**: 재가중은 planning-relevance를 따라 검출을 재배분하며(근/경로상/동적 보존, 원/경로밖/주변 희생), 인과적으로도 버린 검출=plan무영향·결정적 검출=전부 유지.
- **확정 조건**: (1) **B-neural** (inter_gnn 마스킹 counterfactual, dense) — 충돌채널 sparsity 해결, (2) **C-ctrl detach 학습** — 원인이 planning임을 증명, (3) **D gradient 분해** — 기전.
- 우선순위: B-neural + C-ctrl 두 개면 "planning-aware 학습으로 planning에 중요한 검출만 남았다"를 표본 충분·교란 통제·인과 포함 수준으로 방어 가능.

---

## 부록 — 산출물 / 재현
- 그림: `fig1_det_distance_path.png`, `fig2_perclass_deltaAP.png`, `fig3_map_distance.png`, `fig4_relevance_gradient.png`
- 수치: `results_det.json`, `results_map.json`, `relevance.out`, `expB_records_baseline.pkl`
- 코드: `nusc_ap.py`(검증), `analyze_det.py`, `analyze_map.py`, `analyze_relevance.py`, `expB_runner.py`, `expB_analyze.py`
- 정성 뷰어: `tools/viewer/server.py` (baseline vs reweighted, `python tools/viewer/server.py 8077`)
