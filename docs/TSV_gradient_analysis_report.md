# HiP-AD Task Singular Gradients (TSV) 분석 보고서

분석 대상: `ckpts/rev_nusc/70+stage2_{1,3,9,18}ep.pth` (config `ckpts/rev_nusc/E2_E1_stage2_18ep_from70ep.py`)
작성: 2026-06-29 / 환경: conda `hipad` (mmcv 1.7.1, torch 1.13) / GPU RTX 4090

---

## TL;DR (한 줄 요약 다ㅈ섯 개)

1. **HiP-AD의 task들은 공유 weight에서 서로 안 싸운다.** flat cosine만 0인 게 아니라, **특이모드(singular mode) 단위로 분해해도** 충돌이 없다 — "0인데 소수 모드에 숨어있나?"의 답은 **숨어있지 않다.**
2. **plan(계획)의 학습 신호는 거의 rank-2**다. 단 한두 방향으로만 공유 weight를 민다. (det/map/motion도 저랭크지만 plan이 제일 심함.)
3. plan의 그 저차원 **부분공간은 학습 내내 안정**적인데, 그 안에서 **부호(방향)는 epoch마다 뒤집힌다.** ("subspace는 고정, 방향은 비정착.")
4. task끼리 gradient를 공유하는 정도는 **backbone(공유 병목) > neck > decoder** 순. 단 그 공유는 **충돌이 아니라 협력/중립**.
5. plan의 저랭크는 **공유 병목 특유가 아니라 task 내재적**(planning이 ego 쿼리 ~1개로 들어감) — 전용 헤드에서도 거의 rank-1.

---

## 0. 배경: TSV가 뭐고 왜 했나

### 0.1 원 논문 (TSV: Task Singular Vectors, CVPR 2025)
여러 task에 각각 fine-tune한 모델들을 **하나로 합칠(merge)** 때 task끼리 간섭한다. 논문의 관찰: 각 task의 가중치 변화량 `ΔW = W_finetuned − W_pretrained`("task vector")를 layer별로 SVD하면 **저랭크(low-rank)** 다. 그래서 상위 특이성분만 골라 task끼리 직교화하면 간섭이 준다.

### 0.2 우리 상황에 맞게 변형한 것
HiP-AD는 task별 단독 fine-tune이 없다(det/map/motion/plan을 한 모델에서 동시 학습). 그래서 논문의 `ΔW`(누적 가중치 변화)를 그대로 쓸 수 없다. 대신 **task별 gradient를 SVD**한다 — 이름하여 *Task Singular **Gradients***:

> 공유 weight `W`에서 각 task 손실의 평균 gradient `Ḡ_task = ⟨∂L_task/∂W⟩`를 구해 `Ḡ = U S Vᵀ`로 분해.

### 0.3 핵심 질문
기존 HiP-AD gradient 분석에서 task 간 **flat cosine이 거의 0**(−0.05~+0.08)이었다. 그런데 flat cosine은 모든 방향을 한 덩어리로 평균낸다. **"몇 개 강한 모드에서는 충돌하는데 나머지가 희석하는" 구조를 못 본다.** SVD로 보면 *모드 단위*로 충돌을 분해할 수 있다. → **"간섭이 정말 없나, 아니면 소수 특이모드에만 숨어있나?"** 가 1번 질문.

---

## 1. 방법: 정확히 무엇을 어떻게 계산했나

### 1.1 데이터 소스 (제일 중요 — 헷갈리기 쉬움)
- 학습 샘플(batch_size=1) 하나를 모델에 forward → 각 task의 스칼라 손실 `L_task` (그 task의 loss 항들 합).
- `g = torch.autograd.grad(L_task, W)` = **손실을 공유 weight로 미분**한 값. weight와 **같은 모양의 행렬**.
- **고정된 64개 scene**에 대해 평균 → `Ḡ_task` (그 task의 "체계적 descent 방향"). 평균이 핵심: 단일 배치 gradient는 토큰 수에 rank가 묶여(`∂L/∂W = Σ_t δ_t xₜᵀ`) 활성값 통계만 보이므로, 평균내야 노이즈가 상쇄되고 진짜 학습 방향이 남는다.
- `Ḡ_task = U S Vᵀ` SVD.

### 1.2 `W` = 공유 weight 가 정확히 뭔가
HiP-AD는 디코더가 하나(`SparseOneDecoder`)이고 한 번의 forward에서 4개 task 쿼리를 **같이** 처리한다. 한 weight를 4개 task가 공유 → 4개 손실 모두 그 weight에 gradient를 만든다. **이렇게 공유해야 "간섭"이 정의된다.**

디코더 1블록 구조: `concat → gnn → inter_gnn → norm → split → deformable → concat → ffn → norm → split → refine`
- `inter_gnn` = det/map/motion/ego 쿼리가 서로를 보는 **교차-task graph attention**(=`nn.MultiheadAttention`). task 상호작용의 핵심 접점 → **여기를 주 분석 대상으로.**
- `refine` = task **전용** 예측 헤드(공유 아님).

| 분석 | `W`의 정체 | 개수 |
|---|---|---|
| **inter_gnn (주)** | 6개 디코더의 inter_gnn 어텐션: `in_proj_weight`(768×256→Q/K/V 256×256) + `out_proj.weight`(256×256) | 6층×4 = 24개 256×256 |
| backbone/neck | `img_backbone`(ResNet50) conv 53개 + `img_neck`(FPN) conv 8개. conv는 `(out, in·k·k)`로 펴서 행렬 | 61개 |
| 전용(대조) | 각 task의 `*_refine` 헤드 2D weight | task별 36~66개 |

### 1.3 체크포인트 & 환경
- 1ep=`70+stage2_1ep.pth`, 3ep=`_3ep`, 9ep=`_9ep`, **18ep=`70+stage2_18ep.pth`**(최종). stage1은 제외.
- 모든 epoch에서 **동일한 64개 scene**(seed 고정) 사용 → epoch 비교가 apples-to-apples.

---

## 2. 지표 사전 (쉽게)

`Ḡ = U S Vᵀ`. **U**=출력뉴런 방향, **V**=입력뉴런 방향, **S**=각 모드 세기. 모드 r = `s_r·u_r v_rᵀ`.

### 저랭크 정도 (한 시점·한 task)
| 지표 | 정의 | 직관 |
|---|---|---|
| `pr_norm` | (Σsᵢ)²/Σsᵢ² ÷ n | "활성 모드 비율". 1/n≈rank-1, 1.0=노이즈처럼 꽉 참. **낮을수록 저랭크** |
| `energy_top8` | 상위 8모드가 담은 에너지(Σs²) 비율 | 0.93 = 8개가 93% |
| `stable_rank` | ‖Ḡ‖²/s₁² | 1=순수 rank-1, 클수록 퍼짐 |

### task 간 간섭 (두 task, 같은 시점)
| 지표 | 정의 | 직관 |
|---|---|---|
| `flat_cos(plan,aux)` | 두 평균 gradient 통째 코사인 | 두 task가 같은 방향으로 당기나(+협력/−충돌/0무관) |
| `subspace_cos_k8` | plan top-8 부분공간 vs aux top-8 (principal-angle 평균) | 같은 *방향들*을 쓰나. 랜덤 바닥 √(8/n) |
| `mode_contrib` | `s_r·u_rᵀG_aux v_r` | plan의 r번째 모드를 aux가 +로 돕나/−로 방해하나 (크기 비정규화) |
| `neg_energy_ratio` | aux 투영 에너지 중 *반대 모드* 비율 | >0.5 순(純)반대 |
| `pos_frac` | 기여가 양수인 모드 비율 | >0.5 협력 |

### 방향 일관성 (같은 task, 두 epoch)
| 지표 | 정의 | 직관 |
|---|---|---|
| `flat_cos` | 두 epoch Ḡ 통째 코사인 (**TSV 아님**) | 부호 있는 전체 방향 유지? +1유지/0무관/−1역전 |
| `U_cos_k4`,`V_cos_k4` | top-4 좌/우특이부분공간 겹침 (**TSV 본체**, 부호불변) | 지배 출력/입력 방향이 같은 집합? 바닥 √(4/n)≈0.125 |
| `mode1_cos` | 최강 모드 `u₁v₁ᵀ` 일치 | 1등 모드 그대로? (특이값 비슷하면 순서 바뀌어 낮게 나옴) |

> **핵심 구분:** cross-epoch 지표(U/V/mode1/flat) = "같은 task가 시간에 따라 방향 유지하나(**일관성**)", cross-task 지표(subspace/contrib/neg_energy) = "다른 task끼리 방향 겹치나·싸우나(**간섭**)".

---

## 3. 실험별 결과

### Exp 0 — 전제 검증: TSV의 "저랭크" 가정이 이 모델에 성립하나?
가중치 변화량 `ΔW = W(stage2 최종) − W(stage2 시작)`을 SVD (GPU 불필요, 순수 가중치 연산). 이건 *joint*(4 task 합쳐진) 변화라 task별은 아니지만, "layer matrix가 저랭크냐"는 논문 전제를 싸게 검증.

- inter_gnn `ΔW`: `pr_norm` 0.44 (랜덤 0.72) → **부분적으로 저랭크**. 상위 32/256 모드가 에너지 71%. 첫 inter_gnn 층만 강한 저랭크(top-8=76%).
- 전체 이동량 `‖ΔW‖/‖W‖` ≈ 18%. 최종 업데이트 부분공간의 ~50%는 1ep에 이미 정해짐.
- **결론:** ViT만큼 극적이진 않지만 노이즈보단 분명히 저랭크 → gradient 분석으로 갈 가치 있음.

### Exp 1 — (핵심) inter_gnn에서 task별 gradient SVD
**1a. plan gradient는 거의 rank-2** (모든 epoch에서):

| epoch | pr_norm | energy_top8 | stable_rank |
|---|---|---|---|
| 1ep | 0.073 | 0.922 | 2.16 |
| 3ep | 0.089 | 0.895 | 2.36 |
| 9ep | 0.088 | 0.895 | 2.42 |
| 18ep | 0.064 | 0.934 | 1.93 |

상위 8모드(256중)가 90~98% 에너지. 4개 task 다 저랭크지만 plan이 제일 심함(18ep `pr_norm`: plan .064 < motion .120 < det .145 < map .175).

**1b. task 간 간섭 ≈ 0, 모드 단위로 분해해도 (모든 epoch 일관):**
- `flat_cos(plan,aux)` ≈ 0 (±0.02~0.05)
- plan top-8 모드 위 aux 기여: 부호 5:5 (`pos_frac` 0.46~0.59), `neg_energy_ratio` 일관된 편향 없음
- `subspace_cos_k8` 0.22~0.25 vs 랜덤 0.177 — 우연보다 살짝 위일 뿐

> **→ 1번 질문의 답: 간섭은 소수 모드에 숨어있지 않다. 진짜 없다.** flat cosine ≈ 0이 희석 때문이 아니라 실제 직교. "tasks mutually orthogonal, planning unique"를 특이모드 해상도에서 재확인.

### Exp 2 — plan 방향 일관성 (1/3/9/18ep, 같은 64 scene)
처음엔 `flat_cos`로만 봐서 "plan이 제일 비일관(−0.01)"이라 했는데, **그건 TSV 특이벡터가 아니라 raw 코사인**이었다. TSV 특이벡터(U·V 부분공간)로 다시 보면:

| task | flat_cos | U_cos_k4 | V_cos_k4 | mode1_cos |
|---|---|---|---|---|
| det | 0.139 | 0.457 | 0.558 | 0.256 |
| motion | 0.180 | 0.467 | 0.570 | 0.251 |
| **plan** | **−0.029** | 0.449 | 0.554 | 0.214 |
| map | 0.152 | **0.380** | 0.513 | **0.134** |

(랜덤 바닥 U/V_k4 ≈ 0.125)

- **정정:** TSV 특이부분공간 기준 plan은 det/motion과 **거의 동급으로 안정**. 제일 불안정한 건 오히려 **map**.
- task별 추세: **det·motion은 학습하며 방향이 수렴**(부호·부분공간 둘 다 상승, 특히 9→18ep). **map은 부분공간이 끝까지 안 정착**(방랑). **plan은 해리(dissociation)** — 부분공간(U 0.45)은 안정인데 부호 있는 방향(flat −0.03)만 수렴 안 함.
- **의미:** "planning **subspace**는 학습 내내 같은 저차원 축을 재사용, 그 안의 **부호 방향**만 epoch마다 뒤집힘." planning의 WTA/멀티모달(ego_fut argmin)·ego-status 비정상성과 정합.

### Exp 3 — 위치(locus) 스윕: backbone/neck vs decoder
task gradient가 **어디서** 가장 많이 겹치나? (top-8 부분공간 겹침 ÷ 랜덤 바닥)

| 위치 | subspace_cos_k8 | 랜덤 바닥 | 배율 |
|---|---|---|---|
| **backbone** (ResNet) | 0.31~0.36 | 0.125 | **~2.6×** |
| neck (FPN) | 0.21~0.33 | 0.177 | ~1.5× |
| decoder (inter_gnn) | 0.23 | 0.177 | ~1.3× |

- **공유 구조는 병목(backbone)에 집중**, neck→decoder로 갈수록 감소. 표현이 강제로 공유되는 곳에서 task gradient가 가장 많이 같은 저차원을 쓴다 (가설대로). task 분리용으로 설계된 decoder 헤드가 제일 안 겹침.
- **하지만 충돌 아님 — 협력/중립.** `flat_cos(plan,aux)`는 backbone에서도 ≈0 또는 *약간 양수*(det +0.10~+0.15 @1/3/9ep), det의 plan-모드 투영은 다수 양수(`pos_frac` 0.72~0.83 = plan을 *도움*).
- ⚠️ **정정 사례:** n=6 프리뷰에서 "det가 plan을 반대로(−0.19)"가 나왔지만 **n=64에서 사라짐** (det gradient 크기가 커서 소표본에서 부호가 튄 아티팩트). → 항상 n≥64.

### Exp 4 — 공유 vs 전용(refine 헤드) 스펙트럼 (18ep, n=64)
같은 task의 gradient를 *공유* inter_gnn과 *전용* 헤드에서 비교 (방향은 공간이 달라 비교 불가, 스펙트럼·크기만):

| task | scope | pr_norm | energy_top8 | stable_rank | ‖grad‖ |
|---|---|---|---|---|---|
| det | 공유 | 0.173 | 0.781 | 2.50 | 0.504 |
| det | 전용 | 0.205 | 0.885 | 1.99 | 0.498 |
| map | 공유 | 0.243 | 0.649 | 4.11 | 0.089 |
| map | 전용 | 0.240 | 0.781 | 3.27 | 0.129 |
| motion | 공유 | 0.126 | 0.844 | 2.54 | 0.256 |
| motion | 전용 | 0.251 | 0.921 | 1.95 | 0.064 |
| plan | 공유 | 0.113 | 0.859 | 2.51 | 0.270 |
| **plan** | **전용** | 0.158 | **0.991** | **1.32** | 0.520 |

**전용/공유 ‖grad‖ 비율:** det 0.99, map 1.45, plan 1.93, motion 0.25

- **(a) plan 저랭크는 기계적/내재적.** plan은 *전용* 헤드에서도 **거의 rank-1**(stable_rank 1.32, top-8=99%). 저랭크가 위치가 아니라 **task를 따라다님** → planning이 ego 쿼리 ~1개로 들어가는 데서 옴. (모든 task에서 전용이 공유보다 더 저랭크.)
- **(b) "안 싸운다"의 진짜 의미.** **det**(제일 큰 task)는 공유를 전용만큼 세게 씀(비율 0.99) → no-interference가 회피가 아니라 **진짜 협력적 공존**. plan/map은 주로 전용에서 일하고 공유는 가볍게 건드림. motion은 거꾸로 공유를 더 세게 씀(det feature/backbone 경유라는 기존 문서와 정합).

---

## 4. 종합 결론

1. **파괴적 간섭 없음 (강건).** flat cosine, 특이모드 분해, 부분공간 겹침, 4개 epoch, backbone~decoder 모든 위치에서 일관되게 task gradient가 직교~협력. 충돌은 어떤 해상도로 봐도 안 나타남.
2. **공유 구조의 지도는 있다.** 겹침 정도는 backbone(2.6×) > neck(1.5×) > decoder(1.3×). 단 그 공유는 협력적이며, 특히 det는 공유 capacity를 실제로 많이 쓰면서도 plan과 공존.
3. **plan은 특이한 task.** gradient가 거의 rank-2이고(내재적), 그 저차원 부분공간은 안정적이지만 부호 방향은 비정착. det/motion이 학습하며 수렴하는 것과 대조.
4. **TSV-merge 메커니즘(겹치는 특이부분공간 직교화)은 여기서 고칠 게 거의 없다** — 간섭이 이미 ~0이라서. 즉 TSV의 가치는 *간섭 감소*가 아니라 **각 task가 무엇을 저차원으로 학습하는지의 규명**.

---

## 5. TSV 논문을 E2E 모델에 어떻게 적용할까

### 5.1 TSV가 제공하는 도구 3개
- **TSV-C(압축):** task vector를 상위 특이성분만 남겨 ~10% 크기로 (저랭크라 가능).
- **TSV-M(병합):** 여러 모델을 layer별 특이부분공간 **직교화**로 간섭 줄여 하나로 합침.
- **저랭크 통찰:** task별 적응(가중치 변화)은 저랭크 → 저랭크로 표현·조작·전송.

### 5.2 우리 결과가 정해주는 "되는 것 / 안 되는 것"
- ❌ **간섭 감소(TSV-M의 본래 목적)는 inter_gnn에선 고칠 게 거의 없다** — 간섭이 이미 ~0.
- ✅ **저랭크성은 강하다**(plan rank~2, 전용 헤드는 거의 rank-1) → **저랭크 표현/압축/어댑터/병합은 매우 유효.**

### 5.3 구체 제안 (가치 × 실현성 순)

**제안 1 ★ 최우선 — 변이 모델 병합: baseline + planning-aware(aux×0.1)를 TSV-M으로**
- *배경:* aux×0.1 reweighting은 planning은 좋아지나 far/peripheral det+map을 희생한다(기존 결과). 두 모델을 단순 평균하면 양쪽 다 어중간해진다.
- *방법:* 두 모델의 `ΔW`(공통 init 대비)를 layer별 SVD → 상위 특이부분공간을 **직교화 후 병합**(TSV-M). planning 이득이 실린 특이방향과 perception 보존 방향이 다른 모드면, 직교 병합으로 **둘 다 유지** → Pareto front 회복.
- *근거:* task gradient가 직교적·저랭크 → 변이 간 task vector도 저랭크일 가능성↑ → SVD 병합이 잘 먹힘. `tsv_weight_delta.py` 인프라 그대로 재사용.
- *비용:* **중(학습 불필요, 두 체크포인트의 가중치 연산만).** 가장 먼저 시도할 가치.

**제안 2 — 저랭크 task 어댑터 (rank budgeting)**
- 공유 inter_gnn/backbone에 task별 LoRA식 **저랭크 보정**을 붙이되 rank를 *측정한 유효랭크*로(plan≈2, det/map은 더 크게). 적은 파라미터로 task별 미세조정, 간섭 없이 capacity 분리.
- *근거:* 측정된 `pr_norm`/`stable_rank`가 **원리적 rank 예산**을 직접 제공.
- *비용:* 중~고(학습 필요하나 풀 파인튜닝보다 훨씬 쌈).

**제안 3 — 불필요한 gradient surgery 제거 (진단 도구로서의 TSV)**
- repo의 PCGrad는 inter_gnn 간섭을 가정한다. 우리 결과: 모드 분해해도 충돌 ~0 → **PCGrad 불필요**. backbone도 협력적. → gradient surgery를 빼서 학습 단순화·가속.
- *비용:* 거의 0(제거). 즉시 검증 가능(있/없 A/B).

**제안 4 — 저랭크 체크포인트 압축 / fleet OTA 패치**
- stage-2 `ΔW`가 상위 모드에 에너지 집중(Exp 0) → **저랭크로 저장**. 차량 fleet의 OTA 모델 업데이트 시 diff 크기↓·적용 빠름.
- *비용:* 저.

**제안 5 — 학습 건전성 모니터 (plan subspace 추적)**
- plan의 rank-2 subspace는 안정인데 *부호 방향은 흔들린다*(Exp 2). 이 subspace 드리프트·부호 안정성을 학습 중 추적하면 **planning이 정착 중인지/방랑 중인지 조기 신호.** `mean_grads`만 주기적으로 저장하면 됨.
- *비용:* 저.

**제안 6 (천장 높음) — NTK/functional TSV로 확장**
- gradient-방향 직교성 ≠ 출력 간섭. **cross-task Jacobian(NTK)** 에 TSV를 적용하면 "성능에 실제 영향 주는" 간섭이 있는지 검증 가능. `ntk_coupling.py` 인프라 존재.
- *비용:* 고. 정말 숨은 간섭을 찾는다면 여기일 확률이 가장 높음.

### 5.4 한 줄 권고
**제안 1(planning-aware + baseline의 TSV-merge)** 이 기존 연구 라인과 직결되고 학습도 불필요해 가장 먼저 해볼 만하다. 그 다음 **제안 3(PCGrad 제거 A/B)** 이 싸고 결론이 명확하다. "간섭 줄이기"보다 **"저랭크 구조를 활용한 병합·압축·rank 예산"** 이 이 모델에서 TSV의 실질적 쓸모다.

---

## 6. attribution(sensitivity)과의 차이 — 헷갈리지 말 것

| | 기존 attribution | 이번 TSV-gradient |
|---|---|---|
| 분자 | plan loss | 각 task loss |
| **분모(미분 대상)** | **h = 객체 feature/활성값** | **W = 공유 weight** |
| 남기는 것 | **노름(크기)** 1개/객체 | 행렬 전체(**방향** U·S·V) |
| 공간 | feature 공간, 객체별 **벡터** | 파라미터 공간, weight 모양 **행렬** |
| 질문 | planning이 무엇을 **읽나** | task가 weight를 어떻게 **바꾸나** |

> 그래서 **"plan이 map feature를 74% 읽음"(attribution)** 과 **"plan의 weight-gradient가 map과 직교"(TSV)** 는 동시에 참, 모순 아님. *읽는 것* ≠ *그 feature를 만드는 task와 같은 방향으로 weight를 미는 것*.

---

## 7. 한계 / 주의

- **부분공간 겹침의 null이 휴리스틱**(√(k/n))이다. 동일 shape 랜덤행렬 N개 평균의 스펙트럼으로 제대로 된 null을 붙이면 "backbone 2.6×"·"저랭크가 평균화 아티팩트 아님"이 더 단단해진다.
- **표본 수 민감.** n=6에서 부호가 뒤집힌 사례가 있었다. **항상 n≥64.**
- 평균 gradient는 *체계적* 방향만 본다 (per-sample 고차원 구조는 별도 — 기존 메모 `eff_rank≈279` 참고). 둘은 모순 아님: per-sample은 흩어지고 평균이 ~rank-2로 collapse.
- gradient-방향 직교성 ≠ functional(NTK) 무간섭. 성능에 영향 주는 간섭이 있다면 cross-task Jacobian(NTK)에 있을 수 있음(미탐색, 후속 후보).
- `data_nusc → data` 심볼릭 링크를 repo에 생성함(일부 config가 `data_nusc/` 경로 참조). 정식 config는 `data/` 직접 사용.

---

## 8. 재현 방법

환경: `/home/yongjae/miniconda3/envs/hipad/bin/python`, repo `/home/yongjae/e2e/HiP-AD`

```bash
HP=/home/yongjae/miniconda3/envs/hipad/bin/python
cd /home/yongjae/e2e/HiP-AD

# Exp 0: 가중치 델타 SVD (GPU 불필요)
$HP tools/gradient_analysis/tsv_weight_delta.py

# Exp 1: task별 gradient SVD (inter_gnn), 각 epoch
for c in 1ep 3ep 9ep 18ep; do
  $HP tools/gradient_analysis/tsv_gradient.py --n 64 \
     --ckpt-path ckpts/rev_nusc/70+stage2_${c}.pth \
     --out gradient_analysis_results/tsv_gradient/s2_${c}
done

# Exp 2: 방향 일관성 (post-processing, mean_grads.pt 사용)
$HP tools/gradient_analysis/tsv_plan_consistency.py   # flat + U subspace
$HP tools/gradient_analysis/tsv_mode_consistency.py   # U·V·mode1 (TSV 본체)

# Exp 3: backbone/neck locus 스윕
for c in 1ep 3ep 9ep 18ep; do
  $HP tools/gradient_analysis/tsv_gradient.py --target backbone_neck --n 64 \
     --ckpt-path ckpts/rev_nusc/70+stage2_${c}.pth \
     --out gradient_analysis_results/tsv_gradient/bbneck_${c}
done

# Exp 4: 공유 vs 전용
$HP tools/gradient_analysis/tsv_shared_vs_private.py --n 64
```

| 모듈 | 역할 | 산출물 |
|---|---|---|
| `tsv_weight_delta.py` | Exp0 가중치 델타 SVD | `tsv_weight_delta/` |
| `tsv_gradient.py` | Exp1/3 task별 gradient SVD (`--target inter_gnn/backbone/neck/backbone_neck`) | `tsv_gradient/{s2_,bbneck_}*` |
| `tsv_plan_consistency.py` | Exp2 flat + U-subspace 일관성 | `plan_consistency/` |
| `tsv_mode_consistency.py` | Exp2 U·V·mode1 TSV 일관성 | `mode_consistency/` |
| `tsv_shared_vs_private.py` | Exp4 공유 vs 전용 스펙트럼 | `shared_vs_private_18ep/` |

각 산출 디렉터리에 `gradient_spectrum.csv`(저랭크), `subspace_overlap.csv`(간섭), `plan_mode_contrib.csv`(모드 기여) 등 CSV 포함. 중간 산물 `mean_grads.pt`(task별 평균 gradient 행렬)는 후처리 재사용용.
