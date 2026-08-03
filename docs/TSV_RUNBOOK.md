# TSV 분석 — 재시작 런북 (다른 세션용)

> 상세 배경/결과 해석은 [TSV_gradient_analysis_report.md](TSV_gradient_analysis_report.md).
> 이 문서는 "새 세션에서 바로 실행하기" 위한 운영 메모.

---

## 0. 현재 상태 (한 눈에)

- **코드/보고서**: 로컬 커밋 완료 `ff3a612` (브랜치 `nusc/pcgrad`) — **아직 push 안 됨**(이 환경에 GitHub 인증 없음). push하려면 인증 후 `git push origin nusc/pcgrad`.
- **완료 실험**: Exp0(가중치델타) / Exp1(inter_gnn task별 gradient SVD, 1·3·9·18ep) / Exp2(방향 일관성) / Exp3(backbone·neck locus) / Exp4(공유 vs 전용).
- **결과 CSV**: `gradient_analysis_results/tsv_gradient/…` (gitignore — 커밋 안 됨, 로컬에만).
- **미완료 후보**: §5 참고 (per-sample 분석, 제대로된 null, NTK 확장, 변이 병합).

---

## 1. 환경 (복붙용)

```bash
PY=/home/yongjae/miniconda3/envs/hipad/bin/python   # mmcv 1.7.1 / torch 1.13 — base python엔 mmcv 없음!
cd /home/yongjae/e2e/HiP-AD
```

- GPU: RTX 4090 ×4, 코드가 `cuda:0` 사용. 수집(forward+backward)만 GPU, SVD는 CPU.
- **gotcha ①** 일부 config가 `data_nusc/` 경로 참조 → 심볼릭 링크 필요(이미 있음, 없으면):
  ```bash
  ln -sfn data data_nusc
  ```
  (ckpt 동봉 config는 `data/` 직접 사용해서 사실 없어도 됨. grad_analysis sibling config는 필요.)
- **gotcha ②** 반드시 `$PY`(hipad env)로 실행.

---

## 2. 분석 대상 (고정값)

| 항목 | 값 |
|---|---|
| config | `ckpts/rev_nusc/E2_E1_stage2_18ep_from70ep.py` (ckpt 동봉 = 정답) |
| checkpoints | `70+stage2_1ep.pth` / `_3ep` / `_9ep` / **`_18ep`(최종)** |
| 제외 | `70ep_stage1.pth` (stage1, 무시) |
| 공유 W | inter_gnn 어텐션 6층 × {`in_proj_weight`(768×256→Q/K/V), `out_proj.weight`}; backbone/neck conv |
| 전용 head | det→`det_refine`, map→`map_refine`, motion→`motion_refine`, **plan→`plan_refine`**(=plan_cls_branch; ego_refine 아님!) |

- **gotcha ③** plan 손실은 `plan_refine`로 흐름. `ego_refine`(ego status)엔 grad 안 옴 → NaN.
- **gotcha ④** 통계 신뢰하려면 **n≥64**. (n=6에서 det→plan 부호가 뒤집힌 아티팩트 있었음.)
- 모든 epoch에서 seed 고정 → **같은 64개 scene** 사용(epoch 비교 apples-to-apples).

---

## 3. 실행 순서 (명령 그대로)

```bash
PY=/home/yongjae/miniconda3/envs/hipad/bin/python
cd /home/yongjae/e2e/HiP-AD

# Exp0 — 가중치 델타 SVD (GPU 불필요, 전제 검증)
$PY tools/gradient_analysis/tsv_weight_delta.py

# Exp1 — task별 gradient SVD (inter_gnn), epoch별  (~1.5분/ckpt, GPU)
for c in 1ep 3ep 9ep 18ep; do
  $PY tools/gradient_analysis/tsv_gradient.py --n 64 \
     --ckpt-path ckpts/rev_nusc/70+stage2_${c}.pth \
     --out gradient_analysis_results/tsv_gradient/s2_${c}
done

# Exp2 — 방향 일관성 (post-processing, mean_grads.pt 사용, GPU 불필요)
$PY tools/gradient_analysis/tsv_plan_consistency.py    # flat + U subspace
$PY tools/gradient_analysis/tsv_mode_consistency.py    # U·V·mode1 (TSV 본체)

# Exp3 — backbone/neck locus 스윕 (~2~3분/ckpt, GPU)
for c in 1ep 3ep 9ep 18ep; do
  $PY tools/gradient_analysis/tsv_gradient.py --target backbone_neck --n 64 \
     --ckpt-path ckpts/rev_nusc/70+stage2_${c}.pth \
     --out gradient_analysis_results/tsv_gradient/bbneck_${c}
done

# Exp4 — 공유 vs 전용 스펙트럼 (18ep, GPU)
$PY tools/gradient_analysis/tsv_shared_vs_private.py --n 64
```

`tsv_gradient.py` 옵션: `--target {inter_gnn(기본),backbone,neck,backbone_neck}`, `--n`, `--ckpt-path`, `--config`, `--out`.

---

## 4. 모듈 / 산출물

| 모듈 (tools/gradient_analysis/) | 역할 | 산출물 (gradient_analysis_results/tsv_gradient/) |
|---|---|---|
| `tsv_weight_delta.py` | 가중치 델타 SVD (전제) | `tsv_weight_delta/` |
| `tsv_gradient.py` | task별 gradient SVD (핵심) | `s2_*/`, `bbneck_*/` (+ `mean_grads.pt`) |
| `tsv_plan_consistency.py` | flat + U-subspace 일관성 | `plan_consistency/` |
| `tsv_mode_consistency.py` | U·V·mode1 TSV 일관성 | `mode_consistency/` |
| `tsv_shared_vs_private.py` | 공유 vs 전용 스펙트럼 | `shared_vs_private_18ep/` |

각 폴더 CSV: `gradient_spectrum.csv`(저랭크), `subspace_overlap.csv`(간섭), `plan_mode_contrib.csv`(모드 기여). `mean_grads.pt` = task별 평균 gradient 행렬(후처리 재사용 핵심 — Exp2가 이걸 읽음).

---

## 5. 완료 / 미완료

**완료**: Exp0~4 (위 §3 전부).

**미완료 / 다음 후보** (보고서 §5 참고):
1. **per-sample 분석** (요청 중) — 50샘플 개별. 시간 ~2분(평균 낼 때와 사실상 동일, 같은 forward/backward). 단 *샘플별 스펙트럼은 confounded*(단일 샘플 grad는 rank≤토큰수, plan은 거의 rank-1) → **샘플별 cross-task cosine 분포**가 의미 있는 것. 모듈 미작성(신규 필요).
2. **제대로 된 null**: 동일 shape 랜덤행렬 N개 평균의 스펙트럼 → "backbone 2.6×", "저랭크가 평균화 아티팩트 아님" 강화.
3. **NTK/functional TSV**: cross-task Jacobian에 TSV (`ntk_coupling.py` 확장). 숨은 간섭 찾으려면 여기.
4. **★ 변이 병합 실험**: baseline + planning-aware(aux×0.1)를 TSV-M으로 병합 → Pareto 회복. 학습 불필요, `tsv_weight_delta.py` 인프라 재사용. **최우선 후속.**

---

## 6. 문서 / 메모 포인터

- 보고서: `docs/TSV_gradient_analysis_report.md` (배경·방법·결과·지표사전·E2E 적용·재현)
- auto-memory: `hipad-tsv-gradient`(결과 전체), `hipad-ckpt-selection`(ckpt 규칙), `hipad-gradient-structure`(per-sample 구조, 기존)

## 7. git

```bash
# 커밋 ff3a612 로컬에 있음(브랜치 nusc/pcgrad). 인증 후:
git push origin nusc/pcgrad
# 무관한 기존 변경(tools/viewer/*, vis_gt_nusc.py, PDF, data_b2d/)은 커밋 안 함 — 손대지 말 것.
```
