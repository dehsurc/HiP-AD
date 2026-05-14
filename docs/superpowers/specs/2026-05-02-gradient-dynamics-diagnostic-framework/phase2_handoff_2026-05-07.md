/home/yongjae/e2e/HiP-AD/docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_handoff_2026-05-07.md# Phase 2 세션 핸드오프 (2026-05-07)

> 이전 세션이 GPU 미가용으로 중단되었습니다. 새 세션은 이 문서를 먼저 읽고 진행하세요.

## 컨텍스트

HiP-AD (e2e 자율주행) multi-task gradient dynamics 진단 프레임워크 **Phase 2** 실행 중.
- 작업 방식: Subagent-Driven Development (`superpowers:subagent-driven-development`)
- 25-task plan을 task별로 implementer subagent에 dispatch

## 핵심 문서

- **Plan:** [docs/superpowers/plans/2026-05-02-gradient-dynamics-diagnostic-framework-phase2.md](../../plans/2026-05-02-gradient-dynamics-diagnostic-framework-phase2.md)
- **Spec §4 (Phase 2 설계):** [../2026-05-02-gradient-dynamics-diagnostic-framework-design.md](../2026-05-02-gradient-dynamics-diagnostic-framework-design.md)
- **Phase 1 gating:** [phase1_gating_decision.md](phase1_gating_decision.md) — branch **Pass B** (magnitude-dominated, conflict-weak)
- **VAD loss-key snapshot:** `/tmp/vad_loss_keys.txt` (T6 split_losses 입력)

## 진행 상태

| Task | 상태 | 비고 |
|---|---|---|
| T0 VAD env smoke (R1) | ✅ done | 28 loss keys 캡처 |
| T1 Protocol skeleton + TemporalSnapshot | ✅ done | 3 tests pass |
| T2 MockAdapter test scaffold | ✅ done | 5 tests pass |
| **T3 HiP-AD adapter trio** | ⚠️ blocked | dataset config 미스매치 |
| T4–T24 | pending | 21 task 남음 |

검증 명령:
```bash
cd /home/yongjae/e2e/HiP-AD
ls tools/gradient_analysis/adapters/{base.py,__init__.py,hipad.py}
ls tests/gradient_analysis/adapters/test_protocol.py
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/yongjae/miniconda3/envs/hipad/bin/python \
  -m pytest tests/gradient_analysis/adapters/test_protocol.py -v
# Expected: 5 passed
```

## T3 차단 원인 (먼저 해결)

Plan T3의 `HIPAD_CONFIG = ckpts/HiP-AD-Stage2_code.py` 가 `Bench2DriveDataset` 을 사용함. 그러나 `data/` 디렉토리는 nuscenes 데이터로만 셋업됨:

```
data/
├── infos -> /home/yongjae/e2e/HiP-AD/data_nusc/infos
├── nuscenes_infos_train.pkl -> ...
└── nuscenes_infos_val.pkl -> ...
(b2d_infos_train.pkl 없음 → FileNotFoundError)
```

T3 test 둘 다 `FileNotFoundError: data/infos/b2d_infos_train.pkl`로 실패. Phase 1 분석은 nuscenes 데이터로 진행했을 것인데 plan은 b2d config를 가리킨다.

### 해결 옵션 (사용자 결정 필요)

A. **nuscenes config로 변경** (권장) — Phase 1에서 실제 사용한 config 경로 확인:
   ```bash
   grep -i "config\|cfg" gradient_analysis_results_phase1/*.log 2>&1 | head -10
   ls projects/configs/experiments/*.py projects/configs/*nusc*.py 2>/dev/null
   ```
   가능 후보: `projects/configs/hipad_nusc_stage2_expB_smoothl1.py`, `projects/configs/experiments/E2_E1_stage2_18ep_1_3_seed0.py`

B. **b2d 데이터 셋업** — 데이터 다운로드/링크 필요, 큰 작업

권장: **A**. Plan T3 step 1의 test 파일에서 `HIPAD_CONFIG` 경로만 수정.

## Plan 편차 로그 (T0/T3에서 발견 — T4 이후에 반영 필요)

이 편차들은 plan 본문에 아직 반영되지 않았습니다. 다음 implementer dispatch에서 prompt에 명시하거나 plan을 수정하세요.

### VAD API 편차 (T4 이후 적용)

1. **`init_detector` 미존재** — VAD의 mmdet3d 번들에 `init_detector` 없음. `init_model`은 있지만 `train_cfg`를 zero out시켜 training-mode forward 깨짐.
   - **수정:** `mmdet3d.models.build_model(cfg.model) + mmcv.runner.load_checkpoint(model, ckpt)` 사용.
   - 영향: plan T4 step 3 `VadAdapter.build_model` 코드.

2. **`import projects` 부족** — VAD의 `projects/__init__.py`는 빈 파일. plugin 등록 안 됨.
   - **수정:** `import projects.mmdet3d_plugin` 사용.
   - 영향: plan T4 step 3 `_ensure_vad_paths` 함수.

3. **VAD-tiny decoder prefix** — 디코더 3 layer 중 마지막은 prefix 없이 `loss_*`로 보고됨 (`d0.*`, `d1.*`, 그리고 bare `loss_*`). 28 keys total.
   - **수정:** plan T6/T8/T20의 `range(3)` 가정은 OK이나, T6 `_VAD_TASK_PREFIXES` 매처가 prefix 없는 키도 처리하는지 검증 필요 (현재 코드는 OK).

### HiP-AD 편차

4. **ckpt 파일명** — plan은 `HiP-AD-Stage2_1ep.pth`로 적었지만 실제는 `code_epoch1.pth`. T3 test에서 정정됨.

### 환경

5. **pytest-cov 충돌** — pytest-cov 7.0 + Python 3.8 호환성 이슈. pytest 호출 시 항상 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 필요:
   ```bash
   PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $HIPAD_PY -m pytest ... -v
   ```

## 사용자 컨벤션 (필수 — implementer prompt에 매번 박을 것)

1. **한국어로 응답** (정확한 diacritics)
2. **No commit between tasks** — Phase 2 끝에 batch commit. 모든 implementer prompt에 "DO NOT commit" 명시.
3. **Plan 코드 verbatim** — implementer가 임의로 docstring/이름 바꾸지 못하게 prompt에 명시.
4. 큰 변경 전에는 의도 확인.

## 환경

- **HiP-AD env:** `HIPAD_PY=/home/yongjae/miniconda3/envs/hipad/bin/python`
- **VAD env:** `VAD_PY=/home/yongjae/miniconda3/envs/vad/bin/python`
- **HiP-AD repo:** `/home/yongjae/e2e/HiP-AD/`
- **VAD repo:** `/home/yongjae/e2e/VAD/` (read-only로 사용)
- **GPU:** 새 세션 시작 시 `nvidia-smi`로 free memory 확인. cuda:0/1/2/3 중 free한 것 선택.
- **현재 git branch:** HiP-AD `nusc/pcgrad` (main 아님 ✓), VAD `main` (read-only이므로 무관)

## GPU-의존 task vs CPU-가능 task

| GPU 필요 (`model.to('cuda')` 호출) | CPU만으로 가능 |
|---|---|
| T3, T4, T5, T6, T7, T8, T10, T12, T13, T19, T21, T22 | T9 (FrozenMatching code-move), T11 (ModelStateSnapshot code-move), T14 (audit), T15 (collector refactor 일부), T17 (file deletes), T18 (CLI flag), T20 (yaml authoring), T23 (post-hoc CSV diff), T24 (memo) |

**중요:** plan은 R5 인터리브 정책 (HiP-AD/VAD ops를 op pair마다 교차)을 따름. CPU-가능 task만 먼저 처리하면 R5 mitigation을 깨뜨림 — 권장 X. **GPU 확보 후 plan 순서대로 진행하는 게 정석.**

## 새 세션 시작 권장 순서

1. **현재 상태 확인** — 위 검증 명령으로 T0–T2 artifact 정상인지 확인
2. **GPU 확인** — `nvidia-smi`로 free memory ≥ 16 GB 있는 GPU 인덱스 확인
3. **T3 dataset config 결정** — 사용자에게 옵션 A vs B 묻기 (Phase 1 log + projects/configs/ 후보 보여주기)
4. **Subagent-Driven 재개** — 결정된 config로 T3 implementer 재dispatch
   - Subagent prompt template은 [implementer-prompt.md](/home/yongjae/.claude/plugins/cache/claude-plugins-official/superpowers/5.1.0/skills/subagent-driven-development/implementer-prompt.md) 참고
   - 이전 세션의 T3 dispatch에서 사용한 프롬프트는 plan T3 + 위 편차들 + "DO NOT commit" + `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 환경
5. **인터리브 진행** — T3 → T4 → T5 → T6 → T7 → T8 → T9 → T10 → T11 → T12 → T13 → T14 → T15 → T16 → T17 → T18 → T19 → T20 → T21 → T22 → T23 → T24

각 task마다:
- Implementer dispatch (mechanical: haiku, substantive: sonnet)
- Spec compliance + code quality 검증 (verbatim copy task는 직접 verify, substantive는 reviewer subagent)
- TodoWrite 업데이트
- 다음 task로

## 1.5 GPU-day task (T22) 처리

T22는 VAD primary run (4 ckpt × 100 batch ≈ 1.5일). 단일 세션에서 wait 불가.
- **방안:** T21 (smoke) 통과 후 T22를 `nohup ... &`로 background launch + stop. T23/T24는 T22 완료 후 다음 세션에서.

## 다음 세션 시작 프롬프트 (사용자가 새 세션에 paste할 텍스트)

```
HiP-AD multi-task gradient dynamics 진단 프레임워크 Phase 2를 이어서 진행합니다.

이전 세션 핸드오프 문서를 먼저 읽어주세요:
docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_handoff_2026-05-07.md

요약:
- T0/T1/T2 done, T3 dataset config 미스매치로 차단, T4-T24 pending
- 사용자 컨벤션: 한국어, no commit between tasks, plan 코드 verbatim
- 발견된 plan 편차 (VAD API, ckpt 파일명, pytest-cov)는 핸드오프 문서에 명시

다음 액션: T3 dataset config (nuscenes로 가야 함) 결정 후 Subagent-Driven으로 재개.
```
