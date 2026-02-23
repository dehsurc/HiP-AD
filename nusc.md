# HiP-AD nuScenes 구성 정리

## 1) 목표

HiP-AD에서 nuScenes 학습/평가를 B2D 2-stage 흐름과 최대한 유사한 구조로 운영한다.

- Stage1: `det/map/plan/ego` 학습, `motion` 비활성
- Stage2: `det/map/plan/ego/motion` 학습
- Ego supervision: B2D와 동일하게 status 기반(`with_supervise_ego_status=True`)

---

## 2) 현재 코드 구조

### 핵심 config

- `projects/configs/hipad_nusc_stage1.py`
- `projects/configs/hipad_nusc_stage2.py`

### stage별 ego 관련 동작

- Stage1
- `task_select = ["det", "map", "plan", "ego"]`
- `with_supervise_ego_status=True`
- `loss_ego_status` 가중치 `0.0` (B2D stage1 스타일)

- Stage2
- `task_select = ["det", "map", "plan", "ego", "motion"]`
- `with_supervise_ego_status=True`
- `loss_ego_status` 가중치 `1.0` (B2D stage2 스타일)

### ego status 차원

- 현재 nuScenes config 기본값: `ego_status_dims = 6`
- 내부적으로 nuScenes 원본 canbus(10차원)에서 B2D 스타일 6차원으로 매핑
- 매핑 형태: `[vx, ax, ay, wx, wy, steer]`

관련 구현:
- `projects/mmdet3d_plugin/datasets/nuscenes_3d_dataset.py`
- `projects/mmdet3d_plugin/datasets/pipelines/transform.py`

---

## 3) 데이터 준비

`HiP-AD/data`를 심볼릭 링크로 운영 중이면, 아래 생성물은 링크 타깃 경로에 저장된다.

- infos: `data/infos/*.pkl`
- anchors: `data/kmeans/*.npy`

예시 경로 변수:

- `PATH_TO_HIPAD`: HiP-AD 리포 루트
- `PATH_TO_NUSCENES`: nuScenes 데이터 루트

---

## 4) nuScenes infos 변환

작업 디렉토리:

```bash
cd PATH_TO_HIPAD
```

trainval 생성:

```bash
PYTHONPATH="$(pwd)" python tools/data_converter/nuscenes_converter.py nuscenes \
  --root-path PATH_TO_NUSCENES \
  --canbus PATH_TO_NUSCENES \
  --out-dir data/infos \
  --extra-tag nuscenes \
  --version v1.0
```

mini 생성(스모크 테스트용):

```bash
PYTHONPATH="$(pwd)" python tools/data_converter/nuscenes_converter.py nuscenes \
  --root-path PATH_TO_NUSCENES \
  --canbus PATH_TO_NUSCENES \
  --out-dir data/infos \
  --extra-tag nuscenes \
  --version v1.0-mini
```

참고:
- 이 스크립트는 `v1.0` 입력 시 내부적으로 train/val/test 처리 흐름을 탄다.
- `--version v1.0-trainval` 대신 `--version v1.0` 사용.

---

## 5) kmeans anchor 생성

```bash
cd PATH_TO_HIPAD
PYTHONPATH="$(pwd)" python tools/kmeans_nuscenes/kmeans_det.py
PYTHONPATH="$(pwd)" python tools/kmeans_nuscenes/kmeans_map.py
PYTHONPATH="$(pwd)" python tools/kmeans_nuscenes/kmeans_motion.py
PYTHONPATH="$(pwd)" python tools/kmeans_nuscenes/kmeans_plan.py
```

생성 확인:

```bash
ls -l data/kmeans/kmeans_det_900.npy
ls -l data/kmeans/kmeans_map_100.npy
ls -l data/kmeans/kmeans_motion_6.npy
ls -l data/kmeans/kmeans_plan_6.npy
```

시각화 결과(`vis/kmeans/*`)는 학습 필수 파일은 아님.

---

## 6) 학습 실행

### Stage1

```bash
cd PATH_TO_HIPAD
PYTHONPATH="$(pwd)" python tools/train.py \
  projects/configs/hipad_nusc_stage1.py \
  --work-dir work_dirs/hipad_nusc_stage1 \
  --gpus 1
```

### Stage2

```bash
cd PATH_TO_HIPAD
PYTHONPATH="$(pwd)" python tools/train.py \
  projects/configs/hipad_nusc_stage2.py \
  --work-dir work_dirs/hipad_nusc_stage2 \
  --gpus 1
```

기본적으로 Stage2는 아래 체크포인트를 로드한다.

- `./work_dirs/hipad_nusc_stage1/latest.pth`

### Trainval 실제 학습 명령 (Single GPU)

아래 명령은 mini가 아니라 trainval infos를 명시적으로 사용한다.

#### Stage1 (trainval)

```bash
cd PATH_TO_HIPAD
PYTHONPATH="$(pwd)" python tools/train.py \
  projects/configs/hipad_nusc_stage1.py \
  --work-dir work_dirs/hipad_nusc_stage1 \
  --gpus 1 \
  --cfg-options \
    data.train.ann_file=data/infos/nuscenes_infos_train.pkl \
    data.val.ann_file=data/infos/nuscenes_infos_val.pkl \
    data.test.ann_file=data/infos/nuscenes_infos_val.pkl \
    data.train.version=v1.0-trainval \
    data.val.version=v1.0-trainval \
    data.test.version=v1.0-trainval \
    data.train.data_root=PATH_TO_NUSCENES/ \
    data.val.data_root=PATH_TO_NUSCENES/ \
    data.test.data_root=PATH_TO_NUSCENES/ \
    eval_config.ann_file=data/infos/nuscenes_infos_val.pkl \
    eval_config.version=v1.0-trainval
```

#### Stage2 (trainval)

```bash
cd PATH_TO_HIPAD
PYTHONPATH="$(pwd)" python tools/train.py \
  projects/configs/hipad_nusc_stage2.py \
  --work-dir work_dirs/hipad_nusc_stage2 \
  --gpus 1 \
  --cfg-options \
    load_from=work_dirs/hipad_nusc_stage1/latest.pth \
    data.train.ann_file=data/infos/nuscenes_infos_train.pkl \
    data.val.ann_file=data/infos/nuscenes_infos_val.pkl \
    data.test.ann_file=data/infos/nuscenes_infos_val.pkl \
    data.train.version=v1.0-trainval \
    data.val.version=v1.0-trainval \
    data.test.version=v1.0-trainval \
    data.train.data_root=PATH_TO_NUSCENES/ \
    data.val.data_root=PATH_TO_NUSCENES/ \
    data.test.data_root=PATH_TO_NUSCENES/ \
    eval_config.ann_file=data/infos/nuscenes_infos_val.pkl \
    eval_config.version=v1.0-trainval
```

### Trainval 실제 학습 명령 (Multi GPU)

#### Stage1 (trainval, 8 GPU 예시)

```bash
cd PATH_TO_HIPAD
bash tools/dist_train.sh \
  projects/configs/hipad_nusc_stage1.py 8 \
  --work-dir work_dirs/hipad_nusc_stage1 \
  --cfg-options \
    data.train.ann_file=data/infos/nuscenes_infos_train.pkl \
    data.val.ann_file=data/infos/nuscenes_infos_val.pkl \
    data.test.ann_file=data/infos/nuscenes_infos_val.pkl \
    data.train.version=v1.0-trainval \
    data.val.version=v1.0-trainval \
    data.test.version=v1.0-trainval \
    data.train.data_root=PATH_TO_NUSCENES/ \
    data.val.data_root=PATH_TO_NUSCENES/ \
    data.test.data_root=PATH_TO_NUSCENES/ \
    eval_config.ann_file=data/infos/nuscenes_infos_val.pkl \
    eval_config.version=v1.0-trainval
```

#### Stage2 (trainval, 8 GPU 예시)

```bash
cd PATH_TO_HIPAD
bash tools/dist_train.sh \
  projects/configs/hipad_nusc_stage2.py 8 \
  --work-dir work_dirs/hipad_nusc_stage2 \
  --cfg-options \
    load_from=work_dirs/hipad_nusc_stage1/latest.pth \
    data.train.ann_file=data/infos/nuscenes_infos_train.pkl \
    data.val.ann_file=data/infos/nuscenes_infos_val.pkl \
    data.test.ann_file=data/infos/nuscenes_infos_val.pkl \
    data.train.version=v1.0-trainval \
    data.val.version=v1.0-trainval \
    data.test.version=v1.0-trainval \
    data.train.data_root=PATH_TO_NUSCENES/ \
    data.val.data_root=PATH_TO_NUSCENES/ \
    data.test.data_root=PATH_TO_NUSCENES/ \
    eval_config.ann_file=data/infos/nuscenes_infos_val.pkl \
    eval_config.version=v1.0-trainval
```

---

## 7) 빠른 스모크 테스트

`train_mini.sh`를 사용하면 소규모 반복으로 데이터로더/forward를 먼저 점검할 수 있다.

```bash
cd PATH_TO_HIPAD
./train_mini.sh
```

### 2-epoch 스모크(미니셋, Stage1)

아래 명령은 mini ann을 사용해 약 2 epoch 수준(`max_iters=12`)으로 빠르게 학습만 점검한다.
`data/infos/mini/nuscenes_infos_train.pkl`이 없으면 `data/infos/mini/mini/nuscenes_infos_train.pkl` 경로를 사용한다.

```bash
cd PATH_TO_HIPAD
PYTHONPATH="$(pwd)" python tools/train.py \
  projects/configs/hipad_nusc_stage1.py \
  --work-dir work_dirs/debug_nusc_stage1_2ep_mini \
  --gpus 1 \
  --no-validate \
  --cfg-options \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=1 \
    runner.max_iters=12 \
    log_config.interval=1 \
    checkpoint_config.interval=12 \
    data.train.ann_file=data/infos/mini/nuscenes_infos_train.pkl \
    data.train.version=v1.0-mini \
    data.train.data_root=PATH_TO_NUSCENES/
```

### 2-epoch 스모크(미니셋, Stage2)

Stage1 결과를 이어받아 Stage2를 짧게 확인할 때 사용한다.
`data/infos/mini/nuscenes_infos_train.pkl`이 없으면 `data/infos/mini/mini/nuscenes_infos_train.pkl` 경로를 사용한다.

```bash
cd PATH_TO_HIPAD
PYTHONPATH="$(pwd)" python tools/train.py \
  projects/configs/hipad_nusc_stage2.py \
  --work-dir work_dirs/debug_nusc_stage2_2ep_mini \
  --gpus 1 \
  --no-validate \
  --cfg-options \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=1 \
    runner.max_iters=12 \
    log_config.interval=1 \
    checkpoint_config.interval=12 \
    load_from=work_dirs/debug_nusc_stage1_2ep_mini/latest.pth \
    data.train.ann_file=data/infos/mini/nuscenes_infos_train.pkl \
    data.train.version=v1.0-mini \
    data.train.data_root=PATH_TO_NUSCENES/
```

---

## 8) 평가 실행 예시

mini 체크포인트 평가:
`data/infos/mini/nuscenes_infos_val.pkl`이 없으면 `data/infos/mini/mini/nuscenes_infos_val.pkl` 경로를 사용한다.

```bash
cd PATH_TO_HIPAD
PYTHONPATH="$(pwd)" python tools/test.py \
  projects/configs/hipad_nusc_stage2.py \
  work_dirs/debug_nusc_mini/latest.pth \
  --eval bbox \
  --cfg-options \
    data.test.ann_file=data/infos/mini/nuscenes_infos_val.pkl \
    data.test.version=v1.0-mini \
    data.test.data_root=PATH_TO_NUSCENES/ \
    eval_config.ann_file=data/infos/mini/nuscenes_infos_val.pkl \
    eval_config.version=v1.0-mini
```

---

## 9) 자주 나는 에러와 원인

`FileNotFoundError: data/kmeans/kmeans_det_900.npy`
- kmeans anchors 미생성

`KeyError: 'infos'`
- HiP-AD 포맷이 아닌 다른 pkl 사용

`AssertionError: Database version not found: .../v1.0-trainval-trainval`
- converter 실행 시 `--version v1.0-trainval`을 넣은 경우
- `--version v1.0` 사용 필요

---

## 10) 운영 체크리스트

- `data/infos/nuscenes_infos_train.pkl` 존재
- `data/kmeans/kmeans_*.npy` 4종 존재
- Stage1 학습 완료 및 `work_dirs/hipad_nusc_stage1/latest.pth` 생성
- Stage2 학습 시작 시 `load_from` 경로 확인
