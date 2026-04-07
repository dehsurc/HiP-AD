# HiP-AD nuScenes 학습 가이드

## 1. 환경 구성

### 필수 패키지

```
torch==1.13.0+cu117
mmcv-full==1.7.1
mmdet==2.28.2
flash_attn==2.7.0.post2
```

전체 패키지 목록은 `pip_freeze.txt` 참고.

### Conda 환경 생성

```bash
conda create -n hipad python=3.8
conda activate hipad
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmdet==2.28.2
pip install flash-attn==2.7.0.post2 --no-build-isolation
```

### Deformable Aggregation 빌드

```bash
cd projects/mmdet3d_plugin/ops
pip install -e .
```

### ResNet50 pretrained weight

```bash
mkdir -p ckpts
# data/ 심볼릭 링크 안에 이미 포함되어 있으면 아래처럼 심볼릭 링크
ln -s data/resnet50-19c8e357.pth ckpts/resnet50-19c8e357.pth
```

---

## 2. 데이터 준비

### 디렉토리 구조

모든 config는 `data/` 하위의 상대 경로를 사용한다.
`data/`는 심볼릭 링크로 운영해도 되며, 아래 구조를 만족해야 한다.

```
HiP-AD/
├── data/                          # 심볼릭 링크 가능 (예: ln -s /your/data/path data)
│   ├── nuscenes/                  # 심볼릭 링크 가능 (예: ln -s /your/nuscenes data/nuscenes)
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── maps/
│   │   ├── v1.0-trainval/
│   │   └── v1.0-mini/            # (optional, 스모크 테스트용)
│   ├── infos/                     # 생성 필요 (아래 참고)
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   └── mini/                  # (optional)
│   ├── kmeans/                    # 생성 필요 (아래 참고)
│   │   ├── kmeans_det_900.npy
│   │   ├── kmeans_map_100.npy
│   │   ├── kmeans_motion_6.npy
│   │   └── kmeans_plan_6.npy
│   └── resnet50-19c8e357.pth
├── ckpts/
│   └── resnet50-19c8e357.pth      # 심볼릭 링크 가능
└── work_dirs/                     # 심볼릭 링크 가능 (예: ln -s /fast_ssd/work_dirs work_dirs)
```

### 심볼릭 링크 주의사항

- `data/`, `work_dirs/`는 `.gitignore`에 포함되어 있어 git에 추적되지 않는다.
- **서버마다 심볼릭 링크를 새로 만들어야 한다.**
- `data/nuscenes`가 nuScenes 원본 데이터(v1.0-trainval)를 가리켜야 한다.
- `work_dirs/`는 체크포인트 저장 경로이므로 용량이 큰 디스크에 심볼릭 링크 권장.

```bash
# 예시 (서버 환경에 맞게 수정)
ln -s /your/data/path data
ln -s /fast_ssd/work_dirs work_dirs
ln -s /your/nuscenes data/nuscenes
```

### nuScenes infos 생성

```bash
cd HiP-AD
PYTHONPATH="$(pwd)" python tools/data_converter/nuscenes_converter.py nuscenes \
  --root-path data/nuscenes \
  --canbus data/nuscenes \
  --out-dir data/infos \
  --extra-tag nuscenes \
  --version v1.0
```

> `--version v1.0`을 사용한다. `v1.0-trainval`을 넣으면 에러 발생.

mini (스모크 테스트용):

```bash
PYTHONPATH="$(pwd)" python tools/data_converter/nuscenes_converter.py nuscenes \
  --root-path data/nuscenes \
  --canbus data/nuscenes \
  --out-dir data/infos/mini \
  --extra-tag nuscenes \
  --version v1.0-mini
```

### K-means anchor 생성

```bash
PYTHONPATH="$(pwd)" python tools/kmeans_nuscenes/kmeans_det.py
PYTHONPATH="$(pwd)" python tools/kmeans_nuscenes/kmeans_map.py
PYTHONPATH="$(pwd)" python tools/kmeans_nuscenes/kmeans_motion.py
PYTHONPATH="$(pwd)" python tools/kmeans_nuscenes/kmeans_plan.py
```

생성 확인:

```bash
ls data/kmeans/kmeans_{det_900,map_100,motion_6,plan_6}.npy
```

---

## 3. Baseline 학습

### Config 구조

| Config | 설명 | epoch | GPU | batch | load_from |
|---|---|---|---|---|---|
| `hipad_nusc_stage1.py` | Stage1 (det/map/plan/ego) | 24 | 2 | 8 | - |
| `hipad_nusc_stage2.py` | Stage2 (+ motion) | 36 | 2 | 6 | stage1/latest.pth |

### Stage1 학습

```bash
cd HiP-AD
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
  projects/configs/hipad_nusc_stage1.py 2
```

- 결과: `work_dirs/hipad_nusc_stage1/latest.pth`

### Stage2 학습

Stage1 완료 후 실행. `load_from`이 config에 이미 설정되어 있다.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
  projects/configs/hipad_nusc_stage2.py 2
```

- 결과: `work_dirs/hipad_nusc_stage2/`

### Multi-GPU (4 GPU 예시)

```bash
# config의 num_gpus를 4로 수정하거나, --cfg-options로 override
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh \
  projects/configs/hipad_nusc_stage1.py 4
```

> `num_gpus` 변경 시 `num_iters_per_epoch`가 자동 계산되므로 config 수정만으로 충분.

### 동시에 여러 실험 실행

서로 다른 GPU에서 실행할 때 **PORT를 다르게** 설정해야 한다.

```bash
# 실험 A: GPU 0,1
CUDA_VISIBLE_DEVICES=0,1 PORT=28650 bash tools/dist_train.sh config_a.py 2

# 실험 B: GPU 2,3
CUDA_VISIBLE_DEVICES=2,3 PORT=28651 bash tools/dist_train.sh config_b.py 2
```

---

## 4. Distillation 학습

Teacher cache(BEVFusion 예측값)가 필요하다.

### Teacher cache 준비

```bash
# teacher cache를 data/cache/ 하위에 배치
mkdir -p data/cache/det
cp /path/to/bevfusion_teacher_train.pkl data/cache/det/
```

### Distillation configs

| Config | 설명 | distill_mode | det_gt_loss_weight |
|---|---|---|---|
| `hipad_nusc_stage2_distill.py` | 기본 distill (Teacher TP + GT) | teacher_tp | 1.0 |
| `hipad_nusc_stage2_distill_only.py` | Distill only (GT det loss 제거) | teacher_tp | 0.0 |
| `hipad_nusc_stage2_distill_pseudo_gt.py` | Pseudo GT (teacher를 GT로 사용) | pseudo_gt | 1.0 |

### Distillation 실행

```bash
# Stage1 학습 완료 후
CUDA_VISIBLE_DEVICES=0,1 PORT=28650 bash tools/dist_train.sh \
  projects/configs/hipad_nusc_stage2_distill.py 2
```

---

## 5. 기타 Config variants

| Config | 설명 |
|---|---|
| `hipad_nusc_stage1_3layer.py` | 3-layer 축소 모델 (12ep, embed=128) |
| `hipad_nusc_stage2_3layer.py` | 3-layer Stage2 (18ep) |
| `hipad_nusc_stage2_6ep.py` | Stage2 6ep 단축 학습 |

---

## 6. 평가

```bash
PYTHONPATH="$(pwd)" python tools/test.py \
  projects/configs/hipad_nusc_stage2.py \
  work_dirs/hipad_nusc_stage2/latest.pth \
  --eval bbox
```

---

## 7. 스모크 테스트

mini 데이터로 forward pass만 빠르게 확인:

```bash
PYTHONPATH="$(pwd)" python tools/train.py \
  projects/configs/hipad_nusc_stage1.py \
  --work-dir work_dirs/debug_smoke \
  --gpus 1 \
  --no-validate \
  --cfg-options \
    version="mini" \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=1 \
    runner.max_iters=12 \
    log_config.interval=1 \
    checkpoint_config.interval=12
```

---

## 8. 자주 나는 에러

| 에러 | 원인 | 해결 |
|---|---|---|
| `FileNotFoundError: data/kmeans/kmeans_det_900.npy` | anchor 미생성 | kmeans 스크립트 실행 |
| `KeyError: 'infos'` | 잘못된 pkl 사용 | HiP-AD 전용 converter로 재생성 |
| `AssertionError: v1.0-trainval-trainval` | converter에 `--version v1.0-trainval` 사용 | `--version v1.0` 사용 |
| `No module named 'flash_attn'` | 잘못된 conda 환경 | `conda activate hipad` 확인 |
| `CUDA out of memory` | GPU에 다른 프로세스 존재 | `nvidia-smi` 확인 후 정리 |
| `Address already in use (PORT)` | 동일 PORT로 여러 실험 | `PORT=28651` 등으로 변경 |

---

## 9. 체크리스트 (새 서버 셋업)

1. [ ] conda 환경 생성 및 패키지 설치
2. [ ] `deformable_aggregation_ext` 빌드
3. [ ] `data/` 심볼릭 링크 설정 (또는 디렉토리 생성)
4. [ ] `data/nuscenes` → nuScenes 원본 심볼릭 링크
5. [ ] `work_dirs/` 심볼릭 링크 설정
6. [ ] `ckpts/resnet50-19c8e357.pth` 배치
7. [ ] nuScenes infos 생성 (`data/infos/*.pkl`)
8. [ ] K-means anchor 생성 (`data/kmeans/*.npy`)
9. [ ] Stage1 학습 실행
10. [ ] Stage2 학습 실행
