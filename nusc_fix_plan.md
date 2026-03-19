# nuScenes 적응 코드 수정 계획

B2D에서는 정상 동작하지만 nuScenes 데이터 구조로 오면서 깨지거나 잘못 적용되는 부분만 정리한다.
구조 변경이나 하이퍼파라미터 튜닝은 대상이 아님.

---

## 1. DistributedSampler 타임스탬프 비교 버그

**파일:** `projects/mmdet3d_plugin/datasets/samplers/distributed_sampler.py:53`

**현상:**
```python
abs(timestamps[i] - timestamps[i - 1]) > 4
```
B2D 타임스탬프는 초 단위(또는 작은 정수)이므로 `> 4` threshold가 정상 동작한다.
nuScenes 타임스탬프는 **microsecond** 단위 (예: `1532402927647951`).
연속 keyframe 간 차이가 ~500,000이므로 **모든 sample이 새 시퀀스로 인식**된다.

**영향:**
Multi-GPU 평가 시 시퀀스 경계 없이 개별 sample 단위로 GPU에 분배.
시퀀스 내 연속성이 깨져 temporal feature가 GPU 경계에서 단절될 수 있음.

**수정 방향:**
타임스탬프를 초 단위로 정규화한 뒤 비교.
```python
ts_sec = [x["timestamp"] / 1e6 for x in self.dataset.data_infos]
# ...
abs(ts_sec[i] - ts_sec[i - 1]) > 4
```
또는 dataset에 `scene_token`이 있으므로 scene 변경 여부로 시퀀스 분리:
```python
scene_tokens = [x.get("scene_token") for x in self.dataset.data_infos]
if i == 0 or scene_tokens[i] != scene_tokens[i - 1]:
    # 새 시퀀스
```

---

## 2. Canbus 실패 시 ego_status가 0으로 채워지지만 mask가 1

**파일:** `tools/data_converter/nuscenes_converter.py:436-437`

**현상:**
```python
except:
    ego_status = [0] * 10
```
nuScenes에서 약 11/108 scene은 canbus 데이터가 없어 `[0]*10`으로 fallback.
이후 `nuscenes_3d_dataset.py`의 `_build_ego_status`에서 mask 로직은:
- `|speed| > 20` 일 때만 speed mask = 0
- `norm(accel) > 40` 일 때만 accel mask = 0

0값은 이 조건을 만족하지 않으므로 **mask = 전부 1 (유효)**.
결과적으로 Stage2에서 `loss_ego_status(weight=1.0)`으로 **잘못된 0 GT를 정상값으로 학습**.

B2D에서는 canbus 개념이 없고, 시뮬레이터가 항상 정확한 ego_status를 제공하므로 이 문제가 없음.

**수정 방향 (converter 수준):**
converter에서 canbus 실패 시 별도 flag를 저장.
```python
except:
    ego_status = [0] * 10
    info['ego_status_valid'] = False
```
dataset에서 `ego_status_valid == False`이면 `ego_status_mask`를 전부 0으로 설정:
```python
if not info.get('ego_status_valid', True):
    status_mask[:] = 0.0
```

**또는 (dataset 수준, converter 재생성 불필요):**
ego_status가 전부 0인 경우를 감지하여 mask를 0으로:
```python
if np.all(status == 0):
    status_mask[:] = 0.0
```

---

## 3. `NuScenesSparse4DAdaptor.__init__` 오타

**파일:** `projects/mmdet3d_plugin/datasets/pipelines/transform.py:108`

**현상:**
```python
def __init(self):   # ← __init__ 이어야 함 (밑줄 2개씩)
    pass
```
Python이 이를 생성자로 인식하지 않음. 현재 `__init__`에서 아무 작업도 안 하므로 동작에 영향은 없으나, 향후 초기화 로직 추가 시 문제됨.

**수정:** `__init__` → `__init__`

---

## 수정 불필요 확인 항목

아래는 B2D와 다르지만, 코드상 정상 처리되는 항목들:

| 항목 | 상태 | 이유 |
|------|------|------|
| ego_status 6차원 매핑 `[vx,ax,ay,wx,wy,steer]` | OK | B2D와 동일 순서로 변환됨 |
| ego_status_mask threshold (vel>20, accel>40) | OK | B2D의 `limit_vel=20, limit_accel=20/(0.1*5)=40`과 동일 |
| gt_ego_fut_trajs offset 형태 저장 | OK | loss에서 cumsum 처리됨 |
| gt_ego_fut_trajs_2hz 제공 | OK | plan_anchor_types=("temp","2hz")에 대응 |
| ego_fut_cmd=3 one-hot 처리 | OK | argmax로 인덱싱, shape 정상 |
| kmeans_plan_6.npy (3,6,6,2) → 18 anchors | OK | ego_fut_cmd(3) × ego_fut_mode(6) = 18 |
| EgoStatusRefinementModule status_dims=6 | OK | B2D와 동일 출력 차원 |
| ego_vehicle="nus" (SparsePlanDecoder) | OK | nuScenes 차량 크기 적용, 의도된 설정 |
| plan_speed_refer=None | OK | B2D stage1도 None, anchor_types에 speed 없으므로 사용 안됨 |
