PCGrad (Gradient Surgery) 적용 플랜 for HiP-AD
Context
Gradient conflict 분석 결과, 5개 task(plan/det/map/motion/ego) 간 심각한 gradient conflict 확인:

map_vs_motion 60.9%, plan_vs_det 50%, ego_vs_det 50% conflict rate
plan gradient norm(238) vs map(21) — 11배 magnitude imbalance
conflict hotspot: backbone_stem, backbone_layer1, dec5_norm, dec2-4_ffn_prenorm 등
결정 사항:

Projection: Standard PCGrad + Norm 정규화 (unit norm → projection → norm 복원)
최적화: Selective groups + pcgrad_interval 둘 다 적용
정밀도: FP32
수정 파일 목록
#	파일	변경
1	projects/mmdet3d_plugin/hooks/pcgrad_optimizer_hook.py	신규 — PCGradOptimizerHook 구현
2	projects/mmdet3d_plugin/hooks/__init__.py	신규 — hook 모듈 등록
3	projects/mmdet3d_plugin/models/sparse_detector.py	train_step() override — per-task loss 노출
4	projects/mmdet3d_plugin/apis/mmdet_train.py	PCGrad hook 연결 (line 141-149)
5	projects/configs/hipad_b2d_stage2_pcgrad.py	PCGrad config 파일
재사용할 기존 코드:

tools/analyze_gradient_conflict.py:488 — get_shared_parameters_grouped() 함수 (56개 operation-group 파라미터 추출)
tools/analyze_gradient_conflict.py:75 — TASK_GROUPS dict (task→loss key 매핑)
구현 단계
Step 1: sparse_detector.py — per-task loss 노출 (line 135-145)
train_step() override하여 기존 flat loss dict를 5개 task별 scalar로 그룹핑:

TASK_LOSS_PREFIXES = {
    'det': ['det_loss'],
    'map': ['map_loss'],
    'motion': ['motion_loss'],
    'ego': ['ego_loss'],
    'plan': ['plan_loss'],
}
기존 forward_train() 호출 → flat losses dict 반환
loss key를 prefix 매칭으로 task별 합산 → task_losses = {task: scalar_tensor}
_parse_losses()로 기존 logging 유지
outputs에 task_losses 추가
Step 2: pcgrad_optimizer_hook.py — 핵심 (신규 파일)
mmcv OptimizerHook 상속. @HOOKS.register_module() 등록.

Config 파라미터:

pcgrad_groups: PCGrad 적용 대상 group key 리스트 (selective)
pcgrad_interval: 매 N iter마다 PCGrad (나머지는 standard)
normalize_grads: projection 전 unit norm 정규화 (True)
warmup_iters: PCGrad 시작 전 warm-up iteration 수
shared_layers: get_shared_parameters_grouped()에 전달할 layer 이름
after_train_iter(runner) 로직:

if iter < warmup_iters OR iter % pcgrad_interval != 0:
    → 기존 OptimizerHook.after_train_iter() 호출 (standard backward)
    return

1. lazy init: get_shared_parameters_grouped()로 56개 group 구성
2. task_losses = runner.outputs['task_losses']  # {task: scalar}
3. T번 backward pass:
   for i, task in enumerate(tasks):
     shared_params만 zero_grad (non-shared는 누적)
     task_losses[task].backward(retain_graph=(i < T-1))
     per_task_grads[task][group_key] = flatten+clone(group params' grad)
4. PCGrad projection (per group):
   for group in pcgrad_groups:
     task_grads = [per_task_grads[t][group] for t in tasks]
     norms = [g.norm() for g in task_grads]
     task_grads_normed = [g / (n + eps) for g,n in zip(task_grads, norms)]
     for i in range(T):
       for j (random order):
         if dot(projected[i], grads_normed[j]) < 0:
           projected[i] -= dot * grads_normed[j] / ||grads_normed[j]||²
     projected = [p * n for p,n in zip(projected, norms)]  # norm 복원
     merged = mean(projected)
     assign back to param.grad
   for non-pcgrad groups:
     merged = sum of per-task grads (standard)
5. grad_clip (max_norm=25)
6. optimizer.step()
Non-shared params 처리:

T번 backward 동안 shared params만 zero_grad → non-shared는 자연 누적
shared param의 id를 set으로 관리하여 구분
Logging (매 pcgrad iteration):

pcgrad/conflict_rate: projection 발생 비율
pcgrad/projection_magnitude: gradient 변화량 평균
Step 3: hooks/__init__.py
from .pcgrad_optimizer_hook import PCGradOptimizerHook
__all__ = ['PCGradOptimizerHook']
Step 4: mmdet_train.py (line 141-149)
기존 optimizer_config 분기에 pcgrad 조건 추가:

pcgrad_cfg = cfg.get("pcgrad", None)
if pcgrad_cfg is not None:
    optimizer_config = PCGradOptimizerHook(
        grad_clip=cfg.optimizer_config.get('grad_clip'), **pcgrad_cfg)
elif fp16_cfg is not None:
    ...  # 기존 로직 유지
Step 5: Config 파일
기존 stage2 config 상속 + pcgrad 설정:

pcgrad = dict(
    shared_layers=['backbone', 'neck', 'operation_order', 'fc_before', 'fc_after'],
    normalize_grads=True,
    pcgrad_interval=5,
    warmup_iters=500,
    pcgrad_groups=[
        'backbone_stem', 'backbone_layer1',
        'dec0_norm_0', 'dec0_norm_1',
        'dec2_ffn_0_prenorm', 'dec3_ffn_0_prenorm', 'dec4_ffn_0_prenorm',
        'dec5_norm_0', 'dec5_norm_1',
    ],
)
핵심 고려사항
항목	결정
Granularity	56개 operation-group (per-parameter는 noisy, 비추)
Magnitude imbalance	unit norm 정규화로 해결
메모리	selective groups + interval로 overhead 최소화
Warm-up	처음 500 iter은 standard training
Grad clip	PCGrad projection 후 적용 (max_norm=25)
DDP	backward 시 자동 all-reduce → projection은 sync된 gradient 위에서 동작
Non-shared params	shared_param_ids set으로 구분, backward 시 자연 누적
검증
Smoke test: 100 iter 실행
conflict 발생 시 projection 후 dot(g_i, g_j) >= 0 검증
non-shared param gradient 정상 누적 확인
pcgrad 로그 출력 확인
로그 모니터링: conflict_rate, projection_magnitude 추이
Full training: baseline 대비 planning (L2, collision) + perception (NDS, mAP) 비교