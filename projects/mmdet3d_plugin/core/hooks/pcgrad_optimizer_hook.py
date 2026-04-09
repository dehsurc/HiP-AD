"""PCGrad (Projecting Conflicting Gradients) Optimizer Hook for HiP-AD.

Implements gradient surgery to resolve gradient conflicts between tasks
(det, map, motion, ego, plan) in shared decoder parameters.

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
"""

import random
import logging
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.runner import HOOKS, OptimizerHook


logger = logging.getLogger(__name__)


def get_decoder(model: nn.Module):
    """Navigate to the unified decoder from model root."""
    if hasattr(model, 'module'):
        model = model.module
    head = model.head
    if hasattr(head, 'onedecoder_head'):
        return head.onedecoder_head
    if hasattr(head, 'sparse_head'):
        return head.sparse_head
    if hasattr(head, 'operation_order'):
        return head
    raise AttributeError("Could not find unified decoder in model structure")


def compute_decoder_layer_mapping(operation_order: List[str]) -> Dict[int, int]:
    """Map each operation index to its decoder layer index (0-based).
    Each decoder layer ends with a 'refine' operation.
    """
    mapping = {}
    dec_idx = 0
    for i, op in enumerate(operation_order):
        mapping[i] = dec_idx
        if op == 'refine':
            dec_idx += 1
    return mapping


def get_shared_parameters_grouped(
    model: nn.Module,
    shared_layer_names: List[str],
) -> Tuple[OrderedDict, set]:
    """Extract shared parameters grouped by (decoder_layer, operation_type).

    Also adds backbone and neck parameter groups if specified.

    Returns:
        param_groups: OrderedDict[group_key -> list of Parameters]
        shared_param_ids: set of parameter ids (for distinguishing from non-shared)
    """
    if hasattr(model, 'module'):
        raw_model = model.module
    else:
        raw_model = model

    param_groups = OrderedDict()
    shared_param_ids = set()

    # Backbone groups
    if 'backbone' in shared_layer_names and hasattr(raw_model, 'img_backbone'):
        backbone = raw_model.img_backbone
        # Group by stage
        if hasattr(backbone, 'stem'):
            params = [p for p in backbone.stem.parameters() if p.requires_grad]
            if params:
                param_groups['backbone_stem'] = params
        # conv1 + bn1 (ResNet without stem)
        for name in ['conv1', 'bn1']:
            mod = getattr(backbone, name, None)
            if mod is not None:
                params = [p for p in mod.parameters() if p.requires_grad]
                if params:
                    param_groups.setdefault('backbone_stem', []).extend(params)

        for i in range(1, 5):
            layer = getattr(backbone, f'layer{i}', None)
            if layer is not None:
                params = [p for p in layer.parameters() if p.requires_grad]
                if params:
                    param_groups[f'backbone_layer{i}'] = params

    # Neck groups
    if 'neck' in shared_layer_names and hasattr(raw_model, 'img_neck'):
        params = [p for p in raw_model.img_neck.parameters() if p.requires_grad]
        if params:
            param_groups['neck'] = params

    # Decoder operation groups
    decoder = get_decoder(model)
    operation_order = decoder.operation_order
    dec_mapping = compute_decoder_layer_mapping(operation_order)

    op_count_per_dec: Dict[Tuple[int, str], int] = defaultdict(int)

    for i, (op, layer) in enumerate(zip(operation_order, decoder.layers)):
        if layer is None:
            continue
        if op not in shared_layer_names:
            continue

        dec_idx = dec_mapping[i]
        count = op_count_per_dec[(dec_idx, op)]
        op_count_per_dec[(dec_idx, op)] += 1
        group_key = f"dec{dec_idx}_{op}_{count}"

        params = [p for _, p in layer.named_parameters() if p.requires_grad]
        if params:
            param_groups[group_key] = params

    # fc_before / fc_after
    for fc_name in ['fc_before', 'fc_after']:
        if fc_name not in shared_layer_names:
            continue
        fc_mod = getattr(decoder, fc_name, None)
        if fc_mod is None or isinstance(fc_mod, nn.Identity):
            continue
        params = [p for _, p in fc_mod.named_parameters() if p.requires_grad]
        if params:
            param_groups[fc_name] = params

    # Collect all shared param ids
    for params in param_groups.values():
        for p in params:
            shared_param_ids.add(id(p))

    return param_groups, shared_param_ids


@HOOKS.register_module()
class PCGradOptimizerHook(OptimizerHook):
    """Optimizer hook with PCGrad for multi-task gradient conflict resolution.

    Args:
        shared_layers: Layer names to include in shared parameter groups.
        pcgrad_groups: Subset of group keys to apply PCGrad projection.
            If None, applies to all groups.
        pcgrad_interval: Apply PCGrad every N iterations (standard otherwise).
        normalize_grads: Unit-norm normalize before projection, restore after.
        warmup_iters: Standard training for first N iterations before PCGrad.
        grad_clip: Gradient clipping config (max_norm, norm_type).
    """

    def __init__(
        self,
        shared_layers: List[str],
        pcgrad_groups: Optional[List[str]] = None,
        pcgrad_interval: int = 1,
        normalize_grads: bool = True,
        warmup_iters: int = 500,
        grad_clip: Optional[dict] = None,
        log_interval: int = 50,
    ):
        # Pass grad_clip to parent OptimizerHook
        super().__init__(grad_clip=grad_clip)
        self.shared_layers = shared_layers
        self.pcgrad_groups = pcgrad_groups
        self.pcgrad_interval = pcgrad_interval
        self.normalize_grads = normalize_grads
        self.warmup_iters = warmup_iters
        self.log_interval = log_interval

        # Lazy initialized
        self._param_groups = None
        self._shared_param_ids = None
        self._initialized = False

    def _lazy_init(self, model):
        """Initialize parameter groups on first PCGrad iteration."""
        if self._initialized:
            return
        self._param_groups, self._shared_param_ids = \
            get_shared_parameters_grouped(model, self.shared_layers)

        group_keys = list(self._param_groups.keys())
        total_params = sum(
            sum(p.numel() for p in params)
            for params in self._param_groups.values()
        )
        logger.info(
            f"[PCGrad] Initialized {len(group_keys)} parameter groups "
            f"({total_params:,} parameters total)"
        )
        logger.info(f"[PCGrad] Groups: {group_keys}")

        if self.pcgrad_groups is not None:
            active = [g for g in self.pcgrad_groups if g in self._param_groups]
            logger.info(f"[PCGrad] Active PCGrad groups: {active}")
        self._initialized = True

    def _should_use_pcgrad(self, runner) -> bool:
        """Check if PCGrad should be applied this iteration."""
        cur_iter = runner.iter
        if cur_iter < self.warmup_iters:
            return False
        if (cur_iter - self.warmup_iters) % self.pcgrad_interval != 0:
            return False
        return True

    def _flatten_grads(self, params: List[nn.Parameter]) -> torch.Tensor:
        """Flatten gradients of parameters into a single vector."""
        grads = []
        for p in params:
            if p.grad is not None:
                grads.append(p.grad.data.flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        return torch.cat(grads)

    def _assign_grads(self, params: List[nn.Parameter], flat_grad: torch.Tensor):
        """Assign a flat gradient vector back to parameter .grad fields."""
        offset = 0
        for p in params:
            numel = p.numel()
            if p.grad is None:
                p.grad = flat_grad[offset:offset + numel].reshape(p.shape).clone()
            else:
                p.grad.data.copy_(flat_grad[offset:offset + numel].reshape(p.shape))
            offset += numel

    def _pcgrad_project(self, task_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply PCGrad projection with optional norm normalization.

        For each task gradient g_i, project away conflicting components
        from other task gradients (random order).
        """
        T = len(task_grads)
        if T <= 1:
            return task_grads

        eps = 1e-8

        # Save original norms for restoration
        if self.normalize_grads:
            norms = [g.norm() for g in task_grads]
            normed = [g / (n + eps) for g, n in zip(task_grads, norms)]
        else:
            normed = [g.clone() for g in task_grads]
            norms = None

        projected = [g.clone() for g in normed]

        for i in range(T):
            order = list(range(T))
            order.remove(i)
            random.shuffle(order)
            for j in order:
                dot = torch.dot(projected[i], normed[j])
                if dot < 0:
                    projected[i] -= dot / (normed[j].norm() ** 2 + eps) * normed[j]

        # Restore original magnitudes
        if self.normalize_grads and norms is not None:
            projected = [p * n for p, n in zip(projected, norms)]

        return projected

    def _compute_conflict_stats(self, per_task_grads, tasks, pcgrad_set):
        """Compute detailed conflict statistics across all task pairs and groups.

        Returns:
            stats: dict with all logging metrics
            pair_conflicts: dict[(task_i, task_j)] -> (conflict_count, total_groups, cosines)
            group_conflicts: dict[group_key] -> conflict_count across pairs
        """
        T = len(tasks)
        eps = 1e-8

        # Per task-pair: conflict count, cosine similarities
        pair_conflicts = {}
        for ii in range(T):
            for jj in range(ii + 1, T):
                pair_conflicts[(tasks[ii], tasks[jj])] = {
                    'conflicts': 0, 'total': 0, 'cosines': [],
                }

        # Per group: conflict count across all pairs
        group_conflicts = defaultdict(int)
        group_total = defaultdict(int)

        # Per task: gradient norm across all groups (concatenated)
        task_grad_all = {t: [] for t in tasks}

        for gk in per_task_grads[tasks[0]]:
            if gk not in pcgrad_set:
                continue
            task_grad_list = [per_task_grads[t][gk] for t in tasks]

            # Accumulate for per-task norm
            for t, g in zip(tasks, task_grad_list):
                task_grad_all[t].append(g)

            for ii in range(T):
                for jj in range(ii + 1, T):
                    pair_key = (tasks[ii], tasks[jj])
                    g_i, g_j = task_grad_list[ii], task_grad_list[jj]
                    norm_i, norm_j = g_i.norm(), g_j.norm()

                    if norm_i < eps or norm_j < eps:
                        continue

                    cos = (torch.dot(g_i, g_j) / (norm_i * norm_j)).item()
                    pair_conflicts[pair_key]['cosines'].append(cos)
                    pair_conflicts[pair_key]['total'] += 1
                    group_total[gk] += 1

                    if cos < 0:
                        pair_conflicts[pair_key]['conflicts'] += 1
                        group_conflicts[gk] += 1

        # Build stats dict
        stats = {}

        # 1) Overall conflict rate
        total_conflict = sum(v['conflicts'] for v in pair_conflicts.values())
        total_count = sum(v['total'] for v in pair_conflicts.values())
        stats['pcgrad/conflict_rate'] = total_conflict / max(total_count, 1)

        # 2) Per task-pair: conflict rate + mean cosine
        for (t_i, t_j), info in pair_conflicts.items():
            prefix = f'pcgrad/{t_i}_vs_{t_j}'
            rate = info['conflicts'] / max(info['total'], 1)
            stats[f'{prefix}/conflict_rate'] = rate
            if info['cosines']:
                stats[f'{prefix}/mean_cosine'] = sum(info['cosines']) / len(info['cosines'])
                stats[f'{prefix}/min_cosine'] = min(info['cosines'])

        # 3) Per task: gradient norm (concatenated across all pcgrad groups)
        for t in tasks:
            if task_grad_all[t]:
                full_grad = torch.cat(task_grad_all[t])
                stats[f'pcgrad/grad_norm/{t}'] = full_grad.norm().item()

        # 4) Gradient magnitude ratio (max/min across tasks)
        task_norms = {t: stats.get(f'pcgrad/grad_norm/{t}', 0.0) for t in tasks}
        norms_list = [n for n in task_norms.values() if n > 0]
        if len(norms_list) >= 2:
            stats['pcgrad/grad_norm_ratio'] = max(norms_list) / (min(norms_list) + eps)

        # 5) Per group: conflict rate (top-level groups only, to limit log volume)
        for gk in group_conflicts:
            stats[f'pcgrad/group/{gk}/conflict_rate'] = \
                group_conflicts[gk] / max(group_total[gk], 1)

        stats['pcgrad/num_tasks'] = T

        return stats, pair_conflicts, group_conflicts

    def after_train_iter(self, runner):
        """Main hook: either standard backward or PCGrad backward.

        Standard mmcv OptimizerHook flow: zero_grad -> backward -> clip -> step.
        PCGrad flow: zero_grad -> T backward passes -> project -> clip -> step.
        """
        # Standard path: warmup or non-pcgrad iterations
        if not self._should_use_pcgrad(runner):
            runner.optimizer.zero_grad()
            runner.outputs['loss'].backward()
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                            runner.outputs['num_samples'])
            runner.optimizer.step()
            return

        # PCGrad path
        self._lazy_init(runner.model)

        task_losses = runner.outputs.get('task_losses', {})
        tasks = list(task_losses.keys())
        T = len(tasks)

        if T <= 1:
            # Fallback to standard if only 0-1 tasks have loss
            runner.optimizer.zero_grad()
            runner.outputs['loss'].backward()
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                            runner.outputs['num_samples'])
            runner.optimizer.step()
            return

        model = runner.model

        # ---- Step 1: Per-task backward passes, collect shared grads ----
        runner.optimizer.zero_grad()

        per_task_grads = {task: {} for task in tasks}

        for i, task in enumerate(tasks):
            # Zero only shared params before each task backward
            # (non-shared params accumulate naturally = sum of task grads)
            for params in self._param_groups.values():
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()

            retain = (i < T - 1)
            task_losses[task].backward(retain_graph=retain)

            for gk, params in self._param_groups.items():
                per_task_grads[task][gk] = self._flatten_grads(params).clone()

        # ---- Step 2: Compute conflict stats (before projection) ----
        if self.pcgrad_groups is not None:
            pcgrad_set = set(self.pcgrad_groups)
        else:
            pcgrad_set = set(self._param_groups.keys())

        should_log_detail = (runner.iter % self.log_interval == 0)

        if should_log_detail:
            stats, _, _ = self._compute_conflict_stats(
                per_task_grads, tasks, pcgrad_set)

        # ---- Step 3: PCGrad projection per group ----
        projection_mag_sum = 0.0
        projection_count = 0

        for gk, params in self._param_groups.items():
            task_grad_list = [per_task_grads[t][gk] for t in tasks]

            if gk in pcgrad_set:
                original = [g.clone() for g in task_grad_list]
                projected = self._pcgrad_project(task_grad_list)

                for orig, proj in zip(original, projected):
                    projection_mag_sum += (proj - orig).norm().item()
                    projection_count += 1

                merged = torch.stack(projected).mean(dim=0)
            else:
                merged = torch.stack(task_grad_list).sum(dim=0)

            self._assign_grads(params, merged)

        # Non-shared params accumulated naturally across T backward passes.
        # sum of per-task grads = grad of total loss, so no adjustment needed.

        # ---- Step 4: Grad clip + optimizer step ----
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(model.parameters())
            if grad_norm is not None:
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                        runner.outputs['num_samples'])

        runner.optimizer.step()

        # ---- Step 5: Logging ----
        avg_proj_mag = projection_mag_sum / max(projection_count, 1)

        if should_log_detail:
            # Detailed per-pair, per-task, per-group logging
            stats['pcgrad/projection_magnitude'] = avg_proj_mag
            runner.log_buffer.update(stats, runner.outputs['num_samples'])
        else:
            # Lightweight logging on non-detail iterations
            runner.log_buffer.update({
                'pcgrad/projection_magnitude': avg_proj_mag,
                'pcgrad/num_tasks': T,
            }, runner.outputs['num_samples'])
