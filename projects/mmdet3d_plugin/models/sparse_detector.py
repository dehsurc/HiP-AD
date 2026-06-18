from inspect import signature

import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from mmdet.core import multi_apply
from .grid_mask import GridMask

# Task-to-loss-prefix mapping for PCGrad per-task loss grouping
TASK_LOSS_PREFIXES = {
    'det': ['det_loss'],
    'map': ['map_loss'],
    'motion': ['motion_loss'],
    'ego': ['ego_loss'],
    'plan': ['plan_loss'],
}

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseDetector"]


@DETECTORS.register_module()
class SparseDetector(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
        scenes_tokenizer=None,
        ablate_tasks=None,
    ):
        super(SparseDetector, self).__init__(init_cfg=init_cfg)
        # Leave-one-task-out transfer-gain runs: losses of these tasks are
        # detached in forward_train — values still logged, gradients cut.
        self.ablate_tasks = list(ablate_tasks) if ablate_tasks else []
        for _t in self.ablate_tasks:
            assert _t in TASK_LOSS_PREFIXES, \
                f"ablate_tasks entry {_t!r} not in {list(TASK_LOSS_PREFIXES)}"
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if scenes_tokenizer is not None:
            self.scenes_tokenizer = build_from_cfg(scenes_tokenizer, PLUGIN_LAYERS)
        else:
            self.scenes_tokenizer = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            ) 

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps


    def extract_scenes(self, img, feature_maps, data):
        scenes_tokens, scenes_embeds = self.scenes_tokenizer(img, feature_maps)
        data['scenes_tokens'] = scenes_tokens
        data['scenes_embeds'] = scenes_embeds
        data['temp_scenes_tokens'] = self.scenes_tokenizer.cached_scenes_tokens
        data['temp_scenes_embeds'] = self.scenes_tokenizer.cached_scenes_embeds

        # extract future feats
        if "fut_img" in data and self.training:
            fut_img = data["fut_img"]
            fut_mask = data["fut_mask"]
            fut_data = {key.split("fut_")[-1]: value for key, value in data.items() if key.startswith("fut_")}
            with torch.no_grad():
                fut_feature_maps, fut_depths = self.extract_feat(fut_img, True, fut_data)
                fut_data = self.extract_scenes(fut_img, fut_feature_maps, fut_data)
            data["fut_mask"] = fut_mask
            data["fut_scenes_tokens"] = fut_data['scenes_tokens']
            data["fut_scenes_embeds"] = fut_data['scenes_embeds']

        return data

    def extract_fut_feat(self, img, feature_maps, data):
        fut_img = data["fut_img"]
        fut_mask = data["fut_mask"]
        fut_data = {key.split("fut_")[-1]: value for key, value in data.items() if key.startswith("fut_")}
        with torch.no_grad():
            fut_feature_maps = self.extract_feat(fut_img, False, fut_data)
        data["fut_mask"] = fut_mask
        data["fut_feature_maps"] = fut_feature_maps
        return data

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        if "fut_img" in data and self.training:
            data = self.extract_fut_feat(img, feature_maps, data)
        model_outs = self.head(img, feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        if self.ablate_tasks:
            output = self._cut_ablated_grads(output)
        return output

    def _cut_ablated_grads(self, losses):
        """Zero an ablated task's loss gradient while keeping its value + graph.

        Replaces each matched loss L with ``L*0 + L.detach()``: the value is
        unchanged (``_parse_losses`` still logs it, so the ablated task can be
        watched degrading), the gradient w.r.t. every parameter is exactly zero
        (the task trains neither the shared params nor its own), and the tensor
        stays in the autograd graph so DDP's reducer marks those parameters
        ready exactly once.

        Why not a bare ``.detach()``: that orphans the task's exclusive params
        (e.g. motion_refine) -> with find_unused_parameters=False DDP raises
        "Expected to have finished reduction". Why not flip
        find_unused_parameters=True instead: it clashes with the backbone's
        gradient checkpointing (with_cp=True) -> "mark ready only once". The
        zero-multiply keeps the default DDP path that already works for tg_full.
        """
        prefixes = tuple(
            p for t in self.ablate_tasks for p in TASK_LOSS_PREFIXES[t]
        )
        matched = set()
        for key, val in losses.items():
            if isinstance(val, torch.Tensor) and key.startswith(prefixes):
                losses[key] = val * 0.0 + val.detach()
                matched.add(key)
        # Guard: any OTHER loss key belonging to an ablated task (e.g. distill
        # keys det_kd_loss_*/det_pgt_loss_* that fall outside TASK_LOSS_PREFIXES)
        # would silently keep training it. Fail loudly so a future config change
        # cannot quietly invalidate the experiment.
        leaked = [
            k for k, v in losses.items()
            if k not in matched and isinstance(v, torch.Tensor) and v.requires_grad
            and any(k.startswith(f"{t}_") for t in self.ablate_tasks)
        ]
        assert not leaked, (
            f"ablate_tasks={self.ablate_tasks} but these loss keys are outside "
            f"TASK_LOSS_PREFIXES and still train the task: {leaked}"
        )
        return losses

    def train_step(self, data, optimizer):
        """Override train_step to expose per-task losses for PCGrad.

        The standard BaseDetector.train_step() calls _parse_losses() which
        sums all losses into a single scalar. Here we additionally group
        individual loss terms by task so that PCGradOptimizerHook can
        perform per-task backward passes.
        """
        losses = self(**data)

        # Group losses by task using prefix matching
        task_losses = {}
        matched_keys = set()
        for task, prefixes in TASK_LOSS_PREFIXES.items():
            task_sum = None
            for key, val in losses.items():
                if isinstance(val, torch.Tensor) and val.requires_grad:
                    if any(key.startswith(p) for p in prefixes):
                        task_sum = val if task_sum is None else task_sum + val
                        matched_keys.add(key)
            if task_sum is not None:
                task_losses[task] = task_sum

        # Auxiliary losses: trained normally but excluded from PCGrad
        # projection / conflict analysis (e.g., loss_dense_depth).
        aux_losses = {}
        for key, val in losses.items():
            if isinstance(val, torch.Tensor) and val.requires_grad and key not in matched_keys:
                aux_losses[key] = val

        # Standard _parse_losses for logging (creates 'loss' scalar + log_vars)
        loss, log_vars = self._parse_losses(losses)

        # Determine batch size safely
        if 'img_metas' in data:
            num_samples = len(data['img_metas'])
        elif 'img' in data:
            img = data['img']
            num_samples = img.data.shape[0] if hasattr(img, 'data') else img.shape[0]
        else:
            num_samples = 1

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=num_samples,
            task_losses=task_losses,
            aux_losses=aux_losses,
        )
        return outputs

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)
        model_outs = self.head(img, feature_maps, data)
        results = self.head.post_process(model_outs, data)

        output = []
        for result in results:
            out_dict = {}
            if 'metric_results' in result:
                metric_results = result.pop('metric_results')
                out_dict['metric_results'] = metric_results
            out_dict['img_bbox'] = result
            output.append(out_dict)

        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
