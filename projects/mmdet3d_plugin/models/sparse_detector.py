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
from .grid_mask import GridMask

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
        gradnorm=None,
    ):
        super(SparseDetector, self).__init__(init_cfg=init_cfg)
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

        if gradnorm is not None:
            from projects.mmdet3d_plugin.core.gradnorm import GradNormLossWeighter
            self.gradnorm = GradNormLossWeighter(**gradnorm)
            self._gradnorm_csv_path_template = "gradnorm_log.csv"
            self._gradnorm_csv_dumper = None  # lazy-init in _maybe_dump_gn_csv
        else:
            self.gradnorm = None
            self._gradnorm_csv_dumper = None

    @staticmethod
    def _aggregate_task_losses(output, task_names):
        """Sum all ``{task}_loss_*`` items per task into one scalar each."""
        task_losses = {}
        for t in task_names:
            prefix = f"{t}_loss_"
            parts = [v for k, v in output.items() if k.startswith(prefix)]
            assert len(parts) > 0, (
                f"No loss key with prefix '{prefix}' found in head output"
            )
            stacked = torch.stack([
                p if torch.is_tensor(p) else torch.tensor(float(p))
                for p in parts
            ])
            task_losses[t] = stacked.sum()
        return task_losses

    @staticmethod
    def _rename_as_monitor(output, task_prefixes):
        """Rename ``{task}_loss_{name}`` keys to ``monitor_{task}_{name}`` and
        detach their tensors so mmcv ``_parse_losses`` (which sums keys
        containing ``'loss'``) and autograd both skip them.
        """
        new = {}
        for k, v in output.items():
            matched = next((p for p in task_prefixes if k.startswith(p)), None)
            if matched is not None:
                new_k = "monitor_" + k.replace("_loss_", "_", 1)
                new[new_k] = v.detach() if torch.is_tensor(v) else v
            else:
                new[k] = v
        return new

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

        if self.gradnorm is None:
            return output

        task_names = self.gradnorm.task_names
        task_prefixes = {t + "_loss_" for t in task_names}

        task_losses = self._aggregate_task_losses(output, task_names)
        shared_params = self.head.onedecoder_head.collect_ffn_last_fc_params()
        weighted, gn_log = self.gradnorm(task_losses, shared_params)

        output = self._rename_as_monitor(output, task_prefixes)
        output["loss_gradnorm_total"] = weighted

        device = weighted.device
        for k, v in gn_log.items():
            safe_key = "gn_" + k
            # mmcv `_parse_losses` sums any key containing lowercase 'loss'.
            # Sanitize diagnostics so only `loss_gradnorm_total` (and depth)
            # enter the backward sum.
            if "loss" in safe_key:
                safe_key = safe_key.replace("loss", "Loss")
            output[safe_key] = torch.as_tensor(
                v, device=device, dtype=torch.float32
            )

        self._maybe_dump_gn_csv(gn_log, task_losses, data)
        return output

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

    def _maybe_dump_gn_csv(self, gn_log, task_losses, data):
        """Write one CSV row per iter on rank 0 with original + derived metrics."""
        import os
        from datetime import datetime

        try:
            import torch.distributed as dist
            rank = (
                dist.get_rank()
                if (dist.is_available() and dist.is_initialized())
                else 0
            )
        except Exception:
            rank = 0

        if rank != 0:
            return

        if self._gradnorm_csv_dumper is None:
            from projects.mmdet3d_plugin.core.gradnorm.csv_dumper import (
                GradNormCSVDumper,
            )
            work_dir = os.environ.get("HIPAD_WORK_DIR") or os.getcwd()
            csv_path = os.path.join(work_dir, self._gradnorm_csv_path_template)
            self._gradnorm_csv_dumper = GradNormCSVDumper(
                csv_path, rank=0, roll_existing=True
            )

        task_names = self.gradnorm.task_names
        w_vec = self.gradnorm.w.detach().cpu()
        init_w = self.gradnorm.init_weights_cached

        # weighted_L_i: from log if available, else recompute from current w
        weighted_vals = []
        for i, n in enumerate(task_names):
            key = f"weighted_L_{n}"
            if key in gn_log:
                weighted_vals.append(float(gn_log[key]))
            else:
                L = float(task_losses[n].detach().cpu().item())
                weighted_vals.append(float(w_vec[i].item()) * L)

        w_sum_iter = sum(weighted_vals)
        contribution_frac = [
            (v / w_sum_iter) if w_sum_iter > 0 else float("nan")
            for v in weighted_vals
        ]

        grad_norm_vals = [
            gn_log.get(f"grad_norm_{n}", float("nan")) for n in task_names
        ]
        gn_sum = sum(v for v in grad_norm_vals if v == v)
        grad_norm_share = [
            (v / gn_sum) if (gn_sum and v == v) else float("nan")
            for v in grad_norm_vals
        ]

        w_ratio = [
            float(w_vec[i].item()) / float(init_w[i])
            for i in range(len(task_names))
        ]

        # metadata from GradNormMetadataHook env vars
        try:
            epoch = int(os.environ.get("HIPAD_CURRENT_EPOCH", "-1"))
            iter_in_epoch = int(
                os.environ.get("HIPAD_CURRENT_ITER", str(int(gn_log.get("step", 0))))
            )
            lr_model = float(os.environ.get("HIPAD_CURRENT_LR", "nan"))
        except Exception:
            epoch, iter_in_epoch, lr_model = -1, int(gn_log.get("step", 0)), float("nan")

        row = {
            "step": int(gn_log.get("step", 0)),
            "epoch": epoch,
            "iter_in_epoch": iter_in_epoch,
            "phase_id": int(gn_log.get("phase_id", 0)),
            "lr_model": lr_model,
        }
        for i, n in enumerate(task_names):
            row[f"w_{n}"] = float(w_vec[i].item())
        for i, n in enumerate(task_names):
            row[f"grad_norm_{n}"] = grad_norm_vals[i]
        for i, n in enumerate(task_names):
            row[f"rt_{n}"] = gn_log.get(f"rt_{n}", float("nan"))
        for i, n in enumerate(task_names):
            row[f"L_{n}"] = gn_log.get(
                f"L_{n}", float(task_losses[n].detach().cpu().item())
            )
        for i, n in enumerate(task_names):
            row[f"L0_{n}"] = gn_log.get(f"L0_{n}", float("nan"))
        for i, n in enumerate(task_names):
            row[f"weighted_L_{n}"] = weighted_vals[i]
        for i, n in enumerate(task_names):
            row[f"contribution_frac_{n}"] = contribution_frac[i]
        for i, n in enumerate(task_names):
            row[f"grad_norm_share_{n}"] = grad_norm_share[i]
        for i, n in enumerate(task_names):
            row[f"w_ratio_{n}"] = w_ratio[i]
        row["gn_loss"] = gn_log.get("gn_loss", float("nan"))
        row["timestamp_iso"] = datetime.utcnow().isoformat()

        self._gradnorm_csv_dumper.append(row)
