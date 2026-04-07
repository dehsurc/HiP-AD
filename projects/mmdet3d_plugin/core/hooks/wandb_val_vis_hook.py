import glob
import os
from typing import List

from mmcv.runner import HOOKS, Hook
from mmdet.utils import get_root_logger


@HOOKS.register_module()
class WandbValVisHook(Hook):
    """Upload validation visualizations to Weights & Biases.

    This hook expects validation visualization files to be generated on eval
    (e.g., via ``evaluation.out_dir``). It uploads a small image panel and an
    optional video after each validation event.
    """

    def __init__(self, vis_dir="val_vis/visual", max_images=8, interval=1):
        self.vis_dir = vis_dir
        self.max_images = max_images
        self.interval = interval
        self._last_logged_iter = -1

    def _resolve_vis_dir(self, runner):
        if os.path.isabs(self.vis_dir):
            return self.vis_dir

        candidates = [
            self.vis_dir,
            os.path.join(runner.work_dir, self.vis_dir),
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        return candidates[0]

    def _list_images(self, vis_dir: str) -> List[str]:
        patterns = ["*.jpg", "*.jpeg", "*.png"]
        image_paths = []
        for pattern in patterns:
            image_paths.extend(glob.glob(os.path.join(vis_dir, pattern)))
        image_paths.sort(key=os.path.getmtime)
        return image_paths

    def after_train_iter(self, runner):
        if (runner.iter + 1) % self.interval != 0:
            return

        # Only rank-0 should upload media.
        if runner.rank != 0:
            return

        # Eval hook writes this key when validation just ran.
        if "eval_iter_num" not in runner.log_buffer.output:
            return

        if self._last_logged_iter == runner.iter:
            return

        logger = get_root_logger()
        try:
            import wandb
        except Exception as exc:
            logger.warning(f"WandbValVisHook skipped: wandb import failed ({exc}).")
            return

        if wandb.run is None:
            logger.warning("WandbValVisHook skipped: wandb.run is not initialized.")
            return

        vis_dir = self._resolve_vis_dir(runner)
        if not os.path.isdir(vis_dir):
            logger.warning(f"WandbValVisHook skipped: visualization directory not found: {vis_dir}")
            return

        image_paths = self._list_images(vis_dir)
        if image_paths:
            picked = image_paths[-self.max_images :]
            media = [wandb.Image(path, caption=os.path.basename(path)) for path in picked]
            wandb.log({"val/visual_samples": media}, step=runner.iter + 1)

        video_path = os.path.join(vis_dir, "video.avi")
        if os.path.isfile(video_path):
            wandb.log(
                {"val/visual_video": wandb.Video(video_path, fps=7, format="avi")},
                step=runner.iter + 1,
            )

        self._last_logged_iter = runner.iter
