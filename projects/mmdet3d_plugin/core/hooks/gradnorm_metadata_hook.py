import os

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class GradNormMetadataHook(Hook):
    """Publishes epoch / iter_in_epoch / current model lr into env vars so
    ``SparseDetector._maybe_dump_gn_csv`` can record them without threading
    the runner object through forward_train.
    """

    def before_train_iter(self, runner):
        os.environ["HIPAD_CURRENT_EPOCH"] = str(int(runner.epoch))
        os.environ["HIPAD_CURRENT_ITER"] = str(int(runner.iter))
        lr = None
        try:
            lr = runner.current_lr()[0]
        except Exception:
            pass
        if lr is not None:
            os.environ["HIPAD_CURRENT_LR"] = f"{float(lr):.6e}"
        os.environ["HIPAD_WORK_DIR"] = str(
            getattr(runner, "work_dir", os.getcwd())
        )
