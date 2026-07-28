from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class LossWeightWarmupHook(Hook):
    """Linearly ramp specific loss weights from 0 to their target over the
    first ``warmup_iters`` training iterations.

    Use-case: smoothing the stage1 -> stage2 transition where newly-activated
    task losses (motion/plan/ego) produce large gradient bursts through
    randomly-initialised heads, damaging the shared det/map backbone features
    at training onset.

    Target weights are snapshotted at ``before_run`` from the loss modules'
    ``loss_weight`` attribute. From iter 0 the weight is 0 and grows linearly
    to the target at iter ``warmup_iters``; it is clamped to the target
    thereafter.
    """

    def __init__(self, warmup_iters, loss_names, head_path="head.onedecoder_head"):
        self.warmup_iters = int(warmup_iters)
        self.loss_names = list(loss_names)
        self.head_path = head_path
        self._targets = None

    def _resolve_head(self, runner):
        model = runner.model
        if hasattr(model, "module"):
            model = model.module
        obj = model
        for part in self.head_path.split("."):
            obj = getattr(obj, part)
        return obj

    def before_run(self, runner):
        head = self._resolve_head(runner)
        self._targets = {}
        for name in self.loss_names:
            loss_mod = getattr(head, name, None)
            if loss_mod is None or not hasattr(loss_mod, "loss_weight"):
                raise AttributeError(
                    f"LossWeightWarmupHook: {self.head_path}.{name} "
                    f"missing or has no loss_weight attribute")
            self._targets[name] = float(loss_mod.loss_weight)

    def before_train_iter(self, runner):
        head = self._resolve_head(runner)
        if runner.iter >= self.warmup_iters:
            frac = 1.0
        else:
            frac = runner.iter / max(1, self.warmup_iters)
        for name, target in self._targets.items():
            getattr(head, name).loss_weight = target * frac
