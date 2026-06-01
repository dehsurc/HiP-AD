import datetime
from typing import Iterable, Optional

import torch
from mmcv.runner import HOOKS
from mmcv.runner.hooks import TextLoggerHook


@HOOKS.register_module()
class ScientificTextLoggerHook(TextLoggerHook):
    """Text logger with scientific notation for selected scalar keys."""

    def __init__(
        self,
        sci_keys: Optional[Iterable[str]] = None,
        sci_prefixes: Optional[Iterable[str]] = None,
        sci_precision: int = 3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sci_keys = set(sci_keys or [])
        self.sci_prefixes = tuple(sci_prefixes or [])
        self.sci_precision = sci_precision

    def _use_scientific(self, name):
        return name in self.sci_keys or any(
            name.startswith(prefix) for prefix in self.sci_prefixes)

    def _format_value(self, name, val):
        if isinstance(val, float):
            if self._use_scientific(name):
                return f"{val:.{self.sci_precision}e}"
            return f"{val:.4f}"
        return val

    def _log_info(self, log_dict, runner):
        if runner.meta is not None and "exp_name" in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                runner.logger.info(f'Exp name: {runner.meta["exp_name"]}')

        if log_dict["mode"] == "train":
            if isinstance(log_dict["lr"], dict):
                lr_str = []
                for k, val in log_dict["lr"].items():
                    lr_str.append(f"lr_{k}: {val:.3e}")
                lr_str = " ".join(lr_str)
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'

            if self.by_epoch:
                log_str = (
                    f'Epoch [{log_dict["epoch"]}]'
                    f'[{log_dict["iter"]}/{len(runner.data_loader)}]\t'
                )
            else:
                log_str = f'Iter [{log_dict["iter"]}/{runner.max_iters}]\t'
            log_str += f"{lr_str}, "

            if "time" in log_dict.keys():
                self.time_sec_tot += log_dict["time"] * self.interval
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f"eta: {eta_str}, "
                log_str += (
                    f'time: {log_dict["time"]:.3f}, '
                    f'data_time: {log_dict["data_time"]:.3f}, '
                )
                if torch.cuda.is_available():
                    log_str += f'memory: {log_dict["memory"]}, '
        else:
            if self.by_epoch:
                log_str = (
                    f'Epoch({log_dict["mode"]}) '
                    f'[{log_dict["epoch"]}][{log_dict["iter"]}]\t'
                )
            else:
                log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

        log_items = []
        for name, val in log_dict.items():
            if name in [
                    "mode", "Epoch", "iter", "lr", "time", "data_time",
                    "memory", "epoch"
            ]:
                continue
            log_items.append(f"{name}: {self._format_value(name, val)}")
        log_str += ", ".join(log_items)

        runner.logger.info(log_str)
