import torch as _torch_for_patch

if not getattr(_torch_for_patch.load, "_hipad_legacy_default_patched", False):
    _orig_torch_load = _torch_for_patch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    _patched_torch_load._hipad_legacy_default_patched = True
    _torch_for_patch.load = _patched_torch_load

# torch 2.6+ `_get_stream` requires torch.device; mmcv 1.7.1 still passes int indices.
# On torch < 2.6 (e.g. 1.13) `_get_stream(device: int)` indexes a list by int,
# so converting to torch.device breaks it -- only apply the patch on 2.6+.
from torch.nn.parallel import _functions as _torch_parallel_funcs
_torch_ver = tuple(int(x) for x in _torch_for_patch.__version__.split("+")[0].split(".")[:2])
if _torch_ver >= (2, 6) and not getattr(_torch_parallel_funcs._get_stream, "_hipad_intdev_patched", False):
    _orig_get_stream = _torch_parallel_funcs._get_stream
    def _patched_get_stream(device):
        if isinstance(device, int):
            device = _torch_for_patch.device("cuda", device)
        return _orig_get_stream(device)
    _patched_get_stream._hipad_intdev_patched = True
    _torch_parallel_funcs._get_stream = _patched_get_stream
    import mmcv.parallel._functions as _mmcv_parallel_funcs
    _mmcv_parallel_funcs._get_stream = _patched_get_stream

from .datasets import *
from .models import *
from .apis import *
from .core.evaluation import *
from .core.hooks import *
