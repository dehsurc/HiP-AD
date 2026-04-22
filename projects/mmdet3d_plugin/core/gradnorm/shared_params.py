from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn


def collect_last_linear_weights(
    operation_order: Sequence[str],
    layers: Sequence[nn.Module],
) -> List[torch.nn.Parameter]:
    """For every position ``i`` where ``operation_order[i] == "ffn"``, return
    ``layers[i].layers``'s last ``nn.Linear.weight`` in order.

    The caller guarantees ``layers[i]`` is an ``AsymmetricFFN``-compatible
    module exposing ``.layers`` as an ``nn.Sequential``-like container.

    Raises AssertionError if an ``"ffn"`` position has no ``nn.Linear``.
    """
    assert len(operation_order) == len(layers), (
        "operation_order and layers must have the same length"
    )
    params: List[torch.nn.Parameter] = []
    for op, module in zip(operation_order, layers):
        if op != "ffn":
            continue
        inner = getattr(module, "layers", None)
        assert inner is not None, "ffn module must expose .layers attribute"
        last_linear = None
        for sub in reversed(list(inner.children())):
            if isinstance(sub, nn.Linear):
                last_linear = sub
                break
        assert last_linear is not None, (
            "ffn module's .layers contains no nn.Linear child"
        )
        params.append(last_linear.weight)
    return params
