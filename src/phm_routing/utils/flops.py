"""FLOPs measurement helpers.

We use fvcore.FlopCountAnalysis. MOMENT has dynamic control flow, so we
disable warnings and rely on the per-module op count it can resolve.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn


def count_flops(model: nn.Module, dummy_input: torch.Tensor, *, kw: str | None = None) -> int:
    """Return the total FLOP count for one forward pass.

    Pass ``kw`` to supply the kwarg name when ``model`` expects ``model(**{kw: x})``.
    """
    from fvcore.nn import FlopCountAnalysis

    model.eval()
    if kw is None:
        flops = FlopCountAnalysis(model, dummy_input)
    else:
        wrapper = _KwargWrapper(model, kw)
        flops = FlopCountAnalysis(wrapper, dummy_input)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return int(flops.total())


class _KwargWrapper(nn.Module):
    def __init__(self, model: nn.Module, kw: str):
        super().__init__()
        self.model = model
        self.kw = kw

    def forward(self, x: torch.Tensor):
        return self.model(**{self.kw: x})
