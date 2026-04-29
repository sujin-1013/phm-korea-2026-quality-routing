"""Minimal LoRA injection for MOMENT (T5 encoder).

Hand-rolled to avoid the ``peft`` dependency. Walks the model graph, replaces
attention projections (``q``, ``v`` by default — T5 naming) with a
``LoRALinear`` wrapper that keeps the base weight frozen and adds a low-rank
update ``B @ A`` initialised so the wrapped layer is a no-op at step 0.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an ``nn.Linear`` with a low-rank trainable update.

    ``y = W x + (alpha/r) · (B A) x`` where ``W`` is the frozen base weight,
    ``A ∈ R^{r×in}``, ``B ∈ R^{out×r}``. ``B`` starts at zero so the layer is
    output-equivalent to the base ``Linear`` immediately after injection.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.base = base
        self.base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # Leave B at zero so the LoRA contribution is exactly 0 at init.
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + ((x @ self.A.T) @ self.B.T) * self.scaling


def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_names: tuple[str, ...] = ("q", "v"),
) -> list[nn.Parameter]:
    """Replace every ``nn.Linear`` whose attribute name is in ``target_names``.

    Returns the list of newly created LoRA parameters (so the training loop can
    pass them to the optimizer). Does not duplicate base parameters.

    The base ``model``'s parameters are left untouched in terms of identity but
    have ``requires_grad`` set to ``False`` for any wrapped Linear's weight/bias.
    """
    targets = set(target_names)
    lora_params: list[nn.Parameter] = []
    wrapped_count = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                wrapped = LoRALinear(child, r=r, alpha=alpha).to(child.weight.device)
                setattr(module, child_name, wrapped)
                lora_params.extend([wrapped.A, wrapped.B])
                wrapped_count += 1
    if wrapped_count == 0:
        raise RuntimeError(
            f"inject_lora found no Linear modules named {target_names}; check target_names"
        )
    return lora_params


def lora_param_count(params: list[nn.Parameter]) -> int:
    return sum(p.numel() for p in params)
