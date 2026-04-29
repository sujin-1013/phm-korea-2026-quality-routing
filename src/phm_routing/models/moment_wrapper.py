"""Thin wrapper around momentfm's MOMENTPipeline for our 4-class fault task.

Verified API (transformers 5.7.0, momentfm 0.1.4):
- ``MOMENTPipeline.from_pretrained(name, model_kwargs={...})`` then ``.init()``.
- Forward: ``model(x_enc=x)`` where x is ``(B, n_channels, T)``; nominal T=512.
- Output: ``TimeseriesOutputs`` with ``.logits`` of shape ``(B, num_class)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

# Importing momentfm raises noisy warnings about transformers/numpy version mismatches
# but the code paths we use (T5 encoder + classification head) work correctly.
import warnings as _warnings

with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")
    from momentfm import MOMENTPipeline


MomentSize = Literal["small", "base", "large"]

_HF_NAMES: dict[MomentSize, str] = {
    "small": "AutonLab/MOMENT-1-small",
    "base": "AutonLab/MOMENT-1-base",
    "large": "AutonLab/MOMENT-1-large",
}


@dataclass
class MomentConfig:
    size: MomentSize = "small"
    n_channels: int = 1
    num_class: int = 4
    freeze_encoder: bool = True


def build_moment_classifier(cfg: MomentConfig) -> MOMENTPipeline:
    """Load MOMENT and freeze encoder (Phase 1 default).

    The classification head is exposed as the only trainable params when
    ``freeze_encoder=True``.
    """
    name = _HF_NAMES[cfg.size]
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        model = MOMENTPipeline.from_pretrained(
            name,
            model_kwargs={
                "task_name": "classification",
                "n_channels": cfg.n_channels,
                "num_class": cfg.num_class,
            },
        )
        model.init()

    if cfg.freeze_encoder:
        # Freeze everything; un-freeze just the classification head below.
        for p in model.parameters():
            p.requires_grad_(False)
        head = _classification_head(model)
        for p in head.parameters():
            p.requires_grad_(True)
    return model


def _classification_head(model: MOMENTPipeline) -> nn.Module:
    """Return the classification head module of a MOMENTPipeline.

    momentfm names this attribute ``head`` — verified by inspection.
    """
    if not hasattr(model, "head"):
        raise RuntimeError("MOMENTPipeline has no 'head' attribute; momentfm API changed")
    return model.head


def trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
