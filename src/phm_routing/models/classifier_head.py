"""Per-tier classification head used on top of frozen MOMENT.

momentfm's MOMENTPipeline already exposes a ``.head`` for classification when
``task_name='classification'``. This module provides a standalone head we can
attach on top of pre-extracted MOMENT embeddings if we ever want to bypass the
full pipeline (faster batched eval, embedding caching).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearProbeHead(nn.Module):
    """Two-layer MLP on top of fixed-dim embeddings."""

    def __init__(self, emb_dim: int, num_class: int = 4, hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        h = hidden or emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, num_class),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)
