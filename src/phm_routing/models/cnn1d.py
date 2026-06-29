"""1D-CNN capacity ladder used by the final PHM Korea 2026 experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


ModelSize = Literal["small", "medium", "large"]


@dataclass(frozen=True)
class CNNSizeSpec:
    base_channels: int
    n_blocks: int


# The three tiers presented at PHM Korea 2026: Small 133K / Medium 331K / Large 524K.
CAPACITY_SIZES: dict[ModelSize, CNNSizeSpec] = {
    "small": CNNSizeSpec(32, 4),   # ~133K params
    "medium": CNNSizeSpec(32, 5),  # ~331K params
    "large": CNNSizeSpec(64, 5),   # ~524K params
}

FINAL_CAPACITY_LADDER: tuple[ModelSize, ModelSize, ModelSize] = ("small", "medium", "large")


class CNN1D(nn.Module):
    """Compact 1D-CNN for 2048-sample vibration windows.

    The model accepts a batch shaped ``(batch, time)`` and returns class logits.
    This is the architecture used by the quality-routing capacity ladder.
    """

    def __init__(self, base_channels: int, n_blocks: int, num_class: int = 3):
        super().__init__()
        if base_channels <= 0:
            raise ValueError(f"base_channels must be > 0, got {base_channels}")
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be > 0, got {n_blocks}")

        layers: list[nn.Module] = [
            nn.Conv1d(1, base_channels, 64, stride=16, padding=24),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ]
        channels = base_channels
        for _ in range(n_blocks - 1):
            out_channels = min(channels * 2, 256)
            layers += [
                nn.Conv1d(channels, out_channels, 3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.1), nn.Linear(channels, num_class))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"CNN1D expects (batch, time), got {tuple(x.shape)}")
        return self.head(self.pool(self.features(x.unsqueeze(1))))


def build_cnn(size: ModelSize, num_class: int = 3) -> CNN1D:
    try:
        spec = CAPACITY_SIZES[size]
    except KeyError as exc:
        known = ", ".join(CAPACITY_SIZES)
        raise ValueError(f"unknown CNN size {size!r}; expected one of: {known}") from exc
    return CNN1D(spec.base_channels, spec.n_blocks, num_class=num_class)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
