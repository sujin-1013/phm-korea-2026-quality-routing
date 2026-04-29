"""1D CNN noise estimator (~100K params).

Trained to regress a 0~1 noise level from a raw vibration window.
Target is ``sigmoid_snr(snr_db) = sigmoid(-snr_db / 10)`` so 0 ≈ clean, 1 ≈ very noisy.
This places the rule-thresholds (τ₁=0.33, τ₂=0.66) at sensible SNR cuts.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def snr_to_target(snr_db: float | torch.Tensor) -> torch.Tensor:
    """Map SNR (dB) to the 0~1 target the estimator regresses.

    ``snr_db = inf`` → 0, ``snr_db = -inf`` → 1, monotonically decreasing.
    """
    if isinstance(snr_db, (int, float)):
        if math.isinf(snr_db):
            return torch.tensor(0.0 if snr_db > 0 else 1.0)
        snr_db = torch.tensor(float(snr_db))
    return torch.sigmoid(-snr_db / 10.0)


class NoiseEstimator1D(nn.Module):
    """A small 1D CNN regressor.

    With the defaults (channels=[16,32,64,128], 4 stride-2 blocks) the param
    count is ~95K, fitting the plan's "~100K" budget.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        kernel_size: int = 7,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c_prev = in_channels
        for c in channels:
            layers += [
                nn.Conv1d(c_prev, c, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
                nn.BatchNorm1d(c),
                nn.GELU(),
            ]
            c_prev = c
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(c_prev, c_prev),
            nn.GELU(),
            nn.Linear(c_prev, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.backbone(x)
        h = self.pool(h).squeeze(-1)  # (B, C)
        logit = self.head(h).squeeze(-1)  # (B,)
        return torch.sigmoid(logit)
