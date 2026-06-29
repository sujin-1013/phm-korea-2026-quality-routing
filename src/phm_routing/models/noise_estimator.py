"""1D CNN noise estimator (~0.13K params).

Convention: the routing score is a **quality** score ``q ∈ [0, 1]`` where
``q = 1`` ≈ clean (high quality) and ``q = 0`` ≈ very noisy (low quality).

Implementation note: the network head is *trained* to regress the complementary
**noise level** ``n = 1 - q`` (target ``sigmoid(-snr_db / 10)``, so 0 ≈ clean,
1 ≈ very noisy). Quality is therefore obtained at the use-site as
``q = 1 - estimator(x)`` (see ``snr_to_quality``). Keeping the head in noise space
preserves the existing trained checkpoint; expressing routing in quality space is a
label change only — identical decisions, no retraining.

Routing reads naturally in quality space: HIGH q (clean) → cheap tier,
LOW q (noisy) → large/dense tier.
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


def snr_to_quality(snr_db: float | torch.Tensor) -> torch.Tensor:
    """Map SNR (dB) to the 0~1 **quality** score ``q = 1 - noise_level``.

    ``snr_db = inf`` → 1 (clean / high quality), ``snr_db = -inf`` → 0 (very noisy),
    monotonically *increasing*. Equivalent to ``sigmoid(snr_db / 10)``.

    The trained estimator outputs the noise level; recover quality with
    ``q = 1 - estimator(x)``.
    """
    return 1.0 - snr_to_target(snr_db)


def quality_from_estimator(noise_out: torch.Tensor) -> torch.Tensor:
    """Convert a trained-estimator output (noise level, 0 ≈ clean) to the quality
    score used for routing (``q = 1 - noise``, so 1 ≈ clean)."""
    return 1.0 - noise_out


class NoiseEstimator1D(nn.Module):
    """A tiny 1D CNN regressor.

    With the deployed default (channels=(7,), one stride-2 block) the param
    count is ~0.13K (134), matching the PHM Korea 2026 quality-estimator budget.
    Routing only needs a coarse, monotone SNR proxy, so a tiny head suffices.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (7,),
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
