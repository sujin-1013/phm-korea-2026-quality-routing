"""Rule-based router. No training — just two thresholds on a 0~1 noise level.

Plan / Notion §9: τ₁=0.33, τ₂=0.66.  noise<τ₁ → small, τ₁≤noise<τ₂ → base, else large.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RouterConfig:
    tau1: float = 0.33
    tau2: float = 0.66


class HardGateRouter:
    """Maps ``noise_level`` (B,) → tier index (B,) ∈ {0=small, 1=base, 2=large}."""

    SMALL: int = 0
    BASE: int = 1
    LARGE: int = 2

    def __init__(self, cfg: RouterConfig | None = None):
        self.cfg = cfg or RouterConfig()
        if not (0.0 < self.cfg.tau1 < self.cfg.tau2 < 1.0):
            raise ValueError(f"need 0 < tau1 < tau2 < 1, got {self.cfg}")

    def __call__(self, noise_level: torch.Tensor) -> torch.Tensor:
        tier = torch.full_like(noise_level, self.LARGE, dtype=torch.long)
        tier = torch.where(noise_level < self.cfg.tau2, torch.full_like(tier, self.BASE), tier)
        tier = torch.where(noise_level < self.cfg.tau1, torch.full_like(tier, self.SMALL), tier)
        return tier
