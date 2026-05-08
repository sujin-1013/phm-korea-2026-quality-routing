"""Routing modules for SNRGate.

Two variants live here:
  * ``HardGateRouter`` — fixed-rule baseline (Notion §9, τ₁=0.33, τ₂=0.66).
  * ``LearnableSNRGate`` — paper main method. Learnable per-tier centers μ_k and a
    temperature T. Forward returns soft mixture weights so estimator·gate·heads
    co-train under classification loss; at inference T→0 collapses to a single
    tier (argmin distance) so the cost-saving premise holds.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


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


@dataclass
class GateConfig:
    # Wider spread than (0.17, 0.50, 0.83) to give more room before centers can cross.
    # Combined with the soft monotonicity penalty in train_routing, this prevents the
    # binary-collapse observed in the 2026-05-07 run.
    init_centers: tuple[float, float, float] = (0.10, 0.50, 0.90)
    T_init: float = 1.0
    T_min: float = 0.05
    # Number of epochs over which T is exponentially annealed from T_init→T_min.
    anneal_epochs: int = 30


class LearnableSNRGate(nn.Module):
    """Soft-to-hard SNR gate.

    During training, ``forward`` returns soft mixture weights p_k of shape
    ``(B, 3)``. Logits are negative |noise − μ_k| / T, which gives:
      * T=T_init  → diffuse mixture (gradient flows through every tier)
      * T→0       → near-one-hot (single tier dominates).

    At inference call ``hard_tier`` to get a hard ``argmin`` tier index — that
    preserves the per-input single-tier cost-saving premise.
    """

    SMALL: int = 0
    BASE: int = 1
    LARGE: int = 2
    NUM_TIERS: int = 3

    def __init__(self, cfg: GateConfig | None = None):
        super().__init__()
        self.cfg = cfg or GateConfig()
        c = self.cfg.init_centers
        if len(c) != self.NUM_TIERS:
            raise ValueError(f"init_centers must have {self.NUM_TIERS} entries, got {c}")
        if not all(0.0 <= v <= 1.0 for v in c):
            raise ValueError(f"init_centers must lie in [0,1], got {c}")
        self.centers = nn.Parameter(torch.tensor(c, dtype=torch.float32))
        # T is a buffer (not learned) — controlled by the training loop schedule.
        self.register_buffer("T", torch.tensor(float(self.cfg.T_init)))

    def set_temperature(self, T: float) -> None:
        self.T.fill_(float(T))

    @torch.no_grad()
    def clamp_centers(self) -> None:
        """Project centers back into [0,1] after a gradient step.

        Required because ``noise_level`` is bounded in [0,1] but ``centers`` is
        an unconstrained ``nn.Parameter``; without clamping it can drift out of
        range during training and corrupt the soft-mixture distribution.
        Call this once per optimiser step (or per epoch as a coarser fallback).
        """
        self.centers.data.clamp_(0.0, 1.0)

    def anneal_step(self, epoch: int) -> float:
        """Exponential decay from ``T_init`` to ``T_min`` over ``anneal_epochs``."""
        if self.cfg.anneal_epochs <= 0:
            T = self.cfg.T_min
        else:
            ratio = min(1.0, epoch / self.cfg.anneal_epochs)
            log_t = (1.0 - ratio) * torch.log(torch.tensor(self.cfg.T_init)) + ratio * torch.log(
                torch.tensor(self.cfg.T_min)
            )
            T = float(torch.exp(log_t))
        self.set_temperature(T)
        return T

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        """Soft mixture weights ``p_k`` of shape ``(B, 3)``."""
        if noise_level.dim() != 1:
            raise ValueError(f"noise_level must be (B,), got {tuple(noise_level.shape)}")
        d = (noise_level.unsqueeze(-1) - self.centers.unsqueeze(0)).abs()  # (B, 3)
        T = self.T.clamp_min(1e-6)
        return (-d / T).softmax(dim=-1)

    @torch.no_grad()
    def hard_tier(self, noise_level: torch.Tensor) -> torch.Tensor:
        """Inference path: pick argmin-distance tier (one tier active, no soft mixture)."""
        d = (noise_level.unsqueeze(-1) - self.centers.unsqueeze(0)).abs()
        return d.argmin(dim=-1)
