"""Synthetic noise injection at controlled SNR.

Used to (a) generate training data with mixed SNR for the noise estimator,
(b) evaluate models across the SNR sweep {inf, 20, 10, 5, 0, -5, -10} dB.

Reference: Donoho, D. L. (1995). De-noising by soft-thresholding.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def _to_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()


def signal_power(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """E[x^2] along ``dim`` (assumes zero-mean signals; we don't subtract)."""
    return x.pow(2).mean(dim=dim, keepdim=True)


def add_awgn(x: torch.Tensor, snr_db: float, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Add additive white Gaussian noise to reach the target SNR (in dB).

    SNR is computed per-sample along the last dim, so each row gets its own
    noise scale. ``snr_db = float('inf')`` returns ``x`` unchanged.
    """
    x = _to_tensor(x)
    if not np.isfinite(snr_db):
        return x

    sig_pow = signal_power(x)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_pow = sig_pow / snr_linear
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    noise = noise * noise_pow.sqrt()
    return x + noise


def add_pink_noise(
    x: torch.Tensor, snr_db: float, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Add 1/f (pink) noise scaled to ``snr_db``. Used for ablation.

    Implementation: filter white noise by H(f) = 1/sqrt(f) in the frequency domain.
    """
    x = _to_tensor(x)
    if not np.isfinite(snr_db):
        return x

    *batch_dims, length = x.shape
    flat = x.reshape(-1, length)
    out = torch.empty_like(flat)
    sig_pow = signal_power(flat)

    snr_linear = 10.0 ** (snr_db / 10.0)
    target_pow = sig_pow / snr_linear

    freqs = torch.fft.rfftfreq(length, device=x.device)
    scale = torch.where(freqs > 0, 1.0 / freqs.sqrt(), torch.zeros_like(freqs))

    for i in range(flat.shape[0]):
        white = torch.randn(length, generator=generator, device=x.device, dtype=x.dtype)
        spec = torch.fft.rfft(white) * scale
        pink = torch.fft.irfft(spec, n=length)
        pink = pink - pink.mean()
        cur_pow = pink.pow(2).mean()
        if cur_pow > 0:
            pink = pink * (target_pow[i].sqrt() / cur_pow.sqrt())
        out[i] = flat[i] + pink
    return out.reshape(*batch_dims, length)


def snr_sweep_levels() -> list[float]:
    """Default SNR levels used in the paper (per Notion §10-2, plan §1.1).

    Extended (2026-05-06) to include -20 dB for extreme low-SNR robustness analysis.
    """
    return [float("inf"), 20.0, 10.0, 5.0, 0.0, -5.0, -10.0, -20.0]


def random_snr_sample(
    rng: np.random.Generator,
    *,
    levels: Iterable[float] = (float("inf"), 20.0, 10.0, 5.0, 0.0, -5.0, -10.0, -20.0),
) -> float:
    """Pick a single SNR level uniformly. Used for mixed-SNR training."""
    arr = list(levels)
    return float(arr[rng.integers(0, len(arr))])
