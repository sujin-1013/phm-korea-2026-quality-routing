"""Sliding-window utilities for raw vibration time series."""
from __future__ import annotations

import numpy as np


def sliding_windows(x: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Return shape (N, window_size) windows from a 1-D series ``x``.

    Drops trailing samples that don't fill a full window.
    """
    if x.ndim != 1:
        raise ValueError(f"expected 1-D array, got shape {x.shape}")
    if window_size <= 0 or stride <= 0:
        raise ValueError(f"window_size={window_size} stride={stride} must be > 0")

    n = (len(x) - window_size) // stride + 1
    if n <= 0:
        return np.empty((0, window_size), dtype=x.dtype)

    shape = (n, window_size)
    strides = (x.strides[0] * stride, x.strides[0])
    out = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return np.ascontiguousarray(out)


def downsample(x: np.ndarray, factor: int) -> np.ndarray:
    """Decimate by integer factor with anti-aliasing (scipy.signal.decimate)."""
    if factor == 1:
        return x
    from scipy.signal import decimate

    return decimate(x, factor, ftype="iir", zero_phase=True).astype(x.dtype)
