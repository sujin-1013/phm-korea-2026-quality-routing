"""Locked PU-C1 A2R 3-class protocol constants and materialisation wrapper.

This thin wrapper keeps the split shape explicit for quality-routing runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from phm_routing.data.paderborn import materialise_pu_c1

WINDOW = 2048
STRIDE = 2048
NUM_CLASS = 3  # N / IR / OR


def materialise(
    data_root: Path,
    *,
    window: int = WINDOW,
    stride: int = STRIDE,
    max_files_per_bearing: int = 8,
    seed: int = 42,
) -> tuple[
    np.ndarray, np.ndarray,  # train
    np.ndarray, np.ndarray,  # val (within-train; early stopping / best-on-val)
    np.ndarray, np.ndarray,  # test (real damage; paper SNR sweep)
]:
    """PU-C1 A2R 3-class materialisation. Thin wrapper over ``materialise_pu_c1``."""
    return materialise_pu_c1(
        data_root,
        window=window,
        stride=stride,
        max_files_per_bearing=max_files_per_bearing,
        seed=seed,
    )
