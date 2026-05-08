"""Shared SNR sweep for routed classifiers (HardGate / SNRGate).

Each script supplies its own `inference_fn(xb) -> (logits, tier_idx)`; the
sweep handles AWGN injection, batching, Macro-F1, and S/B/L routing shares.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np
import torch

from phm_routing.data.noise_injection import add_awgn
from phm_routing.utils.metrics import macro_f1

InferenceFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def routing_snr_sweep(
    inference_fn: InferenceFn,
    X: np.ndarray,
    y: np.ndarray,
    snr_levels: list[float],
    batch_size: int,
    device: str,
    *,
    aug_seed: int = 123,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Run SNR sweep for a routed classifier.

    Returns ({snr_label: macro_f1}, {snr_label: {"S": p, "B": p, "L": p}}).
    """
    aug_gen = torch.Generator(device="cpu").manual_seed(aug_seed)
    Xv = torch.from_numpy(X).float().unsqueeze(1)
    yv = torch.from_numpy(y).long()
    f1: dict[str, float] = {}
    routing: dict[str, dict[str, float]] = {}
    for snr in snr_levels:
        preds, tiers_used = [], []
        for i in range(0, Xv.shape[0], batch_size):
            xb = Xv[i : i + batch_size].clone()
            if np.isfinite(snr):
                xb = add_awgn(xb, snr_db=float(snr), generator=aug_gen)
            xb = xb.to(device)
            logits, tier_idx = inference_fn(xb)
            preds.append(logits.argmax(-1).cpu().numpy())
            tiers_used.append(tier_idx.cpu().numpy())
        y_pred = np.concatenate(preds)
        tier_arr = np.concatenate(tiers_used)
        label = "inf" if not np.isfinite(snr) else f"{snr:g}"
        f1[label] = macro_f1(yv.numpy(), y_pred)
        c = Counter(tier_arr.tolist())
        total = max(sum(c.values()), 1)
        routing[label] = {
            "S": c.get(0, 0) / total,
            "B": c.get(1, 0) / total,
            "L": c.get(2, 0) / total,
        }
    return f1, routing
