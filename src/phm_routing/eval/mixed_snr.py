"""Mixed-SNR evaluation and capacity-gate threshold selection."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from phm_routing.data.noise_injection import add_awgn
from phm_routing.utils.metrics import macro_f1


MIXED_SNR_LEVELS: tuple[float, ...] = (20.0, 10.0, 5.0, 0.0, -5.0, -10.0, float("inf"))
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = tuple(round(float(x), 2) for x in np.arange(0.05, 0.91, 0.05))


@dataclass(frozen=True)
class RoutingResult:
    name: str
    val_f1: float
    test_f1: float
    val_cost_k: float
    test_cost_k: float
    shares: tuple[float, ...]
    thresholds: tuple[float, ...]


def add_mixed_awgn(
    X: np.ndarray | torch.Tensor,
    *,
    seed: int,
    levels: Sequence[float] = MIXED_SNR_LEVELS,
    batch_size: int = 512,
) -> tuple[torch.Tensor, np.ndarray]:
    """Apply per-window random SNR from ``levels`` with deterministic assignment."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    Xn = torch.as_tensor(X, dtype=torch.float32).clone()
    assignment = rng.integers(0, len(levels), Xn.shape[0])
    assigned_snr = np.asarray([levels[i] for i in assignment], dtype=np.float64)

    for level_idx, snr in enumerate(levels):
        if not np.isfinite(snr):
            continue
        indices = np.where(assignment == level_idx)[0]
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            Xn[batch_indices] = add_awgn(Xn[batch_indices], snr_db=float(snr), generator=gen)
    return Xn, assigned_snr


def route_by_thresholds(score: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Map a monotone noise/quality score to tier index via ascending thresholds."""
    score = np.asarray(score)
    out = np.zeros(score.shape[0], dtype=np.int64)
    for threshold in thresholds:
        out += (score >= threshold).astype(np.int64)
    return out


@torch.no_grad()
def predict_all_models(
    models: Mapping[str, nn.Module],
    X: torch.Tensor,
    *,
    device: str,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.eval()
        preds: list[np.ndarray] = []
        for start in range(0, X.shape[0], batch_size):
            xb = X[start : start + batch_size].to(device)
            preds.append(model(xb).argmax(-1).cpu().numpy())
        out[name] = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
    return out


@torch.no_grad()
def estimate_noise_score(
    estimator: nn.Module,
    X: torch.Tensor,
    *,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Return the estimator's monotone noise score.

    The saved final estimator emits noise level directly: low for clean windows,
    high for noisy windows. Threshold routing therefore sends larger scores to
    larger tiers.
    """
    estimator.eval()
    scores: list[np.ndarray] = []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start : start + batch_size].to(device).unsqueeze(1)
        scores.append(estimator(xb).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float64)


def select_capacity_routes(
    *,
    y_val: np.ndarray,
    val_predictions: Mapping[str, np.ndarray],
    val_score: np.ndarray,
    y_test: np.ndarray,
    test_predictions: Mapping[str, np.ndarray],
    test_score: np.ndarray,
    tiers: Sequence[str],
    parameter_counts: Mapping[str, int],
    estimator_overhead_k: float = 7.6,
    reference_tier: str = "large",
    val_margin: float = 0.01,
    target_cost_k: float = 250.0,
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
) -> dict[str, RoutingResult]:
    """Pick iso, accmatch, and fixed-cost operating points for a tier ladder."""
    tiers = tuple(tiers)
    if len(tiers) < 2:
        raise ValueError("tiers must include at least two models")
    if reference_tier not in val_predictions:
        raise KeyError(f"reference tier {reference_tier!r} missing from predictions")

    combos = list(combinations(tuple(float(v) for v in threshold_grid), len(tiers) - 1))
    if not combos:
        raise ValueError("threshold_grid did not produce any threshold combinations")

    val_large = macro_f1(y_val, val_predictions[reference_tier])
    front: list[RoutingResult] = []
    for thresholds in combos:
        idx_val = route_by_thresholds(val_score, thresholds)
        idx_test = route_by_thresholds(test_score, thresholds)
        pred_val = routed_predictions(val_predictions, tiers, idx_val)
        pred_test = routed_predictions(test_predictions, tiers, idx_test)
        val_cost = active_cost_k(idx_val, tiers, parameter_counts, estimator_overhead_k)
        test_cost = active_cost_k(idx_test, tiers, parameter_counts, estimator_overhead_k)
        shares = tuple(round(float((idx_test == k).mean()), 4) for k in range(len(tiers)))
        front.append(
            RoutingResult(
                name="candidate",
                val_f1=macro_f1(y_val, pred_val),
                test_f1=macro_f1(y_test, pred_test),
                val_cost_k=val_cost,
                test_cost_k=test_cost,
                shares=shares,
                thresholds=tuple(thresholds),
            )
        )

    ok = [r for r in front if r.val_f1 >= val_large - val_margin]
    picks = {
        "iso": min(ok, key=lambda r: r.val_cost_k) if ok else max(front, key=lambda r: r.val_f1),
        "accmatch": min(front, key=lambda r: abs(r.val_f1 - val_large)),
        "fc": min(front, key=lambda r: abs(r.val_cost_k - target_cost_k)),
    }
    return {name: _rename_result(result, name) for name, result in picks.items()}


def routed_predictions(predictions: Mapping[str, np.ndarray], tiers: Sequence[str], tier_idx: np.ndarray) -> np.ndarray:
    stacked = np.column_stack([predictions[tier] for tier in tiers])
    return stacked[np.arange(len(tier_idx)), tier_idx]


def active_cost_k(
    tier_idx: np.ndarray,
    tiers: Sequence[str],
    parameter_counts: Mapping[str, int],
    estimator_overhead_k: float,
) -> float:
    shares = [(tier_idx == k).mean() for k in range(len(tiers))]
    return float(sum(shares[k] * parameter_counts[tiers[k]] for k in range(len(tiers))) / 1e3 + estimator_overhead_k)


def _rename_result(result: RoutingResult, name: str) -> RoutingResult:
    return RoutingResult(
        name=name,
        val_f1=result.val_f1,
        test_f1=result.test_f1,
        val_cost_k=result.val_cost_k,
        test_cost_k=result.test_cost_k,
        shares=result.shares,
        thresholds=result.thresholds,
    )
