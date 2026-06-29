from __future__ import annotations

import numpy as np
import pytest
import torch

from phm_routing.eval.mixed_snr import (
    active_cost_k,
    add_mixed_awgn,
    route_by_thresholds,
    routed_predictions,
    select_capacity_routes,
)
from phm_routing.models.cnn1d import CAPACITY_SIZES, FINAL_CAPACITY_LADDER, build_cnn, count_parameters
from phm_routing.models.noise_estimator import DEPLOYED_ESTIMATOR_PARAM_COUNT, NoiseEstimator1D, quality_from_estimator
from phm_routing.training.train_cnn import cap_arrays


def test_quality_routing_ladder_sizes_are_pinned():
    assert FINAL_CAPACITY_LADDER == ("small", "medium", "large")
    assert CAPACITY_SIZES["small"].base_channels == 32
    assert CAPACITY_SIZES["small"].n_blocks == 4
    assert CAPACITY_SIZES["medium"].base_channels == 32
    assert CAPACITY_SIZES["medium"].n_blocks == 5
    assert CAPACITY_SIZES["large"].base_channels == 64
    assert CAPACITY_SIZES["large"].n_blocks == 5


def test_cnn1d_forward_shape_and_parameter_order():
    x = torch.randn(4, 2048)
    small = build_cnn("small")
    medium = build_cnn("medium")
    large = build_cnn("large")
    assert small(x).shape == (4, 3)
    assert count_parameters(small) < count_parameters(medium) < count_parameters(large)


def test_cnn1d_rejects_non_window_batches():
    with pytest.raises(ValueError):
        build_cnn("small")(torch.randn(4, 1, 2048))


def test_route_by_thresholds_sends_high_quality_to_small_tier():
    score = np.array([0.95, 0.80, 0.50, 0.20, 0.01])
    assert route_by_thresholds(score, (0.33, 0.66)).tolist() == [0, 0, 1, 2, 2]


def test_mixed_noise_is_seed_deterministic():
    X = np.ones((12, 32), dtype=np.float32)
    a, snr_a = add_mixed_awgn(X, seed=7, levels=(0.0, float("inf")))
    b, snr_b = add_mixed_awgn(X, seed=7, levels=(0.0, float("inf")))
    c, snr_c = add_mixed_awgn(X, seed=8, levels=(0.0, float("inf")))
    assert np.array_equal(snr_a, snr_b)
    assert torch.allclose(a, b)
    assert not np.array_equal(snr_a, snr_c)
    assert not torch.allclose(a, c)


def test_quality_from_estimator_is_complement():
    noise = torch.tensor([0.0, 0.25, 1.0])
    assert torch.allclose(quality_from_estimator(noise), torch.tensor([1.0, 0.75, 0.0]))


def test_default_quality_estimator_matches_deployed_parameter_count():
    model = NoiseEstimator1D()
    assert sum(p.numel() for p in model.parameters()) == DEPLOYED_ESTIMATOR_PARAM_COUNT


def test_cap_arrays_is_deterministic_and_preserves_pairs():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    Xa, ya = cap_arrays(X, y, 4, 123)
    Xb, yb = cap_arrays(X, y, 4, 123)
    assert np.array_equal(Xa, Xb)
    assert np.array_equal(ya, yb)
    assert np.array_equal(Xa[:, 0] // 2, ya)


def test_routed_predictions_and_cost():
    preds = {
        "small": np.array([0, 0, 0, 0]),
        "medium": np.array([1, 1, 1, 1]),
        "large": np.array([2, 2, 2, 2]),
    }
    idx = np.array([0, 1, 2, 1])
    out = routed_predictions(preds, ("small", "medium", "large"), idx)
    assert out.tolist() == [0, 1, 2, 1]
    cost = active_cost_k(idx, ("small", "medium", "large"), {"small": 100_000, "medium": 300_000, "large": 500_000}, 0.13)
    assert cost == pytest.approx(300.13)


def test_select_capacity_routes_picks_iso_under_val_margin():
    y = np.array([0, 1, 2, 2])
    val_predictions = {
        "small": np.array([0, 0, 0, 0]),
        "medium": np.array([0, 1, 1, 1]),
        "large": np.array([0, 1, 2, 2]),
    }
    test_predictions = val_predictions
    score = np.array([0.90, 0.50, 0.20, 0.10])
    results = select_capacity_routes(
        y_val=y,
        val_predictions=val_predictions,
        val_score=score,
        y_test=y,
        test_predictions=test_predictions,
        test_score=score,
        tiers=("small", "medium", "large"),
        parameter_counts={"small": 100_000, "medium": 300_000, "large": 500_000},
        estimator_overhead_k=0.0,
        threshold_grid=(0.33, 0.66),
    )
    assert set(results) == {"iso", "accmatch", "fc"}
    assert results["iso"].test_f1 == pytest.approx(1.0)
    assert results["iso"].shares == (0.25, 0.25, 0.5)
