"""Phase 3 — full pipeline evaluation.

Combines:
  (1) trained NoiseEstimator (from exp01_estimator)
  (2) per-tier 4-class heads on frozen MOMENT-{S,B,L}
  (3) HardGateRouter

Reports the 5 paper metrics:
  - Per-SNR macro-F1 (×7 SNR levels)
  - Cost-accuracy Pareto curve (FLOPs vs F1)
  - Routing distribution (S/B/L share per SNR)
  - Estimator calibration (true vs pred noise level)
  - Per-class confusion matrix

Plus 6 baselines:
  S-only, B-only, L-only, random-router, oracle-router, STFT+ResNet18.

This file is a skeleton — full implementation lands after exp01 (estimator)
and per-tier head training are run on real Paderborn data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# TODO: import once the per-tier head training runner is finalised.
# from experiments.exp01_estimator.run import materialise_windows
# from src.phm_routing.training.train_head import train_per_tier_heads
# from src.phm_routing.training.train_estimator import train_estimator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--estimator-ckpt", type=Path)
    p.add_argument("--head-dir", type=Path)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    raise NotImplementedError(
        "exp02_routing.run is a skeleton — implement after exp01 and per-tier head training are wired."
    )


if __name__ == "__main__":
    main()
