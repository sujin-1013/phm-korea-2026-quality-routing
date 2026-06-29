# Quality-Guided Dynamic Model Selection for Bearing Fault Diagnosis under Noisy Conditions

[![Venue](https://img.shields.io/badge/PHM%20Korea%202026-presented-00629B)]()
[![Dataset](https://img.shields.io/badge/Paderborn%20PU-PU--C1%20A2R-4b8bbe)](https://groups.uni-paderborn.de/kat/BearingDataCenter/)

A quality-guided dynamic routing method for bearing-fault diagnosis from noisy vibration
signals. A tiny SNR-supervised quality estimator scores each window, and a validation-calibrated
gate routes it to one of three 1D-CNN tiers (133K / 331K / 524K params), running a single
classifier per window. On the Paderborn PU benchmark under per-window mixed-SNR noise, the gate
matches the large model's macro-F1 within 0.3 pp while using 49% fewer average active parameters.

## Highlights

- **Quality estimator** — a tiny 1D-CNN supervised by injected SNR (no fault labels) emits a score `q ∈ [0,1]` (higher = cleaner), used only for routing.
- **Quality gate** — two validation-calibrated thresholds send high-quality windows to the small tier and low-quality windows to the large tier; exactly one model runs per window.
- **−49% active params** — 0.615 macro-F1 at 266K average active params vs 524K for the large tier (0.51× cost) at only −0.3 pp F1.
- **In-distribution scope** — larger tiers stay more robust as SNR drops; cross-domain transfer (A2R / bearing-disjoint) is left as future work.

## Results

Paderborn PU, trained on clean signals only, tested under per-window mixed SNR (clean → −10 dB); macro-F1 and average active parameters.

| Method | Macro-F1 | Active params | Param ratio (Large=1) |
|:---|---:|---:|---:|
| Small | 0.553 | 133K | 0.25 |
| Medium | 0.611 | 331K | 0.63 |
| Large | **0.618** | 524K | 1.00 |
| **Quality gate** | **0.615** | **266K** | **0.51** |

The claim is cost efficiency, not accuracy gain: the gate matches the large tier (−0.3 pp) at roughly half the average compute. Numbers follow the PHM Korea 2026 presentation (2026-06-26).

## Installation & Usage

```bash
pip install -e ".[dev]"
bash scripts/download_paderborn.sh

# Reproduce the final routing result
python experiments/quality_routing/run.py --seed 42
python -m pytest -q
```

Protocol: Paderborn `vibration_1` only, 3 classes (Normal / inner-race / outer-race), 2048-sample
non-overlapping windows (32 ms at 64 kHz). Classifiers train on clean signals; evaluation injects
per-window AWGN from the SNR pool `{clean, 20, 10, 5, 0, -5, -10}` dB, with BatchNorm fixed in
`.eval()` mode. The quality estimator (≈0.13K params) is self-supervised on injected SNR. Add `--retrain` /
`--retrain-estimator` to regenerate checkpoints; `--scenario {indist,bdis,loco,a2r}` selects the split.

## Repository structure

```
phm-korea-2026-quality-routing/
├── src/phm_routing/
│   ├── data/                 # Paderborn loading, windowing, AWGN injection, PU-C1 split
│   ├── models/               # 1D-CNN capacity ladder + quality (noise) estimator
│   ├── training/             # CNN and estimator training loops
│   ├── eval/                 # mixed-SNR routing selection + metrics
│   └── utils/                # seeding, metric helpers
├── experiments/
│   └── quality_routing/
│       └── run.py            # canonical end-to-end routing experiment
├── scripts/
│   └── download_paderborn.sh # fetch the Paderborn PU dataset
├── tests/                    # PU-C1 split + routing unit tests
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Citation

Method presented at PHM Korea 2026; proceedings in preparation. Please cite the presentation and
the Paderborn benchmark dataset:

```bibtex
@misc{Choi2026QualityRouting,
  title  = {Quality-Guided Dynamic Model Selection for Bearing Fault
            Diagnosis under Noisy Conditions},
  author = {Choi, Sujin},
  note   = {PHM Korea 2026, Korean Society for Prognostics and Health
            Management, Busan, Korea},
  year   = {2026}
}

@inproceedings{Lessmeier2016,
  title     = {Condition Monitoring of Bearing Damage in Electromechanical Drive
               Systems by Using Motor Current Signals of Electric Motors: A
               Benchmark Data Set for Data-Driven Classification},
  author    = {Lessmeier, Christian and Kimotho, James Kuria and Zimmer, Detmar
               and Sextro, Walter},
  booktitle = {PHM Society European Conference},
  volume    = {3},
  number    = {1},
  year      = {2016},
  doi       = {10.36001/phme.2016.v3i1.1577}
}
```

## License & contact

No license file is included yet; the method paper is in preparation, so please contact the
author before reuse. The Paderborn PU dataset is distributed by Paderborn University under its
own terms and is **not redistributed** in this repository — download it via the script above.

Sujin Choi — KETI, Next-Generation Power System Research Center · [@sujin-1013](https://github.com/sujin-1013)
