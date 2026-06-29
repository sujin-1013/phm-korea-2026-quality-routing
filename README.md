# PHM Korea 2026 Quality Routing

[![Venue](https://img.shields.io/badge/PHM%20Korea%202026-in%20preparation-orange)]()
[![Dataset](https://img.shields.io/badge/Paderborn%20PU-PU--C1%20A2R-4b8bbe)](https://groups.uni-paderborn.de/kat/BearingDataCenter/)

This repository contains the cleaned PHM Korea 2026 quality-routing method for
bearing-fault diagnosis. The canonical method is a compact 1D-CNN capacity
ladder with validation-selected quality routing over:

- `base`: CNN(32, 4), about 133K parameters
- `x300`: CNN(32, 5), about 331K parameters
- `large`: CNN(64, 5), about 524K parameters

The always-on noise estimator emits a monotone noise score. The final routing
rule chooses thresholds on validation data and uses the cheapest operating point
whose validation macro-F1 is within 0.01 of the large model.

## Final Result

Mixed-SNR test, repeated measurement summary:

| Method | Macro-F1 | Avg active params | vs large |
|:---|---:|---:|---:|
| single x300 | 0.611 | 331K | 1.58x cheaper |
| single large | 0.624 | 524K | 1.00x |
| quality gate over {133K, 331K, 524K} | 0.619 | 246K | 2.13x cheaper |

The honest claim is cost efficiency, not accuracy gain. The routed model does
not beat the single large model in F1, and A2R / bearing-disjoint evaluation
remains near chance because the artificial-to-real damage domain gap dominates.

## Dataset

[Paderborn University bearing dataset](https://groups.uni-paderborn.de/kat/BearingDataCenter/)
(Lessmeier et al., PHME 2016), using only `vibration_1`:

- 3 classes: Normal / inner-race fault / outer-race fault
- 2048-sample non-overlapping windows, 32 ms at 64 kHz
- AWGN injected per window from the SNR pool `{clean, 20, 10, 5, 0, -5, -10}` dB
- BatchNorm is fixed in `.eval()` mode for reported evaluation

## Installation

```bash
pip install -e ".[dev]"
bash scripts/download_paderborn.sh
```

## Reproduce The Final Path

```bash
python experiments/quality_routing/run.py --seed 42
python -m pytest -q
```

Add `--retrain` and `--retrain-estimator` only when regenerating checkpoints
from raw data.

## Repository Structure

```text
phm-korea-2026-quality-routing/
├── src/phm_routing/
│   ├── data/                  # Paderborn loading, windowing, AWGN, PU-C1 split
│   ├── models/                # final CNN ladder + noise estimator
│   ├── training/              # CNN and estimator training helpers
│   ├── eval/                  # mixed-SNR routing selection and metrics
│   └── utils/
├── experiments/
│   └── quality_routing/       # canonical final experiment entrypoint
└── tests/
```
