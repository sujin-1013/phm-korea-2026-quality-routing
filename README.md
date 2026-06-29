# Capacity-Ladder Quality Routing for Bearing Fault Diagnosis under Mixed-SNR Noise

[![Venue](https://img.shields.io/badge/PHM%20Korea%202026-in%20preparation-orange)]()
[![Dataset](https://img.shields.io/badge/Paderborn%20PU-PU--C1%20A2R-4b8bbe)](https://groups.uni-paderborn.de/kat/BearingDataCenter/)

A compact 1D-CNN capacity ladder for vibration bearing-fault diagnosis that routes each
input to the cheapest model whose validation macro-F1 stays within a margin of the largest
model. An always-on noise estimator emits a monotone noise score, and routing thresholds are
selected on validation data — so the gate trades compute for accuracy without ever beating the
large model, delivering 2.13× lower average cost at comparable macro-F1.

## Highlights

- **Capacity ladder** — three 1D-CNNs (`base` ≈133K, `x300` ≈331K, `large` ≈524K params) form a cost/accuracy ladder.
- **Quality routing** — an always-on noise estimator produces a monotone score; validation-selected thresholds pick the cheapest tier within 0.01 macro-F1 of `large`.
- **2.13× cheaper** — the gate matches single-`large` accuracy at 246K average active params instead of 524K.
- **Honest scope** — the claim is cost efficiency, not accuracy gain; artificial-to-real (A2R) and bearing-disjoint evaluation remain near chance because the damage-domain gap dominates.

## Results

Mixed-SNR test, repeated-measurement summary on the Paderborn PU-C1 protocol (macro-F1):

| Method | Macro-F1 | Avg active params | vs large |
|:---|---:|---:|---:|
| single x300 | 0.611 | 331K | 1.58× cheaper |
| **single large** | **0.624** | 524K | 1.00× |
| **quality gate over {133K, 331K, 524K}** | 0.619 | **246K** | **2.13× cheaper** |

The routed model does not beat single-`large` in F1; the value is matching it at less than half the average compute.

## Installation & Usage

```bash
pip install -e ".[dev]"
bash scripts/download_paderborn.sh

# Reproduce the final routing result
python experiments/quality_routing/run.py --seed 42
python -m pytest -q
```

Data protocol: Paderborn `vibration_1` only, 3 classes (Normal / inner-race / outer-race),
2048-sample non-overlapping windows (32 ms at 64 kHz), AWGN injected per window from the SNR
pool `{clean, 20, 10, 5, 0, -5, -10}` dB, with BatchNorm fixed in `.eval()` mode for reported
evaluation. Add `--retrain` / `--retrain-estimator` only when regenerating checkpoints from raw
data; `--scenario {indist,bdis,loco,a2r}` selects the evaluation split.

## Repository structure

```
phm-korea-2026-quality-routing/
├── src/phm_routing/
│   ├── data/                 # Paderborn loading, windowing, AWGN injection, PU-C1 split
│   ├── models/               # 1D-CNN capacity ladder + noise estimator
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

Method paper in preparation (PHM Korea 2026). Please cite the Paderborn benchmark dataset:

```bibtex
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

Sujin Choi — KETI · [@sujin-1013](https://github.com/sujin-1013)
