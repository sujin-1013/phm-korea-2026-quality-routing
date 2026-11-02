# SNRGate: Capacity-Ladder Quality Routing for Bearing Fault Diagnosis

[![Venue](https://img.shields.io/badge/PHM%20Korea%202026-in%20preparation-orange)]()
[![Dataset](https://img.shields.io/badge/Paderborn%20PU-PU--C1%20A2R-4b8bbe)](https://groups.uni-paderborn.de/kat/BearingDataCenter/)

This repository contains the PHM Korea 2026 bearing-fault diagnosis experiments
after the final-method cleanup. The current canonical method is a compact
1D-CNN capacity ladder with quality-gated routing over:

- `base`: CNN(32, 4), about 133K parameters
- `x300`: CNN(32, 5), about 331K parameters
- `large`: CNN(64, 5), about 524K parameters

The always-on quality estimator emits `q ∈ [0, 1]`, where higher values mean a
cleaner signal. The final routing rule calibrates two thresholds on validation
data: high-quality windows use the small tier, low-quality windows use the
large tier, and exactly one classifier runs per window.

## Final Result

Mixed-SNR test, repeated measurement summary:

| Method | Macro-F1 | Avg active params | Param ratio (Large=1) |
|:---|---:|---:|---:|
| Small (code: `base`) | 0.553 | 133K | 0.25 |
| Medium (code: `x300`) | 0.611 | 331K | 0.63 |
| Large (code: `large`) | 0.618 | 524K | 1.00 |
| **Quality gate** | **0.615** | **266K** | **0.51** |

The honest claim is cost efficiency, not accuracy gain. The routed model does
not beat the single large model in F1; it stays within -0.3 pp while using 49%
fewer active parameters and 20% fewer active parameters than the medium tier.
The table follows the PHM Korea 2026 presentation and is traced in
`experiments/exp_scenario/presentation_summary.csv`.

## Dataset

[Paderborn University bearing dataset](https://groups.uni-paderborn.de/kat/BearingDataCenter/)
(Lessmeier et al., PHME 2016), using only `vibration_1`:

- 3 classes: Normal / inner-race fault / outer-race fault
- 2048-sample non-overlapping windows, 32 ms at 64 kHz
- AWGN injected per window from the SNR pool `{clean, 20, 10, 5, 0, -5, -10}` dB
- BatchNorm is fixed in `.eval()` mode for reported evaluation
- Quality estimator: 131 parameters (`channels=(2, 2, 2, 2)`), supervised
  by injected SNR without fault labels

## Installation

```bash
pip install -e ".[dev]"
bash scripts/download_paderborn.sh
```

Legacy MOMENT/LoRA reproduction code is no longer the default path. Install it
only when auditing the retired pipeline:

```bash
pip install -e ".[legacy-moment,dev]"
```

## Reproduce The Final Path

```bash
python experiments/quality_routing/run.py --seed 42
python experiments/exp_scenario/fig_scripts/make_pareto_params_capacity.py
python -m pytest -q
```

The old `experiments/exp_scenario/fig_scripts/exp_4tier.py` path remains as a
compatibility wrapper for older notes. Add `--retrain` and
`--retrain-estimator` only when regenerating checkpoints from raw data.

## Repository Structure

```text
phm-korea-2026-snrgate/
├── src/phm_routing/
│   ├── data/                  # Paderborn loading, windowing, AWGN, PU-C1 split
│   ├── models/                # final CNN ladder + noise estimator; legacy modules retained
│   ├── training/              # final CNN training helpers; legacy MOMENT trainers retained
│   ├── eval/                  # mixed-SNR routing selection and metrics
│   └── utils/
├── experiments/
│   ├── quality_routing/       # canonical final experiment entrypoint
│   └── exp_scenario/          # final/supporting/retracted CSVs, figures, checkpoints
└── tests/
```

See `experiments/exp_scenario/MANIFEST.md` for which artifacts are final,
supporting, or retracted/negative.

## Legacy Note

Earlier MOMENT-S/B/L + LoRA SNRGate results, including the old 0.583 headline,
are retained only for reproducibility audit. `REPRODUCIBILITY_FINDINGS.md`
documents why that pipeline is not the current paper claim.
