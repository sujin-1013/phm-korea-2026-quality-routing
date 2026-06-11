# SNRGate: SNR-Aware Dynamic Routing of Time-Series Models for Bearing Fault Diagnosis

[![Venue](https://img.shields.io/badge/PHM%20Korea%202026-in%20preparation-orange)]()
[![Dataset](https://img.shields.io/badge/Paderborn%20PU-PU--C1%20A2R-4b8bbe)](https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter)

**SNRGate** is an SNR-aware dynamic routing framework for bearing fault
diagnosis. A lightweight noise estimator scores the noise level of each input
vibration signal, and a learnable gate routes it to one of three time-series
foundation model tiers (MOMENT-S/B/L). Training uses a differentiable soft
mixture; inference executes a single tier, preserving the cost savings of a
sparse mixture-of-experts.

## Highlights

- **SNR-aware routing** — a ~93K-parameter 1D-CNN noise estimator
  (self-supervised with AWGN injection) feeds a learnable gate with ordered
  prototype centers and temperature annealing.
- **+21.7 pp macro-F1** (7-SNR average) over the best single-tier baseline
  (MOMENT-B), winning at every SNR level from clean to −10 dB.
- **−62% effective parameters** versus always running MOMENT-L
  (341M → 130M on average), since only one tier executes per input.
- **Learnable beats rule-based** — the learnable gate outperforms a
  threshold-based HardGate by +4.9 pp; decomposition shows most of the gain
  comes from joint-training-induced tier specialization, not routing alone.

## Results

Macro-F1 on the Paderborn PU-C1 A2R 3-class test set (artificial-to-real
damage transfer), 7-SNR sweep, seed 42:

| Method | clean | 20 dB | 10 dB | 5 dB | 0 dB | −5 dB | −10 dB | avg |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| MOMENT-S only | 0.325 | 0.326 | 0.324 | 0.322 | 0.311 | 0.296 | 0.288 | 0.313 |
| MOMENT-B only | 0.404 | 0.402 | 0.383 | 0.372 | 0.368 | 0.335 | 0.297 | 0.366 |
| MOMENT-L only | 0.389 | 0.385 | 0.382 | 0.357 | 0.284 | 0.257 | 0.241 | 0.328 |
| HardGate (rule-based ablation) | 0.600 | 0.600 | 0.589 | 0.572 | 0.522 | 0.486 | 0.363 | 0.533 |
| **SNRGate (ours)** | **0.636** | **0.636** | **0.633** | **0.624** | **0.591** | **0.563** | **0.396** | **0.583** |

## Dataset

[Paderborn University bearing dataset](https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter)
(Lessmeier et al., PHME 2016) under the locked **PU-C1 A2R** protocol:
17 training bearings with artificial damage, 10 test bearings with real
damage, 3 classes (Normal / Inner-race / Outer-race), 2048-sample windows
(32 ms at 64 kHz). SNR augmentation pool {−10, −5, 0, 5, 10, 20, ∞} dB.

## Installation & Usage

```bash
pip install -e .                       # PyTorch (CUDA), momentfm, peft, pandas, matplotlib
bash scripts/download_paderborn.sh     # academic-use license agreement required
```

Reproduce the paper experiments:

```bash
python experiments/exp03_snrgate/run.py          # main: SNRGate training + SNR sweep
python experiments/exp_baselines/run.py --diag   # single-tier S/B/L baselines
python experiments/exp02_routing/run.py          # HardGate rule-based ablation
python experiments/per_class_per_bearing_eval.py # re-evaluate saved checkpoints
python experiments/aggregate_results.py          # figures + summary tables
pytest tests/                                    # 29 unit tests
```

Experimental setup: MOMENT-{S,B,L} encoders frozen, LoRA r=8 (α=16) on Q/V
projections only, AdamW lr 2e-4, weight decay 1e-3, batch 8, bf16 AMP,
seed 42. Verified on a single RTX 3070 (8 GB).

## Repository structure

```
phm-korea-2026-snrgate/
├── src/phm_routing/
│   ├── data/                          # Paderborn loader, windowing, AWGN, PU-C1 A2R protocol
│   ├── models/                        # NoiseEstimator1D, MOMENT wrapper, LoRA, gates
│   ├── training/                      # SNRGate joint training, baselines, estimator, heads
│   ├── eval/                          # routing sweep, baseline diagnostics, Pareto
│   └── utils/                         # metrics, FLOPs counting, seeding
├── experiments/
│   ├── exp03_snrgate/                 # main experiment: SNRGate training + sweep
│   ├── exp03_snrgate_main/            # main-run results (per-bearing, per-class, sweep CSVs)
│   ├── exp_baselines/                 # single-tier S/B/L baselines
│   ├── exp02_routing/                 # HardGate rule-based ablation
│   ├── exp00_sanity/ exp01_estimator/ # sanity checks and estimator calibration
│   ├── per_class_per_bearing_eval.py  # checkpoint re-evaluation
│   └── aggregate_results.py           # figures and summary tables
├── scripts/
│   └── download_paderborn.sh
├── tests/                             # 29 unit tests
└── pyproject.toml
```

The paper manuscript, figures, and protocol documents are kept in a local
`paper/` directory and are not part of this repository.

## Citation

```bibtex
@inproceedings{choi2026snrgate,
  title     = {SNRGate: SNR-Aware Dynamic Routing of Time-Series Models for Bearing Fault Diagnosis},
  author    = {Choi, Sujin},
  booktitle = {Korean Society for Prognostics and Health Management (KSPHM) Conference},
  year      = {2026},
  note      = {Paper in preparation},
}
```

The BibTeX entry will be updated with official information after the
conference. Dataset citation (mandatory):

```bibtex
@inproceedings{lessmeier2016paderborn,
  title     = {Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set},
  author    = {Lessmeier, Christian and Kimotho, James Kuria and Zimmer, Detmar and Sextro, Walter},
  booktitle = {PHM Society European Conference (PHME)},
  year      = {2016},
}
```

## Contact

Sujin Choi — KETI (Korea Electronics Technology Institute) ·
[@sujin-1013](https://github.com/sujin-1013)
