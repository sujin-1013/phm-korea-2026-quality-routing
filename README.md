# SNRGate — SNR-Aware Gated Routing of Time-Series Models for Bearing Fault Diagnosis

📄 한국PHM학회 2026 (KETI) — *논문 제출 준비중*

This repository contains the implementation and resources for **SNRGate** — an SNR-aware dynamic routing framework that estimates the noise level of an input vibration signal and selects one of three time-series foundation models (TSFM) at inference time. Soft mixing is used during training, and a single tier is executed during inference, preserving the cost saving of sparse mixture-of-experts.

---

## 🔍 Overview

산업 환경의 베어링 진동 신호는 부하·회전 속도·센서 잡음에 따라 입력 품질이 크게 달라진다. 기존 진단 모델은 입력 품질과 무관하게 동일한 모델을 쓰기 때문에 — **경량 모델은 저품질 신호에서 성능이 무너지고**, **대형 모델은 고품질 신호에서도 불필요한 연산**을 수행한다.

**SNRGate** 는 다음을 통해 이 문제를 해결한다:

- 입력 신호의 노이즈 수준을 [0, 1] 정규화 점수로 추정 (`NoiseEstimator1D`, ~93 K params)
- 학습 가능한 prototype 중심 + 온도 어닐링을 갖는 게이트로 **MOMENT-{S, B, L} 중 하나로 라우팅**
- 학습 시 soft mixture로 미분 가능 / 추론 시 argmax 환원 → **sparse-MoE 형태의 비용 절감 보존**
- 단조성 정규화로 prototype 중심 순서를 강제, **방법-특화적 라우팅 실패** 진단 포함

---

## 🎯 Key Components

- **Noise Level Estimator**
  - 1D CNN, AWGN 주입 자기지도 학습
  - sigmoid(−σ_dB / 10) soft label 회귀
- **Learnable SNR Gate**
  - Prototype centers μ_S < μ_B < μ_L + temperature anneal
  - Ordering penalty: `ReLU(μ_S − μ_B + 0.05)² + ReLU(μ_B − μ_L + 0.05)²`
- **MOMENT 3-Tier Backbone**
  - Encoder frozen, LoRA r=8 on Q/V projections only
  - 단별 3-class linear head (Normal / Inner-race / Outer-race)
- **HardGate (rule-based ablation)**
  - 비교용 임계값 라우팅 (τ₁=0.33, τ₂=0.66)

---

## 🛠️ Code Structure

```
phm-korea-2026-snrgate/
├── src/phm_routing/
│   ├── data/        Paderborn-PU loader · windowing · AWGN · locked PU-C1 A2R protocol
│   ├── models/      NoiseEstimator1D · MOMENT wrapper · LoRA · LearnableSNRGate · HardGateRouter
│   ├── training/    train_routing (SNRGate main) · train_baseline (single-tier) · 추정기·헤드
│   └── eval/        routing_sweep (HardGate/SNRGate 공용) · baseline_diagnostic · Pareto
├── experiments/
│   ├── exp03_snrgate/                 메인 실험 — SNRGate 학습 + sweep
│   ├── exp_baselines/                 단일 tier baselines (`--diag` 로 per-class·per-bearing)
│   ├── exp02_routing/                 HardGate 룰 기반 ablation
│   ├── per_class_per_bearing_eval.py  저장된 ckpt 재평가
│   └── aggregate_results.py           figures + 종합 표 자동 생성
├── tests/                             29 unit tests
└── pre/                               초기 sanity·estimator scaffold (archive)
```

> 논문 원고·figure·protocol 문서는 로컬 `paper/` 디렉토리에 보관되며 공개 저장소에는 포함되지 않습니다.

---

## 🧪 Dataset

- **Paderborn University Bearing Dataset** (Lessmeier *et al.*, PHME 2016)
- **Protocol**: PU-C1 A2R — *artificial → real* damage transfer
- **Classes (3)**: Normal · Inner-race · Outer-race
- **Splits**: 17 train bearings (artificial damage) · 10 test bearings (real damage)
- **Window**: 2048 samples (32 ms @ 64 kHz), stride 2048
- **SNR augmentation pool**: `{−20, −10, −5, 0, 5, 10, 20, ∞}` dB
- **Evaluation sweep**: `{∞, 20, 10, 5, 0, −5, −10, −20}` dB (8 levels)

---

## 📊 Result Summary

> **Macro-F1**, 7-SNR canonical window (clean ~ −10 dB), seed=42, single run.

| Method | clean | 20 dB | 10 dB | 5 dB | 0 dB | −5 dB | −10 dB | **avg** |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| MOMENT-S only | 0.325 | 0.326 | 0.324 | 0.322 | 0.311 | 0.296 | 0.288 | 0.313 |
| MOMENT-B only | 0.404 | 0.402 | 0.383 | 0.372 | 0.368 | 0.335 | 0.297 | 0.366 |
| MOMENT-L only | 0.389 | 0.385 | 0.382 | 0.357 | 0.284 | 0.257 | 0.241 | 0.328 |
| HardGate (rule, ablation) | 0.600 | 0.600 | 0.589 | 0.572 | 0.522 | 0.486 | 0.363 | 0.533 |
| **SNRGate (Ours)** | **0.636** | **0.636** | **0.633** | **0.624** | **0.591** | **0.563** | **0.396** | **0.583** |

| Metric | SNRGate (Ours) | Comparison |
|:---|---:|---:|
| Macro-F1 (7-SNR avg) | **0.583** | **+21.7 pp** vs MOMENT-B (strongest single tier) |
| Effective parameters (avg) | **~130 M** | **−62%** vs MOMENT-L (341.2 M) |
| Gate design | learnable | **+4.9 pp** vs HardGate (rule) |

> 🔹 모든 SNR에서 SNRGate가 우위. −20 dB까지 sweep 시 모든 방법이 공통 붕괴 — 자세한 한계 분석은 논문 §6 참조.

---

## 🚀 Quick Start

### Environment

```bash
pip install -e .
```
PyTorch (CUDA), `momentfm`, `peft` (LoRA), `pandas`, `matplotlib` 필요.
검증 환경: **NVIDIA RTX 3070 8 GB · bf16 AMP · batch=8**.

### Data

```bash
bash scripts/download_paderborn.sh   # 학술 사용 라이센스 동의 필요
```

### Reproduce

```bash
# 1) 메인 — SNRGate 학습 + 8-SNR sweep + 라우팅 분포
python experiments/exp03_snrgate/run.py

# 2) Baselines — Single-tier S/B/L (--diag 옵션으로 per-class·per-bearing)
python experiments/exp_baselines/run.py --diag

# 3) Ablation — HardGate 룰 기반 라우팅 (메인 ckpt 재사용)
python experiments/exp02_routing/run.py

# 4) 재평가 + figures
python experiments/per_class_per_bearing_eval.py
python experiments/aggregate_results.py
```

### Tests

```bash
pytest tests/    # 29 tests
```

---

## ⚙️ Setup (locked)

| Component | Detail |
|:---|:---|
| **Backbone** | MOMENT-{S, B, L} (Goswami *et al.*, ICML 2024), encoder frozen |
| **Adapter** | LoRA r=8, α=16, Q/V projections only |
| **Estimator** | NoiseEstimator1D, ~93 K params |
| **Gate** | LearnableSNRGate (centers + temperature anneal) |
| **Optimizer** | AdamW · lr 2e-4 (LoRA · head · gate) · weight_decay 1e-3 |
| **Batch / Precision** | 8 / bf16 AMP |
| **Seed** | 42 |

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{choi2026snrgate,
  title     = {SNRGate: SNR-Aware Dynamic Routing of Time-Series Models for Bearing Fault Diagnosis},
  author    = {Choi, Sujin},
  booktitle = {Korean Society for Prognostics and Health Management (KSPHM) Conference},
  year      = {2026},
  note      = {Paper in preparation},
}
```

> BibTeX는 학회 발표 후 공식 정보로 갱신될 예정입니다.

Dataset citation (mandatory):

```bibtex
@inproceedings{lessmeier2016paderborn,
  title     = {Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set},
  author    = {Lessmeier, Christian and Kimotho, James Kuria and Zimmer, Detmar and Sextro, Walter},
  booktitle = {PHM Society European Conference (PHME)},
  year      = {2016},
}
```

---

## 🧑‍💻 Author

**Su-jin Choi (최수진)** — KETI (Korea Electronics Technology Institute)
🐙 [@sujin-1013](https://github.com/sujin-1013)

📄 Previous work: [Task-Aware Dynamic Model Optimization (IEEE Access 2023)](https://github.com/sujin-1013/task-aware-DMO)
