# SNRGate — SNR-Aware Gated Routing of TSFMs

> **한국PHM학회 2026 (KETI) — 베어링 결함 진단을 위한 SNR-인지 동적 라우팅**

입력 진동 신호의 노이즈 수준을 추정하고, 학습 가능한 prototype 중심·온도 어닐링을 갖는 SNR-인지 게이트를 통해 MOMENT-S/B/L 3-tier 중 하나로 라우팅한다. 학습 시 soft mixture로 미분 가능하게 결합되고, 추론 시 단일 tier로 환원되어 sparse-MoE 형태의 비용 절감을 보존한다.

<table>
<tr>
<td align="center"><b>Macro-F1</b><br>(7-SNR avg)</td>
<td align="center"><b>Effective Params</b><br>(vs MOMENT-L)</td>
<td align="center"><b>Learnable vs Rule-based</b><br>gate</td>
</tr>
<tr>
<td align="center"><b>+21.7 pp</b><br>vs MOMENT-B</td>
<td align="center"><b>−62%</b><br>341 M → 130 M</td>
<td align="center"><b>+4.9 pp</b><br>vs HardGate</td>
</tr>
</table>

> 논문 본문·실험 protocol 문서는 본 저장소에 포함되지 않습니다 (학회 제출 후 별도 공개 예정).

---

## Contents

1. [Results](#results)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Setup (locked)](#setup-locked)
5. [Layout](#layout)
6. [License & Citation](#license--citation)

---

## Results

Paderborn-PU **PU-C1 A2R** 3-class (N/IR/OR), 7-SNR canonical window, seed=42.

| Method | clean | 20 dB | 10 dB | 5 dB | 0 dB | −5 dB | −10 dB | **avg** |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| MOMENT-S only | 0.325 | 0.326 | 0.324 | 0.322 | 0.311 | 0.296 | 0.288 | 0.313 |
| MOMENT-B only | 0.404 | 0.402 | 0.383 | 0.372 | 0.368 | 0.335 | 0.297 | 0.366 |
| MOMENT-L only | 0.389 | 0.385 | 0.382 | 0.357 | 0.284 | 0.257 | 0.241 | 0.328 |
| HardGate (rule, ablation) | 0.600 | 0.600 | 0.589 | 0.572 | 0.522 | 0.486 | 0.363 | 0.533 |
| **SNRGate (ours)** | **0.636** | **0.636** | **0.633** | **0.624** | **0.591** | **0.563** | **0.396** | **0.583** |

- **Accuracy.** 평균 Macro-F1에서 가장 강한 단일 tier baseline (MOMENT-B) 대비 **+21.7 pp**. 모든 SNR에서 SNRGate가 우위.
- **Cost.** 추론 시 단일 tier만 실행 → 평균 effective parameter **약 62% 절감** (vs MOMENT-L).
- **Gating.** Learnable centers + temp anneal이 rule-based 임계값(τ₁=0.33, τ₂=0.66) 대비 **+4.9 pp**.
- **Limit.** −20 dB까지 8-level sweep을 수행하지만 모든 방법이 공통 붕괴 (논문 §6 한계 분석 참조).

---

## Quick Start

### 1. Environment

```bash
pip install -e .
```

PyTorch (CUDA), `momentfm`, `peft` (LoRA), `pandas`, `matplotlib` 필요.
RTX 3070 8 GB · bf16 AMP · batch_size=8 에서 검증됨.

### 2. Data

Paderborn-PU bearing dataset (학술 사용만 허용 — 라이센스 동의 필요):

```bash
bash scripts/download_paderborn.sh
```

### 3. Reproduce

```bash
# 메인 메소드 — SNRGate 학습 + 8-SNR sweep + 라우팅 분포
python experiments/exp03_snrgate/run.py

# Single-tier baselines (S/B/L 각 LoRA 학습) + per-class·per-bearing 진단
python experiments/exp_baselines/run.py --diag

# HardGate 룰 기반 ablation (메인 ckpt 재사용)
python experiments/exp02_routing/run.py

# 저장된 ckpt 재평가 (per-class F1 + per-bearing accuracy)
python experiments/per_class_per_bearing_eval.py

# Figures + 종합 표 (로컬 paper/figures/ 디렉토리에 자동 저장)
python experiments/aggregate_results.py
```

### 4. Tests

```bash
pytest tests/    # 29 tests
```

---

## Architecture

```
input  (B, 1, 2048)  raw vibration @ 64 kHz
  │
  ├─ NoiseEstimator1D     (~93 K, [trained])           →  noise ∈ [0, 1]
  │     │
  │     └─ LearnableSNRGate                            →  weights w_S, w_B, w_L
  │           centers μ_{S,B,L} + temp anneal, [trained]
  │           학습: soft mixture · 추론: argmax → 단일 tier
  │
  └─ MOMENT 백본 3-tier   (S/B/L, [frozen] + LoRA r=8 on Q/V, [trained])
        │
        └─ per-tier 3-class linear head ([trained])    →  {N, IR, OR}
```

**Inference.** Estimator는 항상 forward, gate은 argmax로 환원되어 1개 tier만 실행 → cost saving 보존.

**Training.** Soft gate를 통해 `estimator + 게이트 중심 + LoRA + head`가 분류 손실로부터 공동 학습. 단조성 정규화로 prototype 중심 순서를 강제:

```
L_order = ReLU(μ_S − μ_B + 0.05)² + ReLU(μ_B − μ_L + 0.05)²
```

---

## Setup (locked)

**Data**

| | |
|:---|:---|
| Source | Paderborn-PU PU-C1 A2R |
| Class | 3 (N / IR / OR), 인공→실 손상 전이 |
| Window | 2048 samples (32 ms @ 64 kHz), stride 2048 |
| Train aug SNR | {−20, −10, −5, 0, 5, 10, 20, ∞} dB pool |
| Eval SNR sweep | {∞, 20, 10, 5, 0, −5, −10, −20} dB (8 levels) |
| Avg definition | clean … −10 dB 7-SNR arithmetic mean |

**Model**

| | |
|:---|:---|
| Backbone | MOMENT-{S, B, L}, encoder frozen |
| Adapter | LoRA r=8, α=16, Q/V projections |
| Estimator | NoiseEstimator1D, ~93 K params |
| Gate | LearnableSNRGate (centers + temp anneal) |
| Head | per-tier 3-class linear |

**Optimization**

| | |
|:---|:---|
| Optim | AdamW |
| LR | 2e-4 (LoRA · head · gate) |
| Weight decay | 1e-3 |
| Batch / precision | 8 / bf16 AMP |
| Seed | 42 |

---

## Layout

```
src/phm_routing/
├── data/        Paderborn-PU loader, windowing, AWGN, locked PU-C1 A2R protocol
├── models/      NoiseEstimator1D, MOMENT wrapper, LoRA, LearnableSNRGate, HardGateRouter
├── training/    train_routing (SNRGate main), train_baseline (single-tier 공통), 추정기·헤드
└── eval/        routing_sweep (HardGate/SNRGate 공용), baseline_diagnostic, Pareto

experiments/
├── exp03_snrgate/                 메인 실험 — SNRGate 학습 + sweep
├── exp_baselines/                 단일 tier baselines (--diag로 per-class·per-bearing 진단)
├── exp02_routing/                 HardGate 룰 기반 ablation
├── exp03_snrgate_main/            메인 ckpt + paper 결과 CSV/JSON
├── per_class_per_bearing_eval.py  저장된 ckpt에서 SNRGate+HardGate 재평가
└── aggregate_results.py           figures + 종합 표 자동 생성

pre/          초기 sanity·estimator scaffold (archive)
tests/        29 unit tests
```

> 논문 원고·figure·protocol 문서는 로컬 `paper/` 디렉토리에 보관되며 공개 저장소에는 포함되지 않습니다.

---

## License & Citation

Paderborn-PU는 학술 사용만 허용. 데이터 사용 시 다음 인용이 필수:

> Lessmeier, C., Kimotho, J. K., Zimmer, D., & Sextro, W. (2016). Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set. *PHM Society European Conference (PHME 2016)*.

본 저장소의 메소드(SNRGate)를 인용할 경우에는 한국PHM학회 2026 논문이 공개된 후 해당 BibTeX를 사용해 주십시오 (학회 발표 후 본 README에 추가될 예정).
