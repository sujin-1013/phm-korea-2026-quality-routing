# SNRGate — SNR-Aware Gated Routing of TSFMs

**한국PHM학회 2026 논문 (KETI) 코드.**
SNRGate는 입력 진동 신호의 노이즈 수준을 추정하고, 학습 가능한 단별(prototype) 중심과 온도 어닐링을 갖는 SNR-인지 게이트를 통해 MOMENT-S/B/L 3-tier 중 하나로 라우팅하는 백본 무관 프레임워크다. 학습 시에는 soft mixture로 미분 가능하게 결합되고, 추론 시에는 단일 tier로 환원되어 sparse MoE 형태의 비용 절감을 보존한다.

논문 본문: `paper/manuscript_ko.md` · 초록: `paper/abstract_ko.md` · 데이터 프로토콜: `paper/DATA_PROTOCOL_LOCKED.md`

---

## 핵심 결과

Paderborn-PU **PU-C1 A2R** 3-class (N/IR/OR) 셋업, 7-SNR canonical window (clean ~ -10 dB), seed=42.

| Method | clean | 20 | 10 | 5 | 0 | -5 | -10 | **avg** |
|---|---|---|---|---|---|---|---|---|
| MOMENT-S only | 0.325 | 0.326 | 0.324 | 0.322 | 0.311 | 0.296 | 0.288 | 0.313 |
| MOMENT-B only | 0.404 | 0.402 | 0.383 | 0.372 | 0.368 | 0.335 | 0.297 | 0.366 |
| MOMENT-L only | 0.389 | 0.385 | 0.382 | 0.357 | 0.284 | 0.257 | 0.241 | 0.328 |
| **SNRGate (ours)** | **0.636** | **0.636** | **0.633** | **0.624** | **0.591** | **0.563** | **0.396** | **0.583** |

- 7-SNR 평균 Macro-F1: **+21.7 pp** vs 최강 단일 tier baseline (MOMENT-B)
- 평균 effective parameter: **약 62% 절감** vs MOMENT-L 단독 (341.2 M → 130 M)
- 학습 가능한 게이트 vs rule-based HardGate 임계값(τ₁=0.33, τ₂=0.66): **+4.9 pp**
- 평가 시에는 -20 dB까지 8-level sweep을 수행하지만 모든 방법이 공통 붕괴 — 한계 분석은 `paper/manuscript_ko.md` §6 참조

---

## 빠른 시작

### 환경

```bash
pip install -e .
```

PyTorch (CUDA 빌드), `momentfm`, `peft`(LoRA), `pandas`, `matplotlib` 필요. RTX 3070 8 GB에서 학습 검증됨 (bf16 AMP, batch=8).

### 데이터

Paderborn-PU bearing dataset (학술 사용만 허용):

```bash
bash scripts/download_paderborn.sh   # 라이센스 동의 필요
```

### 재현

```bash
# 1) SNRGate (메인 메소드) 학습 + 8-SNR sweep + 라우팅 분포
python experiments/exp03_snrgate/run.py

# 2) Single-tier baselines (S/B/L 각 LoRA 학습) + per-class·per-bearing 진단
python experiments/exp_baselines/run.py --diag

# 3) HardGate 룰 기반 라우팅 ablation (1)에서 학습된 ckpt 재사용
python experiments/exp02_routing/run.py

# 4) 저장된 SNRGate ckpt 재평가 (per-class F1 + per-bearing accuracy)
python experiments/per_class_per_bearing_eval.py

# 5) Figures + 종합 표
python experiments/aggregate_results.py
```

### 단위 테스트

```bash
pytest tests/    # 29 tests
```

---

## 디렉토리

| 경로 | 용도 |
|---|---|
| `src/phm_routing/data/` | Paderborn-PU loader, windowing, AWGN injection, **lock된 PU-C1 A2R protocol** |
| `src/phm_routing/models/` | NoiseEstimator1D, MOMENT wrapper, LoRA, LearnableSNRGate / HardGateRouter |
| `src/phm_routing/training/` | `train_routing` (SNRGate 메인), `train_baseline` (single-tier 공통), 추정기·헤드 |
| `src/phm_routing/eval/` | `routing_sweep` (HardGate/SNRGate 공용), `baseline_diagnostic` (per-class·per-bearing), Pareto |
| `experiments/exp03_snrgate/` | **메인 실험** — SNRGate 학습 + sweep |
| `experiments/exp_baselines/` | 단일 tier baselines + (`--diag`) 진단 |
| `experiments/exp02_routing/` | HardGate 룰 기반 ablation |
| `experiments/per_class_per_bearing_eval.py` | 저장된 ckpt에서 SNRGate+HardGate per-class/per-bearing 재평가 |
| `experiments/aggregate_results.py` | figures + paper 표 자동 생성 |
| `paper/` | abstract, manuscript, figures, data protocol, experiment log |
| `pre/` | 초기 sanity·estimator scaffold (archive, 메인 결과 외 재현용) |
| `tests/` | 29 unit tests |

---

## 아키텍처

```
input (B, 1, 2048)  raw vibration @ 64 kHz
  ├─ NoiseEstimator1D (~93K, 🔥)            →  noise ∈ [0,1]
  │     └─ LearnableSNRGate (centers μ_{S,B,L} + temp anneal, 🔥)
  │           학습: soft mixture / 추론: argmax → 단일 tier
  └─ MOMENT 백본 3-tier (S/B/L, 🔒 frozen + 🔥 LoRA r=8 on Q/V)
        └─ per-tier 3-class linear head (🔥)  →  {N, IR, OR}
```

추론 시 estimator는 항상 forward, gate은 argmax로 환원돼 1개 tier만 실행 (cost saving 보존). 학습 시에는 soft gate를 통해 estimator·게이트 중심·LoRA·head가 분류 손실로부터 공동 학습되며, 단조성 정규화 `ReLU(μ_S - μ_B + 0.05)² + ReLU(μ_B - μ_L + 0.05)²` 로 prototype 중심 순서를 강제한다.

---

## 학습·평가 셋업 (lock)

| 항목 | 값 |
|---|---|
| Window | 2048 samples (32 ms @ 64 kHz), stride 2048 |
| Class | 3 (N/IR/OR), PU-C1 A2R 인공→실 손상 전이 |
| Train aug SNR pool | {-20, -10, -5, 0, 5, 10, 20, ∞} dB |
| Eval SNR sweep | {∞, 20, 10, 5, 0, -5, -10, -20} dB (8 levels) |
| 평균 정의 | clean ~ -10 dB 7-SNR 산술 평균 (paper §5 표준) |
| LoRA | r=8, α=16, Q/V projections, encoder frozen |
| Optim | AdamW, lr=2e-4 (LoRA·head·gate), weight_decay=1e-3 |
| Batch / precision | 8 / bf16 AMP |
| Seed | 42 (메인 결과) |

상세는 `paper/DATA_PROTOCOL_LOCKED.md` 및 `paper/experiment_summary.md` 참조.

---

## 데이터 라이센스 / 인용

Paderborn-PU는 학술 사용만 허용. 사용 시 다음 인용:

> Lessmeier, C., Kimotho, J. K., Zimmer, D., & Sextro, W. (2016). Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set. *PHM Society European Conference (PHME 2016)*.

본 저장소를 인용할 때는 PHM Korea 2026 논문 (`paper/manuscript_ko.md`)을 참조해 주십시오.
