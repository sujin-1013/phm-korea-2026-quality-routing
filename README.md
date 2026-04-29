# PHM Korea 2026 — Noise-Adaptive MOMENT Routing

KETI 한국PHM학회 2026 논문 코드. 진동 신호 fault diagnosis에서 입력의 noise level을 추정하고, 추정된 SNR에 따라 MOMENT (TSFM) S/B/L 3-tier 중 하나로 hard-gate routing 한다.

자세한 계획은 `/home/agent05/.claude/plans/keti-phm-sparkling-moth.md` 참조.

## 빠른 시작

```bash
# 1) 환경 (이미 시스템 Python 3.13 + torch 2.11 + cuda 12.6 설치됨)
pip install -e .

# 2) Paderborn-PU 베어링 데이터 다운로드 (라이센스 동의 필요)
bash scripts/download_paderborn.sh

# 3) Sanity check (Phase 1, ~2-3일)
python -m experiments.exp00_sanity.run \
    experiment=sanity_moment_s_vs_l \
    data.snr_levels='[inf,20,10,5,0,-5,-10]' \
    seeds='[42,1234,9999]'
```

## 디렉토리

| 경로 | 용도 |
|---|---|
| `src/phm_routing/data/` | Paderborn-PU loader, windowing, AWGN/pink injection |
| `src/phm_routing/models/` | Noise estimator, MOMENT wrapper, router, classifier head |
| `src/phm_routing/training/` | Phase 1 head training, estimator training, LoRA fine-tune |
| `src/phm_routing/eval/` | SNR sweep, Pareto, routing distribution, calibration |
| `experiments/exp00_sanity/` | Week 1 sanity check (decision gate) |
| `experiments/exp01_estimator/` | Noise estimator 학습·평가 |
| `experiments/exp02_routing/` | 풀 파이프라인 |
| `paper/` | abstract, manuscript, figures |

## 핵심 아키텍처

```
input (1×T raw vibration)
  ├─ Noise Estimator (1D CNN, ~100K, 🔥)  →  noise_level ∈ [0,1]
  │     └─ Router (rule, τ₁=0.33, τ₂=0.66)
  └─ MOMENT-{S|B|L} (40M/125M/385M, 🔒 frozen / 🔥 LoRA Phase3)
        └─ per-tier 4-class head (🔥)  →  {normal, inner, outer, RE}
```

추론 시 estimator는 항상 forward, 그 다음 1개 tier만 forward (hard gate → cost saving 보존).

## MOMENT 입력

- 명목상 `seq_len=512`, `patch_len=8` (64 patches)
- `(B, 1, 2048)` 같은 longer input도 forward 통과 확인됨
- 64 kHz 원본 → MOMENT 입력 정합을 위해 다운샘플 또는 다중 윈도우 평균 옵션 보유

## 데이터 라이센스

Paderborn-PU는 학술 사용만 허용. 사용 시 다음 인용:

> Lessmeier, C., Kimotho, J. K., Zimmer, D., & Sextro, W. (2016).
> Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set.
> PHME Society European Conference.
