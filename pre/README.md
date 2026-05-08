# pre/ — Archived prior approach (SNRGate / DualHead)

본 폴더는 **PHM Korea 2026 paper의 이전 접근 (SNRGate / DualHead Router)** 코드를 보관합니다. 2026-05-04 기준 ConfGate (Confidence Cascade) 방향으로 pivot 하면서 이전 코드 일체를 여기로 이동.

## 이동된 내역

### experiments/
| 폴더 | 무엇 | 결과 |
|---|---|---|
| `exp00_sanity/` | MOMENT-S vs MOMENT-L linear probe, 4-fold × 7 SNR sweep | Negative — S가 모든 SNR에서 L 이상 |
| `exp01_estimator/` | NoiseEstimator1D smoke (synthetic, MSE 0.0085→0.0041) | Pipeline OK, 실데이터 미실행 |
| `exp02_routing/` | Skeleton (NotImplementedError) | 미구현 |
| `exp03_snrgate/` | Joint SNRGate full training, 1 epoch / 15 부분 결과 | 시간 초과로 중단 (1 epoch 2h51m) |

### src/phm_routing/
| 모듈 | 역할 |
|---|---|
| `data/noise_injection.py` | AWGN / pink noise injection at known SNR |
| `models/noise_estimator.py` | 1D CNN noise score regressor (~95K) |
| `models/router.py` | HardGateRouter (rule) + LearnableSNRGate (centers + T anneal) |
| `training/train_estimator.py` | Estimator pretraining loop |
| `training/train_routing.py` | Joint estimator + gate + 3-tier MOMENT (LoRA) training |
| `training/train_head.py` | Per-tier head training (mixed-SNR aug 옵션) |

### tests/
| 파일 | 내용 |
|---|---|
| `test_snrgate.py` | LearnableSNRGate (9 tests) + HardGateRouter (1) + LoRA (5), 모두 통과 |

## Pivot 이유 (요약)

1. Sanity 결과 MOMENT-S ≥ MOMENT-L at every SNR → noise-based routing의 전제 무너짐
2. Joint training이 8GB GPU에 ~43h 소요 → 시간 budget 불가
3. Noise estimator의 supervision은 합성 AWGN 라벨 → 배포 transfer 불확실
4. Saturation 문제 (모두 noisy 시 routing 무의미) 해결 안 됨

새 방향: **Confidence Cascade (ConfGate)** — 작은 모델 confidence가 낮을 때만 큰 모델 escalate. 자세한 내용은 노션 한국PHM학회 페이지의 "Confidence Cascade plan — 비판적 검토 + Prerequisite 실험" 페이지 참조.

## 유지된 모듈 (메인 src/)

다음 모듈은 ConfGate 실험에서도 사용하므로 메인 `src/`에 그대로 유지:

- `data/paderborn.py` — Paderborn-PU loader (Protocol 2 split 추가 예정)
- `data/windowing.py` — sliding window
- `models/moment_wrapper.py` — MOMENT-S/B/L wrapper
- `models/lora.py` — LoRA injection (q, v projections)
- `models/classifier_head.py` — Linear probe head
- `eval/pareto.py` — Pareto front 계산 + plot
- `utils/{seed, metrics, flops}.py` — 공통 유틸

## 참조

- 종합 정리: `paper/experiment_summary.md` (이전 SNRGate 실험 결과 정리)
- ConfGate plan: 노션 한국PHM학회 → "Confidence Cascade plan" 페이지
- 이전 commits: `git log -- pre/`
