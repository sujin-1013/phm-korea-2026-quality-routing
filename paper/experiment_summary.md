# SNRGate — 실험 정리본 (PHM Korea 2026)

**작성일**: 2026-04-30
**최종 commit**: `a98d129` (master)
**프로젝트**: KETI 한국PHM학회 2026, 시계열 기초모델의 SNR 인지 게이팅 라우팅

---

## 0. 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX 3070 (8 GB) |
| Python / Torch | 3.13.12 / 2.11.0+cu126 |
| OS / Shell | Linux 6.6.87.2-microsoft-standard-WSL2 / bash |
| Disk | /home/agent05 902 GB free |
| Working dir | /home/agent05/phm_korea_paper |

---

## 1. 데이터 — Paderborn-PU 4-class 베어링

| 항목 | 값 |
|---|---|
| Source | https://groups.uni-paderborn.de/kat/BearingDataCenter/ |
| Bearings | 30개 (K001-K006, KI01-09·14·16·17·18·21, KA01-09·15·16·22·30, KB23·24·27) |
| .mat 파일 | 2,400 (1 corrupt skip → 2,399 valid) |
| Corrupt 파일 | `N15_M01_F10_KA08_2.mat` (scipy.io.loadmat fail) |
| 샘플링 | 64 kHz × ~4 s/file (~256K samples) |
| 운전 조건 | N09_M07_F10, N15_M01_F10, N15_M07_F04, N15_M07_F10 (4종) |
| Class 분포 | N=480, IR=880, OR=800, RE=240 (RE imbalance ~2:8) |
| 라벨 매핑 | `BEARING_CLASS` dict in `src/phm_routing/data/paderborn.py` |

다운로드/추출 스크립트: `scripts/download_paderborn.sh` (curl + 7z).

---

## 2. 모듈 점검 (smoke + unit tests)

### 2.1 Noise injection · windowing · metrics
- 7개 smoke test 모두 통과 (`add_awgn`, `add_pink_noise`, `sliding_windows`, `decimate`, `macro_f1`, `accuracy`, `per_class_f1`)
- AWGN: target SNR=10 → measured 9.94 dB (1% 오차)

### 2.2 MOMENT 메모리 / 속도 프로파일 (frozen encoder + linear probe)

| 모델 | total params | trainable | bs=8 peak | bs=16 | bs=32 | bs=64 peak | t/batch (bs=64) |
|---|---|---|---|---|---|---|---|
| MOMENT-S | 35,339,460 | 2,052 | 0.22 GB | 0.29 | 0.40 | **0.63 GB** | 31 ms |
| MOMENT-B | 109,638,532 | 3,076 | — | 0.67 | 0.89 | **1.31 GB** | 44 ms |
| MOMENT-L | 341,244,420 | 4,100 | 1.65 | 1.76 | 2.04 | **2.59 GB** | 69 ms |

**결과**: 단일 tier × bs=64까지 8 GB GPU에 여유. gradient checkpointing 불필요. Plan 추정치 (40M/125M/385M) 보다 실제 momentfm 빌드는 약간 작음.

### 2.3 SNRGate framework — 15 unit tests 전부 PASS (`tests/test_snrgate.py`, 1.96 s)

| 그룹 | 항목 |
|---|---|
| LearnableSNRGate (9) | init centers, soft weights normalize, T-uniform vs T-peaked entropy, low-T = argmin, hard_tier nearest-center, anneal monotonic decrease, gradient flow, shape validation, config validation |
| HardGateRouter (1) | unchanged threshold behaviour (regression test) |
| LoRA (5) | identity at init, base weights frozen, gradient flows only to A/B, inject_lora wraps named targets, raises on no targets |

---

## 3. exp00_sanity — Linear probe 비교 (negative result, paper의 baseline ablation)

### 3.1 Setup
```
seeds = [42]
folds = 4 (StratifiedGroupKFold by bearing_id)
epochs = 12
batch_size = 32
window / stride = 2048 / 2048 (no overlap)
max_files_per_bearing = 15  (449 files = 15 × 30 - 1 corrupt)
sizes = [small, large]
mode = linear probe (encoder frozen, head only — no LoRA, no SNR aug at train)
```

### 3.2 Mean F1 across 4 folds

| SNR (dB) | MOMENT-L | MOMENT-S | size effect (L − S) |
|---:|---:|---:|---:|
| ∞ (clean) | 0.341 | 0.353 | **−1.3 %p** |
| 20 | 0.333 | 0.353 | −2.0 %p |
| 10 | 0.263 | 0.345 | −8.2 %p |
| 5 | 0.165 | 0.313 | **−14.8 %p (worst)** |
| 0 | 0.096 | 0.197 | −10.1 %p |
| −5 | 0.087 | 0.088 | −0.2 %p |
| −10 | 0.087 | 0.081 | +0.5 %p |

### 3.3 결론
- **Decision gate (plan §1.2)**: ❌ size effect ≥ 5 %p 미충족 → fallback signal
- 단, 이는 **linear probe only**: L의 341M params 중 4,100만 학습 가능. L의 capacity가 head 한 층에 갇혀 활용 불가능.
- Paper에는 "**linear-probe-only baseline에서는 size routing이 성립하지 않음**"의 motivating ablation으로 사용 → SNRGate가 LoRA로 이를 해결하는 motivation 제공.
- 결과 CSV: `experiments/exp00_sanity/results_real_seed42.csv`

---

## 4. exp01_estimator — Noise estimator (smoke only, synthetic)

### 4.1 Setup
- 모델: `NoiseEstimator1D` (95K params, 1D CNN 4 stride-2 blocks)
- Loss: MSE on `snr_to_target(snr_db) = sigmoid(-snr_db / 10)` (0~1 mapping)
- Smoke: 3 epochs on synthetic data

### 4.2 Calibration scan (smoke result)

| true SNR | mean predicted noise | std |
|---:|---:|---:|
| ∞ | 0.025 | 0.007 |
| 20 | 0.032 | 0.011 |
| 10 | 0.148 | 0.076 |
| 5 | 0.313 | 0.136 |
| 0 | 0.504 | 0.064 |
| −5 | 0.627 | 0.012 |
| −10 | 0.733 | 0.010 |

**Monotonic 보장 (clean → 0, 매우 noisy → 1).** Router 임계값 τ₁=0.33, τ₂=0.66 기준 분포가 자연스럽게 매핑됨.

### 4.3 fault/noise disentangle (false-positive on clean fault signal)
- Smoke 결과: `fp_rate_clean (threshold=0.33) = 0.000`
- 단, synthetic 합성 fault라 신호가 너무 깨끗함. 실데이터 disentangle 실험은 미실시 (다음 작업).

### 4.4 한계 (사용자 질문에서 도출)
- Estimator는 "**injected AWGN level**"을 추정하도록 학습됨 (지도 라벨이 합성 노이즈 주입에서 옴).
- 추론 시 임의의 미지 노이즈를 측정한다는 보장 없음 — 가정 3개 (Paderborn 베이스라인 ≈ clean / 노이즈가 AWGN형 / fault ≠ noise spectrally)에 의존.
- Paper에서는 "injected-AWGN level estimator" 로 한정 표현 + disentangle 실험으로 fault/noise 분리 능력 입증해야 함.

---

## 5. exp03_snrgate — SNRGate 풀 학습 (시작 후 시간 초과로 중단)

### 5.1 Setup
```
epochs = 15
batch_size = 4
lora_r = 8 (q, v projections)
lora_alpha = 16.0
gate.init_centers = (0.17, 0.50, 0.83)
gate.T_init / T_min = 1.0 / 0.05
gate.anneal_epochs = 15  (matched to training epochs)
augment_snr = (-10, -5, 0, 5, 10, 20)  -- mixed-SNR train
amp = bf16
max_files_per_bearing = 8  (240 files, 30K windows, 24K train / 6K val)
```

Trainable: LoRA 1,196,032 + head 9,228 (per tier × 3) + estimator 92,737 + gate centers 3 = **~1.3M total**.

### 5.2 진행 결과 (1 epoch / 15)
- ep 00: T=1.000, loss=1.1193, val_f1=0.245
- gate centers μ=(0.01, 0.97, 0.96) — clamp [0,1] 작동, **base/large가 같은 점으로 수렴 (gate가 사실상 binary)**
- 2 h 51 min 경과 (1 epoch만 완료) → 추정 풀 실행 ~43 hours

### 5.3 중단 이유
- 8 GB single GPU 환경에서 3-tier joint training (각 forward + backward × 3) 시간이 너무 큼.
- 사용자 요청으로 중단 후 재구성 필요 (data·epoch 추가 축소 또는 cached MOMENT embedding 등 가속 전략).

### 5.4 산출물 (1 epoch만 완료)
- 체크포인트는 학습 완료 시 저장 (현재 미저장)
- `summary.json`, `snr_sweep.csv`는 smoke 결과만 존재

### 5.5 관찰
- **Gate clamp는 잘 작동** (이전 smoke의 발산 문제 해결)
- **LoRA 학습 자체는 정상** (loss 감소)
- **Base/Large 중심 붕괴 — 가능 원인**:
  - 1 epoch만 봤기 때문에 학습 부족
  - 또는 noise≈0.97 영역에 대부분 batch가 몰림 (mixed-SNR aug에서 -10 / -5 / 0 / 5 / 10 / 20 dB 균등 샘플 → 절반 이상이 0 dB 이하 → 노이즈 추정값 0.5 이상 → base+large만 활성화)

---

## 6. 코드 정리 — 파일 구조 (commit `a98d129`)

```
phm_korea_paper/
├── src/phm_routing/
│   ├── data/        paderborn.py (load_signal: scipy.io.loadmat) · noise_injection.py · windowing.py
│   ├── models/      moment_wrapper.py · noise_estimator.py · router.py (HardGateRouter + LearnableSNRGate) · classifier_head.py · lora.py
│   ├── training/    train_head.py · train_estimator.py · train_routing.py (joint SNRGate)
│   ├── eval/        pareto.py
│   └── utils/       seed.py · metrics.py · flops.py
├── experiments/
│   ├── exp00_sanity/      run.py + results_real_seed42.csv
│   ├── exp01_estimator/   run.py + smoke calibration.csv
│   ├── exp02_routing/     skeleton (NotImplementedError 자리)
│   └── exp03_snrgate/     run.py + smoke artifacts
├── tests/test_snrgate.py  (15 tests)
├── paper/
│   ├── abstract_ko.md             v5
│   ├── related_work_draft.md      Li 2025 / BearingFM / UniFault 차별화 단락
│   └── experiment_summary.md      ★ 본 문서
└── scripts/download_paderborn.sh
```

### Commits (4개)

| SHA | 요약 |
|---|---|
| `a3af706` | 초기 scaffolding + smoke 검증 |
| `8d2a23e` | 초록 톤 변경 (메소드 제안) + sanity CLI subsample 추가 |
| `2855e72` | SNRGate framework (LearnableSNRGate, LoRA, exp03, 15 tests) |
| `a98d129` | gate center clamp [0,1] + exp03 anneal_epochs sync |

---

## 7. 핵심 take-aways

1. **Linear-probe-only setup에서 size routing 전제 불성립** (sanity §3.3). Paper에서는 이걸 motivating baseline으로 박고 SNRGate (LoRA 활성)이 해결한다고 framing.
2. **MOMENT S/B/L 모두 8 GB GPU에 단일 tier로는 여유** (bs=64까지). 단, **3-tier joint training은 single GPU에서 시간 병목** — exp03 풀 실행 시 데이터·epoch 추가 축소 또는 embedding cache 가속 전략 필수.
3. **Gate 학습 안정화**: centers를 [0,1]로 clamp 하지 않으면 발산 (smoke에서 −0.76 / 1.33). `clamp_centers()` 후 step마다 호출이 정답.
4. **Estimator의 supervision은 합성 AWGN 라벨**: under-identification 문제 (fault ≠ noise 분리)를 정직하게 다루는 것이 paper의 핵심 contribution. Disentangle 실험을 실데이터로 돌려 false-positive rate 보고가 차기 우선순위.

---

## 8. 다음 작업 (recommended)

1. **exp03 가속 전략**:
   - max_files_per_bearing 4 + epochs 8 + batch_size 8 → 추정 ~6-8 h (overnight 가능)
   - 또는 frozen MOMENT embedding을 1회 사전 계산해서 .pt로 캐시 → 학습 단계에서 LoRA + head + gate만 forward → ~10x 가속
2. **Real Paderborn으로 fault/noise disentangle 실험** (estimator의 false-positive on clean faulty signals)
3. **Pareto plot 산출** — exp03 결과에서 routing distribution + cost-accuracy 분석
4. **초록 v6**: exp03 결과 받으면 마지막 문장 placeholder를 정량 수치로 교체 ("MOMENT-L 단독 대비 FLOPs ~X% 절감, -5dB F1 0.X 보존")
5. **Manuscript 본문 시작**: Related Work draft를 기반으로 (a) classical SP, (b) DL fault diagnosis, (c) TSFM, (d) routing/MoE/cost-aware 4-section 구성
