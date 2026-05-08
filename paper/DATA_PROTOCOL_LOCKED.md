# 🔒 DATA PROTOCOL — LOCKED

**일자**: 2026-05-06
**결정자**: 사용자 (KETI)
**Scope**: PHM Korea 2026 — SNRGate paper 의 모든 후속 실험·코드·페이퍼 작성
**Status**: **변경 금지 (재옵션화·재질문 금지)**

본 문서는 본 프로젝트의 데이터 셋업에 대한 **단일 진실 소스(Single Source of Truth)** 다.
모든 실험 스크립트·논문 작성은 본 protocol을 그대로 따르며, 변경은 사용자 명시적
결정 없이 불가능하다.

---

## 1. 데이터셋 — Paderborn PU-C1 Artificial-to-Real (A2R), 3-class

### 1.1 클래스 정의

| Index | Class | Bearings 출처 |
|---:|:---|:---|
| 0 | **N** (Healthy / Normal) | K001, K002 |
| 1 | **IR** (Inner Race damage) | KI01–KI09 (artificial), KI14/16/17/18/21 (real) |
| 2 | **OR** (Outer Race damage) | KA01–KA09 (artificial), KA15/16/22/30 (real) |

**제외**: KB23/24/27 (combined damage — A2R 비대칭으로 부적합), K003–K006 (unused).

### 1.2 Train / Val / Test split

| Split | Bearings | 비고 |
|:---|:---|:---|
| **Train (artificial damage)** | K002 + KA01/03/04/05/06/07/09 + KI01/03/04/05/07 = **13 bearings** | A2R source domain |
| **Within-train Val** | K002 (file-level 20%) + KA08 + KI08 = **2 disjoint OR/IR bearings + healthy file split** | early stopping / best-on-val 메트릭 |
| **Test (real damage)** | K001 + KA15/KA16/KA22/KA30 + KI14/KI16/KI17/KI18/KI21 = **10 bearings** | A2R target domain — 모든 paper 결과는 이 set 에서 계산 |

> **Spec deviation (2026-05-06 보정)**: 원 spec L85 는 KA01–KA09 + KI01–KI09 (총 18 베어링) 을
> train artificial 으로 명시했으나, KAt 공식 서버 직접 probe 결과 **KA02, KI02, KI06, KI09 의 4개 베어링은
> 302 redirect 후 404** (deprecated/제거됨). KA06, KA09 는 200 OK 로 추가 다운로드 성공.
>
> 결과: 가용 artificial 베어링 KA01/03/04/05/06/07/08/09 (8개 OR) + KI01/03/04/05/07/08 (6개 IR) = 14개.
> 그 중 KA08, KI08 을 val 로, 나머지 7 OR + 5 IR + K002 = **13 베어링을 train 으로** 재할당.
> Test set 은 변경 없음 (real damage 베어링 모두 가용).
>
> **OR/IR 비대칭**: train 에 OR=7, IR=5 → weighted CE 의 inverse-freq 가 자동 보정.
> 이 비대칭은 KAt 서버 사정으로 발생한 것이며 method 의 한계 아님 → Limitations 에 명시.

> **Val healthy class 사례 처리**: K002 가 단일 healthy 베어링(K003–K006 unused 정책 박제)
> 이라 베어링 단위 disjoint 가 healthy 에서 불가능. 따라서 K002 의 ~80 measurement
> file 을 80/20 file-level split 으로 train/val 공유. 다른 모든 베어링은 bearing-level
> disjoint. Limitations 에 명시.

### 1.3 Window 처리

| 항목 | 값 |
|---|---|
| Window | 2048 samples (no overlap) |
| Stride | 2048 |
| Sampling rate | 64 kHz (Paderborn-PU 표준) |
| `max_files_per_bearing` | **8** — 모든 phase 통일 |
| Per-bearing file 선택 | seeded random (`np.random.default_rng(seed)`) |
| Normalization | window-wise z-score **AWGN injection 후** (SNR signal-power-faithful) |

### 1.4 Class imbalance

- **Weighted Cross-Entropy**: `weights = inverse_class_frequency(y_train)`
- 학습 시 `nn.CrossEntropyLoss(weight=class_weights)` 로 적용
- Per-class F1 별도 보고

---

## 2. AWGN augmentation (training)

| 항목 | 값 |
|---|---|
| Train SNR pool | uniform from {clean, 20, 10, 5, 0, -5, -10, -20} dB |
| Generator | `torch.Generator().manual_seed(0)` (재현 가능) |
| Per-batch | 한 batch 의 모든 sample 에 동일 SNR 적용 (`random_snr_sample`) |

---

## 3. Test SNR sweep (evaluation)

8 levels: **inf, 20, 10, 5, 0, -5, -10, -20 dB** (사용자 요청으로 -20 dB 추가).
모든 paper main table / figure 는 8 levels 전체 보고.

---

## 4. Seeds (정책 유보)

- **현재**: `seed=42` 단일로 파이프라인 검증 우선
- **Spec 권장**: 42, 1, 2 (3 seeds, mean ± std)
- seed=42 결과가 promising 하면 1, 2 추가 후 mean ± std 갱신

---

## 5. 코드 단일 진입점

본 protocol 의 모든 정의는 `src/phm_routing/data/paderborn.py` 의 상수 / 함수에서만
정의된다. 다른 파일에서 bearing 리스트 / class 매핑을 hardcode 하지 않는다.

```python
from phm_routing.data.paderborn import (
    PU_C1_TRAIN_BEARINGS,      # ["K002", "KA01", ..., "KA09", "KI01", ..., "KI09"]
    PU_C1_TEST_BEARINGS,       # ["K001", "KA15", ..., "KI21"]
    PU_C1_VAL_BEARINGS,        # ["KA09", "KI09"]  (K002 file-split는 별도)
    CLASS_NAMES_3,             # ("N", "IR", "OR")
    CLASS_TO_IDX_3,            # {"N":0, "IR":1, "OR":2}
    pu_c1_a2r_split,           # entries → (train_idx, val_idx, test_idx)
    materialise_pu_c1,         # entries → (X_tr, y_tr, X_val, y_val, X_test, y_test)
    inverse_freq_weights,      # y_tr → torch.Tensor of class weights
)
```

---

## 6. 사용 금지 패턴 (영구 폐기 — 본 결정으로 코드에서 제거)

- ❌ `np.random.permutation(len(X))` 으로 윈도우 단위 train/val split (data leakage)
- ❌ 4-class (N/IR/OR/RE) 사용 — RE 베어링 3개뿐 + A2R 비대칭으로 reviewer 방어 불가
- ❌ `max_files_per_bearing` 4 또는 8 외의 값 (실험 비교 불가능해짐)
- ❌ Test set 에 artificial damage 베어링 포함
- ❌ Train set 에 real damage 베어링 포함
- ❌ `num_class=4` 를 RoutingTrainConfig 의 default 로 두기 (3 으로 변경)

---

## 7. 변경 절차

본 protocol 의 변경은 다음을 모두 만족해야 한다:

1. 사용자가 명시적으로 변경 요청
2. `paper/DATA_PROTOCOL_LOCKED.md` 의 "변경 이력" 섹션에 변경 사유·날짜 기록
3. `src/phm_routing/data/paderborn.py` 의 상수 + 모든 코드 진입점 동시 갱신
4. 변경 전 / 후 결과 비교를 paper Limitations 또는 Discussion 에 명시

### 변경 이력

- 2026-05-06: **초기 lock-in** — 사용자 결정. 4-class random split → 3-class PU-C1 A2R 로 영구 전환.
- 2026-05-06 (sanity 보정): 로컬 Paderborn-PU download 에 KA02/06/09 + KI02/06/09 누락 확인.
  Train artificial 을 K002 + KA01/03/04/05/07 + KI01/03/04/05/07 = 11 베어링으로 축소,
  Val 을 KA08 + KI08 로 재할당. Test 변경 없음.
- 2026-05-06 (download 추가): KAt 서버 probe 후 KA06, KA09 (200 OK) 추가 다운로드 성공.
  KA02, KI02, KI06, KI09 는 server-side 404 (deprecated). Train 을 13 베어링으로 확장
  (K002 + KA01/03/04/05/06/07/09 OR + KI01/03/04/05/07 IR). OR/IR 비대칭(7:5)은 dataset
  부산물로 weighted CE 가 보정.

---

## 8. 참고 spec

- `.omc/specs/deep-dive-noise-problem-phm-method.md` §Phase 1, L85 (split 정의)
- `.omc/specs/deep-dive-noise-adaptive-masked-ssl.md` §Setup (동일 protocol 재사용)
- Lessmeier, C. et al. (2016). Paderborn-PU dataset paper.
- FaultFormer (Bayer 2023), Few-Shot TL (PMC9497688) — A2R 3-class protocol 직접 선례.
