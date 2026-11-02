# SNRGate Reproducibility Findings (2026-06-16)

조사 목적: 출판된 SNRGate 헤드라인(7-SNR 평균 macro-F1 **0.583**, best single-tier 대비 **+21.7 pp**)이
이 저장소의 코드로 재현되는지 검증.

## 결론 (요약)

**출판 0.583은 커밋된 코드로 재현되지 않으며, 원본 산출물(체크포인트)이 존재하지 않는다.**
모든 변수를 통제한 재현 시도는 일관되게 7-SNR test 평균 **~0.30–0.36**을 산출했다.
출판 수치 자체도 내부 모순(test > val)을 보인다. 현재 상태로는 0.583 / +21.7 pp 헤드라인을
검증하거나 방어할 수 없다.

---

## 1. 재현 시도 결과 (버전 일치 환경)

환경: Python 3.11, torch 2.5.1+cu121, **transformers 4.33.3** (momentfm 핀), numpy 1.26.4.
프로토콜: PU-C1 A2R 3-class, max_files_per_bearing=8 (n_train≈12,746 ≈ 출판 12,747), 8 epochs, seed 42.

| 설정 | 7-SNR test 평균 | best_val | 게이트 라우팅 |
|---|---:|---:|---|
| **출판 (exp03_snrgate_main)** | **0.583** | 0.465 | 정상 (S65 / B4 / L30) |
| SBL, 추정기 from-scratch (문서화된 기본 경로) | 0.322 | 0.626 | 붕괴 (B 0%) |
| SBL, 사전학습 추정기 로드 | 0.301 | 0.561 | SNR따라 S→B→L 이동 |
| SSS (small×3, ablation) | 0.361 | 0.666 | 한 tier로 붕괴 |
| (참고) 깨진 환경 transformers 5.x | 0.186 | 0.342 | 붕괴 |

- 어떤 조합도 0.58대에 닿지 못함. 추정기 사전학습(가설) 추가해도 0.301.
- 출판 summary가 기록한 정확한 설정(`lora_r=8`, `best_epoch=7` → 8 epochs, `sweep_eval_set: test (PU-C1 real-damage bearings)`)을 그대로 실행한 결과가 ~0.30.

## 2. 코드가 결과 생성 시점 이후 바뀌지 않았음

- 결과 CSV는 커밋 `bc8b14b` ("feat: SNRGate — full pipeline + paper artefacts")에서 추가됨.
- `git diff bc8b14b HEAD -- src/phm_routing/training/ src/phm_routing/eval/ src/phm_routing/data/ experiments/exp03_snrgate/run.py` = **빈 결과**.
- 즉 **현재 실행한 코드 = 출판 결과를 커밋한 바로 그 코드.** 그런데도 재현 안 됨.

## 3. 다른 버전(pre/ 아카이브)도 재현 못 함

- 삭제된 `pre/` 스캐폴드(커밋 `f14cb7e`로 제거)의 `exp03_snrgate/snr_sweep.csv`:
  7-SNR 평균 **0.267**, val_f1 2개(2 epochs)만, final_centers `[-0.757, 1.329, 0.081]` (범위 [0,1] 밖으로 발산).
- 미완성 초기 run이며 0.583과 무관.

## 4. 원본 체크포인트 부재

- `experiments/exp03_snrgate_main/`에 체크포인트 커밋 없음 (`git ls-files` 확인).
- 디스크/머신 전체 검색(`find /home -path "*exp03_snrgate_main*" -name "*.pt"`)에서도 0건.
- 원 학습 환경(`/home/agent05/phm_korea_paper`, summary.json의 source_ckpt 경로)은 머신에 더 이상 존재하지 않음.
- 따라서 원본 모델을 로드해 sweep만 재평가하는 검증도 불가.

## 5. 출판 수치의 내부 모순

- 출판: best_val_f1 = **0.465**, test 7-SNR 평균 = **0.583** → **test > val**.
- 어려운 A2R 전이(test = 실제 손상 베어링)에서 test가 val보다 높은 것은 매우 이례적.
- 본 조사의 모든 재현은 정상 방향(test 0.30 ≪ val 0.56).

## 6. 조사 중 발견한 저장소 결함

1. **추정기 save/load 계약 불일치**: `exp01_estimator/run.py`는 `{"state_dict": ...}`로 저장하나
   `train_routing.build_classifier`는 `state.get("model", state)`로 로드 → 키 불일치로 추정기 로딩이
   원래 작동하지 않음. (조사 중 형식 변환으로 우회.) 이는 출판 파이프라인이 커밋된 코드와 다른 코드로
   실행됐음을 시사.
2. **손상 데이터 파일 1개**: `KA08/N15_M01_F10_KA08_2.mat` (scipy loadmat 실패). 메인 로더는 skip하나
   exp01 로더는 crash. 격리 후 진행. (영향: 2000개 중 1개, 무시 가능.)

---

## 권고

1. **헤드라인 0.583 / +21.7 pp를 현재 상태로 주장하지 말 것.** 코드 공개 시 재현 불가가 즉시 드러난다.
2. 원저자가 **사라진 원본 코드/체크포인트를 다른 백업/서버에서 복구**할 수 있으면 그것으로 재검증.
   복구 불가 시 헤드라인을 새로 세워야 함.
3. **재현 가능한 정직한 baseline 확립** (진행 중): 동일 환경에서
   - SNRGate 다중 시드 → 평균±std
   - S/B/L 각 모델을 자기 검증 최고점까지 학습(generous epochs + best-val)한 fair 비교
   를 통해 유령 수치 대신 검증된 수치로 논문을 재구성.

## 7. 정직한 대체 결과 (버전 일치 환경, 재현 가능)

### 7-1. SNRGate 다중 시드 (추정기 사전학습 로드, r=8, 8ep)
- seed 42 = 0.301, seed 0 = 0.340 → **평균 0.320 ± 0.019**.
- 출판 0.583과 무관. 재현 가능한 SNRGate는 ~0.32.

### 7-2. 공정 단일 모델 비교 (각 모델을 자기 검증 최고점까지: r=32, 40 epochs, best-val checkpoint)
A2R test sweep 7-SNR 평균:

| 모델 | test avg | (참고) val | 기존(불공정, r=8/저에폭) |
|---|---:|---:|---:|
| MOMENT-S | 0.292 | — | 0.313 |
| MOMENT-B | 0.298 | — | 0.366 |
| MOMENT-L | 0.296 | **0.494** | 0.328 |

**핵심 발견:**
1. **공정 예산에서 S≈B≈L (test 0.292/0.298/0.296).** 기존 "B>L 역전"은 학습 예산 불균형 아티팩트였다.
2. **L은 val 0.494로 충분히 학습되나 test는 0.296** — 인공 손상엔 적합하지만 실제 손상(A2R) 일반화 실패.
3. **학습량을 늘릴수록 test 하락** (B 0.366→0.298): 인공 손상 과적합이 A2R 일반화를 악화.

### 7-3. 함의
- **병목은 모델 용량이 아니라 A2R 인공↔실제 손상 도메인 갭.** test는 용량·학습량에 둔감하며 모두 ~0.30에서 정체.
- 단일 tier들이 test에서 동등하므로, "노이즈/품질에 따라 다른 용량으로 라우팅"이라는 SNRGate 전제의 근거가 약함.
- 재현된 SNRGate(0.32)가 단일 tier(~0.30) 대비 우위는 +0.02~0.03 수준으로, 출판 +21.7pp과 상이.

## 8. 추가 검증 (2026-06-17): CNN 아키텍처 + 용량 라우팅

frozen-MOMENT 대신 처음부터 학습하는 1D-CNN으로 "방법 개선"을 정직하게 시도. 모든 설정 사전 고정, 결과 그대로 보고.

### 8-1. CNN 용량 사다리 (in-dist, test 7-SNR 평균)
| tier | params | in-dist | LOCO |
|---|---:|---:|---:|
| tiny | 1K | 0.368 | 0.356 |
| small | 9K | 0.390 | 0.280 |
| base | 133K | **0.543** | 0.267 |
| large | 524K | 0.532 | 0.292 |

- **CNN(0.543) > frozen-MOMENT(0.483)**: end-to-end 학습이 절대 성능을 올림.
- **용량은 tiny→base까지 강하게 도움, base→large는 정체** (base가 sweet spot).
- LOCO(운전조건 전이)는 CNN으로도 낮고 무질서 — 도메인 전이는 용량과 무관.

### 8-2. 용량 라우팅 {tiny, small, base} vs 단일 base (in-dist)
사전학습 추정기(frozen) + 학습형 게이트 + soft-mixture 공동학습, hard 라우팅 추론. val은 7-SNR 평균 라우팅 F1로 선택.

| 방법 | 7-SNR avg | eff_params |
|---|---:|---:|
| 단일 base | **0.543** | 133K |
| 라우터 (seed42 / seed1 평균) | 0.502 | 113K |

- **라우터 < 단일 base** (0.50 vs 0.54). 이기지도, 동급-저비용도 아님.
- 라우팅 분포: 노이즈 구간은 base로(동등), **clean을 tiny로 보내 0.23으로 폭락** → 비용 절감(15%)이 정확도 손실로 상쇄.
- 구조적 이유: **base가 전 노이즈 구간 최고 → base를 벗어나면 정확도 손실. 라우팅이 이길 여지 없음.**

### 8-2b. 게이트 파라미터화 변형: 2개의 learnable threshold (center 대신)
center+softmax 대신 **2개의 학습형 임계값 τ1<τ2**로 noise 점수를 3구간 분할(soft sigmoid 학습, hard 추론). 동일 in-dist 데이터/aug/추정기.

| 방법 | 7-SNR avg | 학습된 (τ1, τ2) |
|---|---:|---|
| 단일 base | **0.543** | — |
| center 게이트 (mono) | 0.502 | — |
| **threshold 게이트 (seed42/1 평균)** | 0.454 ± 0.036 | τ1→~0.002, τ2→~0.14 |
| HardGate (고정 0.33/0.66) | 0.416 | — |

- **두 임계값 모두 0 근처로 수렴 → 거의 전부 base로 라우팅**, clean만 tiny로 보내져 clean f1이 0.218/0.313으로 폭락.
- 게이트가 스스로 3-tier를 ~2-tier(주로 base)로 붕괴시킴. center보다 더 공격적으로 base에 몰려 clean 손실이 커지고 평균이 더 낮음(0.454<0.502).
- **함의:** 게이트 구조(center↔threshold)를 바꿔도 근본은 동일 — base가 전 구간 최고라 학습형 라우터는 "전부 base"로 수렴하려 하고, 작은 tier로 보내는 순간 손해. 라우팅이 이길 여지가 구조적으로 없음을 재확인.

### 8-2c. ~50K 중간 tier(mid) + 활성-파라미터(MoE) Pareto 검증
small(9K)↔base(133K) 간극을 메우려 **mid=(20,4)≈53K** tier 추가. threshold 게이트로 {tiny, mid, base} 라우팅(2 seed) + 동일 예산 단일 mid(2 seed) 비교. MoE처럼 **활성(=eff) 파라미터** 기준으로 평가(라우터는 입력당 1 tier만 실행).

| 구성 | active params | 7-SNR F1 |
|---|---:|---:|
| 단일 tiny | ~1K | 0.368 |
| 단일 small | ~9K | 0.390 |
| 단일 mid | ~53K | 0.488 |
| **단일 base** | ~133K | **0.543** |
| 단일 large | ~524K | 0.532 |
| 라우터 center(mono) | ~113K | 0.502 |
| 라우터 thresh (tiny/small/base) | ~106K | 0.454 |
| 라우터 thresh (tiny/mid/base) | ~43K | 0.424 |

- **동일 ~50K 예산: 단일 mid(0.488) > mid-라우터(0.424).** active-param(MoE) 회계를 그대로 받아도 라우팅이 짐.
- 라우터는 mid로 보내 eff_params는 낮췄으나(43K) 정확도가 더 떨어짐; base는 거의 안 씀.
- **단일 모델 곡선이 곧 Pareto frontier이고 모든 라우터는 그 아래(dominated).** 어느 active 예산에서도 "그 크기 단일 모델 하나"가 라우팅보다 낫거나 같음.
- 게이팅 오버헤드 각주: noise estimator(~93K)는 입력마다 상시 실행 → CNN 체제에선 비-무시(tier와 비슷한 크기). MOMENT 체제에선 무시 가능(진짜 MoE 게이트와 동일).

### 8-2d. SNR-영역별 전문가 specialization (positive 결과)
가설: 라우터가 진 이유는 joint soft-mixture에서 게이트가 붕괴해 **각 tier가 영역 전문가로 형성되지 못해서**다. 검증: 각 tier를 자기 SNR 밴드에만 명시적으로 학습(닭-달걀 제거) 후 라우팅.
- clean-expert = small(9K) ← {clean,20,10}, mid-expert = mid(53K) ← {5,0}, noise-expert = base(133K) ← {−5,−10,−20}.
- oracle(진짜 SNR) + estimator(추정기 점수, val로 임계값 선택) 라우팅, 2 seed.

| SNR | 일반화 base(133K) | 전문가 라우팅(2-seed avg) |
|---|---:|---:|
| clean | 0.513 | 0.833 |
| 20 | 0.511 | 0.828 |
| 10 | 0.526 | 0.787 |
| 5 | 0.535 | 0.858 |
| 0 | 0.538 | 0.798 |
| −5 | 0.553 | 0.447 |
| −10 | 0.626 | 0.536 |
| **7-SNR avg** | **0.543 @133K** | **0.727 @~58K** |

- oracle 0.727 / estimator 0.728 (거의 동일 → 배포 가능 라우팅으로도 천장 도달). active ~58K < base 133K → **더 정확하고 더 쌈 (+0.18)**.
- **이득은 clean~mid SNR(+0.26~0.32)에 집중**, 무거운 노이즈(−5,−10)는 전문가도 일반화 base에 약간 짐(용량/데이터 한계).
- 메커니즘: 일반화 base는 7-SNR을 한 모델로 학습해 "노이즈 둔감하지만 평범(0.51 평탄)"한 절충점에 수렴 → clean의 분리 가능 정확도(0.85)를 버림. 전문가는 그걸 회수.
- **함의: SNRGate 기여 재정의 — "노이즈 강건성용 용량 라우팅"(8-2/8-2b/8-2c에서 실패)이 아니라 "SNR 영역별 전문가 specialization"(여기서 성공).**

#### 8-2d-i. 통제 1: 동일 용량(3×base) — 이득이 specialization인가 용량매칭인가
세 전문가를 모두 base(133K)로 두고 동일 specialization (in-dist, 2 seed):

| 구성 | active | 7-SNR F1 |
|---|---:|---:|
| 단일 base (일반화) | 133K | 0.543 |
| 용량사다리 전문가 (small/mid/base) | ~58K | 0.727 |
| 3×base 전문가 (동일 용량) | 133K | 0.743 |

- 3×base 전문가도 0.743 ≫ 단일 base 0.543 (**+0.20**) → **이득의 원천은 용량매칭이 아니라 specialization 자체**.
- 사다리(0.727@58K) ≈ 3×base(0.743@133K) → **용량매칭은 정확도 손실 0.016으로 active 비용 2.3배 절감하는 공짜 최적화**.

#### 8-2d-ii. 통제 2: A2R 하드 전이 — specialization 이득이 전이되는가
동일 용량사다리 전문가를 A2R(인공→실제 손상)에서 (2 seed):

| 구성 | active | 7-SNR F1 |
|---|---:|---:|
| 전문가 라우팅 (A2R, 용량사다리) | ~58K | **0.331** |
| 전문가 라우팅 (A2R, 3×base) | 133K | **0.314** |
| (참고) in-dist 전문가 | ~58K | 0.727 |
| (참고) 기존 A2R 모든 시도 | — | 0.29~0.36 |

- A2R에선 전문가도 **0.331**(사다리)·**0.314**(3×base) — 기존 SNRGate(0.32)·공정 S/B/L(0.30)과 동일 자리. **용량(3배↑)도 specialization도 전이 안 됨.**
- clean-expert: val(인공손상) ~0.82 ↔ test(실제손상) ~0.29 → val≫test 갭이 곧 도메인 갭. 용량을 키워도 갭 그대로.
- A2R에선 노이즈 있을 때(−5: 0.46)가 clean(0.30)보다 높은 역전 — 노이즈가 인공↔실제 차이를 가리는 정규화.

#### 8-2d-iii. 통제 3: bearing-disjoint in-dist — specialization 이득이 누설인가
in-dist 이득(8-2d)이 진짜인지 검증. recording-level split(같은 베어링이 train/test에 존재 → 누설)을 **bearing-disjoint**(베어링 단위 분리, 같은 손상유형 풀, 운전조건 공유)로 바꿔 재실행 (2 seed, 동일 셋팅).

| 분할 | 단일 base | 전문가 라우팅 | 이득 |
|---|---:|---:|---:|
| recording-level in-dist (누설) | 0.543 | 0.727 | **+0.18** |
| bearing-disjoint in-dist (누설 제거) | 0.316 | 0.304 | **−0.01** |
| A2R (인공→실제) | ~0.35 | 0.331 | ~0 |

- **누설 제거 즉시 specialization 이득 소멸.** 전문가 라우팅(0.304) ≤ 단일 mid/base(0.32).
- 전문가 clean F1: 누설 0.83 → bearing-disjoint 0.28 → A2R 0.30. **0.83은 베어링 누설 산물.**
- 단일 모델도 동반 추락(small 0.39→0.29, base 0.54→0.32) → recording-level split이 절대수치를 ~0.10~0.18 부풀림.

#### 8-2d-iv. joint vs non-joint 라우팅 on A2R (보강)
"joint 학습을 안 해서 전문가가 안 됐나" 가설의 반대편도 검증: 게이트+tier soft-mixture **joint** 라우터(small/mid/base)를 A2R에서 (2 seed).
- joint 라우터 A2R = 0.358 (s42 0.334 / s1 0.381) ≈ 독립 전문가 0.331 ≈ 단일 mid/base ~0.36. **joint도 A2R 못 살림.**

#### 8-2d 종합 (통제 3종 이후 — 정정된 최종 결론)
- 8-2d의 recording-level in-dist 이득(+0.18~0.20)은 **베어링 누설 산물**이었음. **bearing-disjoint(누설 제거) 즉시 소멸**(전문가 0.304 ≤ 단일 base 0.316), A2R에서도 ~0.33으로 소멸.
- 즉 **SNR-영역 전문가 라우팅은 누설 없는 어떤 평가에서도 단일 모델을 못 이김.** "in-dist positive"는 발표 헤드라인으로 부적합.
- joint·non-joint, 용량사다리·동일용량 모두 동일 결론 — 라우팅 구조와 무관.
- **남는 정직한 기여: (a) 출판 0.583 재현 불가 검증, (b) recording-level split이 절대수치를 ~0.10~0.18 부풀린다는 정량적 누설 진단, (c) 인공→실제 손상 도메인 갭이 진짜 병목(~0.33 천장)이며 SNR/용량/specialization/라우팅 무엇도 못 넘음.**

### 8-2e. 사전등록 공정 검증: "큰 모델 = 노이즈 강건"인가 (H1)
사용자 가설 H1("용량↑ → 노이즈 강건 + 라우팅 용량대비 이득")을 공정하게 검증. **누설 없는 bearing-disjoint**, 큰 모델에 불리하지 않게 **train 데이터 5배↑(20k→100k, 가용 169,836 중)**, **40ep + best-val**, 용량 사다리 tiny~xl(1K~898K). 지표·판정규칙은 **실행 전 고정**(M1=노이즈{0,−5,−10} 평균, M2=유지율 noisy/clean, 규칙: M1·M2 모두 양·단조면 지지).

3-seed (42/1/7) 평균 7-SNR macro-F1:

| 모델 | params | s42 | s1 | s7 | mean±sd |
|---|---:|---:|---:|---:|---:|
| mid | 53K | 0.386 | 0.279 | 0.338 | 0.334±0.044 |
| base | 133K | 0.358 | 0.292 | 0.359 | 0.336±0.031 |
| large | 524K | 0.349 | 0.289 | 0.391 | 0.343±0.042 |
| xl | 898K | 0.458 | 0.267 | 0.326 | 0.350±0.080 |

- 평균 Spearman(cap↔avg/M1/M2)는 +1.0/+1.0/+0.8로 **표면상 H1 지지처럼 보이나 통계 착시**:
  1. 효과 크기(mid→xl = +0.016) ≪ 시드 노이즈(±0.03~0.08).
  2. xl "1등"은 **단일 시드(s42=0.458)** 산물 — 나머지 두 시드는 0.267(전체 꼴찌)/0.326. **s42 제외 시 xl이 최하(0.297)**로 추세 역전.
  3. n=4·스프레드 0.016·노이즈 0.08 → 단조 Spearman은 우연.
- **판정: H1 기각.** 용량 효과가 시드 노이즈보다 작아 mid≈base≈large≈xl ≈ 0.33~0.35로 사실상 동일. 강건성(M2) 무관, 절대성능(M1) 무관.
- 근본 한계: 테스트 베어링 5개 → 시드 오차 ±0.05, 그 안에서 용량 신호 없음. (데이터셋 한계상 오차 축소 불가.)

### 8-2f. 진단: chance-바닥의 정체 + IR-vs-OR + weighting (ADHD 비판 + codex 합의 후 검증)
"누설 닫으면 전부 chance(~0.33)"가 너무 강한 주장인지 검증. trivial-baseline macro-F1: A2R/bdis 모두 **stratified-random=0.333**, 우리 모델 0.31-0.42 → 표면상 chance. 혼동행렬·IR-vs-OR로 분해:

- **N(건강) 클래스가 붕괴**: A2R N-recall=0.2%, bdis 5-6% (건강 베어링 2개뿐 → 전이 가능한 정상 시그니처 학습 불가). 3-class macro-F1을 chance로 끌어내리는 주범.
- **IR-vs-OR 이진 신호는 실재**(chance 0.5 위): A2R 0.60, bdis 0.54-0.66. clean/0dB에서 0.6-0.69, −10dB에서 ~chance로 붕괴. → "전면 학습 실패"는 아님.
- **체크포인트 선택**: A2R best-val>last-epoch(+0.04), bdis last-epoch>best-val(+0.07) — train-베어링 val 선택이 ±0.05-0.07 교란(시드 노이즈급), 일관된 방향 없음.

**용량/라우팅을 IR-vs-OR로 재측정 (N 마스킹 제거, bdis 2seed):**

| 방법 | IR-vs-OR 7SNR (s42/s1) | 평균 |
|---|---:|---:|
| 단일 mid 53K | 0.603/0.476 | 0.540 |
| 단일 base 133K | 0.554/0.432 | 0.493 |
| 단일 large 524K | 0.618/0.404 | 0.511 |
| 단일 xl 898K | 0.519/0.467 | 0.493 |
| 라우팅(small/mid/base) | 0.533/0.463 | 0.498 |
| base **unweighted** CE | 0.602/0.440 | 0.521 |

- **용량 효과 없음**(비단조, 0.49-0.54, 시드 ±0.1 내). 라우팅 ≤ 단일. → **방법 효과는 N에 가려진 게 아니라 실제로 없음.**
- **weighting**: inverse-freq 가중을 쓰고 있었고(N 붕괴 원인 아님 — 데이터 한계). unweighted가 IR/OR +0.028 (전이 불가 N에 용량 낭비하던 미세 왜곡 제거) 하나 시드 노이즈 내, 구조 불변.

**8-2f 종합:** macro-F1 chance-바닥 = 전이 불가 N클래스의 평균효과. IR-vs-OR엔 약한 실신호 존재하나 **용량·라우팅·loss-weighting 어느 것도 개선 못 함.** "큰 모델 강건/라우팅 이득" 가설은 누설·N마스킹·가중치를 모두 통제한 뒤에도 미지지.

### 8-2g. full-aug generalist 라우팅 (원래 용량-라우팅 셋업) + 게이트 동일비교
"밴드-specialist 대신 full-aug로 학습한 모델들을 라우팅하는 게 더 맞는 셋업 아니냐"는 질문 검증. 각 tier(small/mid/base)를 **전 SNR aug로 학습**(generalist)한 뒤 노이즈 점수로 라우팅. (codex 합의 동반.)

| 분할 | 단일 base | route-oracle | route-estimator | route-HardGate | (참고 밴드-specialist) |
|---|---:|---:|---:|---:|---:|
| recording-level | 0.543 @133K | 0.528 @57K | 0.561 @64K | 0.551 @46K | 0.756 (누설) |
| bearing-disjoint | ~0.32–0.36 | 0.312 @57K | 0.346 @98K | 0.304 | 0.30–0.33 |

- **full-aug 라우팅 ≈ 단일 base** (정확도 이득 없음). 밴드-specialist의 +0.18은 full-aug로 바꾸니 증발 → 누설 아티팩트 확정.
- **codex 합의:** (a) full-aug 라우팅이 원래 용량-라우팅 가설에 맞는 정직한 셋업. (b) base가 전 SNR 최고면 SNR 결정 라우터는 단일 base를 수학적으로 못 이김(per-instance 상보성 없이는). (c) 라우팅의 정당한 이득은 정확도가 아니라 **정확도-비용 Pareto**(estimator가 base 정확도를 ~절반 active params로) — 비용 미보고 시 라우팅 주장 불완전. (d) 밴드-specialist는 "조건부 전문가 학습"으로 명명해야지 "compute-saving 라우팅"으로 부르면 안 됨.
- **결론:** SNRGate의 정직한 가치 제안 = "더 정확"이 아니라 "비슷한 정확도를 더 싸게". 단, 누설 없는 평가에선 그 정확도가 ~0.33 바닥이라 의미 약함.

### 8-2h. 추정기 축소(93K→7.6K)로 라우팅 비용 이점 복원
8-2g에서 CNN 체제 비용 이점이 사라진 원인은 상시 추정기(93K)였다. 추정기를 ~7.6K(channels 8,16,16,32)로 줄여 동일 자기지도 레시피로 재학습 → 라우팅 재측정.
- **7.6K 추정기 calibration 거의 완벽**(pred≈target, val MSE 0.0014) — 노이즈 추정은 쉬운 과제, 작은 모델로 충분.

| 추정기 | full-aug route-est (recording-level) | tier eff | +추정기 | 총 활성 |
|---|---:|---:|---:|---:|
| 93K | 0.561 | 64K | 93K | 157K (≥ base 133K) |
| **7.6K** | 0.582 | 17K | 7.6K | **~24K** (base 133K의 1/5) |

- 단일 base = 0.543 @133K. → **7.6K 추정기로 라우팅이 ≈base 정확도를 ~24K 활성**으로 냄 = "비슷한 정확도 @ ~5× 저렴" Pareto 이점 복원 (CNN 체제).
- bdis에선 7.6K 라우팅 = 0.349 @~45K ≈ base(~0.34) — 비용↓이나 정확도 ~chance라 의미 약함.
- ⚠️ 한계: (1) "더 정확"이 아니라 "≈정확도 더 싸게", (2) 그 정확도는 누설(recording-level) 산물, (3) 단일-run ±0.02~0.03 노이즈(다중시드 미확인).

### 8-2i. 이질적 노이즈(per-window 랜덤 SNR)에서의 라우팅 — 다양해져도 못 이김
표준 sweep은 "전체 test에 동일 SNR"이라 한 SNR 안 노이즈 점수가 거의 상수(std~0.01)→라우팅이 ~100% 한 tier로 몰림(고정/학습형 게이트 무차별). "현실은 window마다 노이즈가 달라 점수가 퍼지고 라우팅이 다양해질 것"이라는 지적 검증: **window마다 독립 랜덤 SNR** 주입 후 평가.

| 메소드 | recording-level | bdis |
|---|---:|---:|
| 단일 small / mid / base | 0.384 / **0.578** / 0.571 | 0.314 / 0.304 / **0.372** |
| 라우팅 oracle | 0.548 | 0.316 |
| 라우팅 estimator | **0.578** | 0.351 |
| 라우팅 HardGate(.33/.66) | 0.572 | 0.308 |

- **라우팅은 실제로 다양해짐**: 선택비율이 sweep의 100/0/0 → 이질적에선 43/29/28·28/44/28로 세 tier에 퍼짐(점수가 window마다 달라서). ✓ 지적 옳음.
- **그러나 단일 best를 못 이김**: recording-level 라우팅 best(est 0.578) = 단일 mid(0.578) 동률; bdis 라우팅 best(est 0.351) < 단일 base(0.372). 학습형 est ≈ HardGate 여전.
- **이유:** 어떤 노이즈 레벨에서도 작은 tier가 큰 tier를 능가하지 않음(상보성 부재) → 다양하게 나눠도 일부 window가 약한 tier로 가 손해.
- **결론:** 라우팅 전제는 sweep뿐 아니라 **이질적 노이즈(라우팅이 의미 있을 유일한 무대)에서도 미지지** — 근본 문제는 평가 방식이 아니라 **tier 간 상보성 부재**.

### 8-2j. Cost-matched routing (목표: large 정확도 ≈, 비용↓) — 부분 성공
목표(claim B): 라우팅이 always-large와 정확도 비슷하면서 평균 active 비용 절감. 후보 small/base/large clean-only, 노이즈 추정기 7.6K, 2-tier{base,large}·3-tier 임계값을 val에서 "비용 최소화 s.t. F1≥large−0.01"로 선택. noisy test(sweep+mixed), recording-level+bdis, 2 seed.

| split·seed | always-large (sweep) | 2-tier 라우팅 | 비용 |
|---|---:|---:|---:|
| indist s42 | 0.600 @524K | **0.608** @**180K** (2.9× ↓) | ✅ |
| indist s1 | 0.593 @524K | 0.578 @358K | △ −0.015 |
| indist 평균 | 0.597 @524K | **0.593 @269K** (~2× ↓) | ≈(노이즈 내) |
| bdis s42 | 0.303 | 0.239 @141K | ❌ |
| bdis s1 | 0.172 | 0.156 @141K | ❌ |

- **달성(조건부):** in-dist 7-SNR sweep에서 base→large 2-tier 라우팅이 always-large와 **동급(평균 −0.004, 시드노이즈 내)을 ~2배 싼 평균비용**으로. seed42는 1/3 비용에 +0.008로 명확.
- **메커니즘:** base가 clean~중간 SNR에서 large에 근접(clean 0.88 vs 0.90) → clean/중간은 base(133K), 고잡음만 large(524K) → 평균비용↓, 정확도 유지.
- **한계(정직):** (1) seed-불안정(s42 성공/s1 마진미달, 임계값 선택 분산), (2) **mixed-SNR test에선 −0.02로 미달**, (3) **bearing-disjoint 실패**(누설 없으면 base가 전 구간 약해 라우팅이 large 못 따라감), (4) 비용에 추정기 7.6K 상시 포함.
- **주장 유형:** 정확도 향상 아님 — **always-large 대비 "비슷한 정확도, 더 낮은 평균 비용"(claim B, in-dist sweep 한정).**

### 8-2k. MOMENT 파운데이션 tier에서의 cost-match (frozen + LoRA + head)
비용 절감이 의미 있는 무거운-tier 체제 검증. MOMENT-S/B/L(35M/110M/341M), frozen encoder + **LoRA(q/v, r=8, α=16)** + head, clean-only 학습, 노이즈 추정기(2048, 7.6K)로 라우팅, in-dist noisy test, seed42.

| 구성 | 7-SNR avg F1 | active | vs large |
|---|---:|---:|---|
| 단일 small | 0.396 | 35M | |
| 단일 base | 0.417 | 110M | |
| 단일 large | 0.423 | 341M | (천장) |
| 2-tier{base,large} 라우팅 | 0.422 | 133M | −0.001, 2.6× ↓ |
| 3-tier{small,base,large} | 0.417 | 110M | −0.006, 3.1× ↓ |

- **cost-match 성립:** 라우팅이 always-large와 ≈동급(−0.001)을 2.6× 싸게. 341M→133M은 절대량으로 의미 있는 절감(CNN의 수백K 반올림 절감과 대조).
- **단, 이유는 용량 포화:** small 0.396 ≈ base 0.417 ≈ large 0.423 (10× params에 차이 0.027). 라우팅이 large를 매칭하는 건 large가 별로 안 좋아서.
- **always-base가 라우팅을 dominate:** base 0.417@110M이 라우팅(0.422@133M)과 ≈동급을 더 싸게 → 라우팅 대신 "그냥 base"가 더 단순·저렴.
- **MOMENT 절대 정확도(0.42) < CNN(clean-only large 0.62)** — frozen MOMENT는 이 과제에 더 약함. LoRA(clean-only) 추가해도 향상 미미.
- **함의:** 파운데이션 체제 메시지 = "tier 용량 포화 → 작은 tier로 충분, 큰 모델 불필요"(라우팅이 영리해서가 아님). 메인 테이블은 CNN(용량=강건성 뚜렷)이 적합, MOMENT는 보조.

### 8-2l. 실제 A2R 데이터의 자연 노이즈 수준 — 노이즈 라우팅이 degenerate
"실제(real) 손상 데이터엔 합성 AWGN을 안 넣어도 자연 노이즈가 있으니, noise-tier 라우팅(clean→small, 고잡음→large)이 작동하지 않냐"는 질문 검증. 저장된 노이즈 추정기(7.6K)를 **실제 A2R test(9,997 윈도우)**에 적용. estimator score는 주입 SNR에 단조(낮을수록 clean).

**보정 곡선(A2R test에 합성 SNR 주입 시 estimator score 평균):** clean 0.045 / 20 dB 0.069 / 10 dB 0.248 / 5 dB 0.375 / 0 dB 0.500 / −5 dB 0.626 / −10 dB 0.711.

**실제 A2R(합성 미주입) score:** mean 0.045, std 0.020, median 0.041, p95 0.075.
- → 실제 데이터는 보정상 **clean 지점(0.045)에 정확히 위치**, p95조차 ~20 dB 수준. 실험실 수집이라 **매우 깨끗**(고-SNR).

**라우팅 비율(실제 A2R, ours τ):** small 71.7% / mid 27~28% / **large 0.2~1.0%**.
- **결론:** 실제 데이터에 자연 노이즈가 0은 아니나(약간의 분포 존재), 거의 전부 **clean 끝단**에 몰려 0/−5/−10 dB 영역과 멀다. 따라서 noise-tier 라우팅은 **사실상 degenerate**(거의 small로만 감, 고잡음 전문가 large는 ~1% 미만 선택). 즉 "real이라 노이즈가 있어 라우팅이 작동"은 이론상 가능하나 **PU에선 실효 없음**. A2R 병목은 노이즈가 아니라 **인공→실손상 도메인 갭**임을 재확인. 데이터: 추정기 `/tmp/small_est.pt`, A2R는 `materialise("a2r")`.

### 8-2m. 단일-large 천장을 넘으려는 성능-향상 기법들 (정직 검증, mixed-SNR 3-seed)
라우팅은 비용만 줄일 뿐 정확도로 large를 못 넘는다(상보성 부재). "정확도 자체를 올릴 정당한 방법이 있나?"를 사전 선언한 표준 기법들로 검증. 모두 동일 mixed-SNR test·저장모델·3-seed, 결과는 win/no-win 그대로 기록(seed cherry-pick 없음). 기준: 단일 large 0.624±0.028.

**① Tier ensemble (softmax 동일가중 soft-vote)** — 저장 small/mid/large 재사용, 학습 없음.

| 구성 | mixed-SNR F1 | 비용 | vs large |
|---|---:|---:|---|
| ensemble(small+mid+large) | 0.569 ± .042 | 586K | **−0.055** |
| ensemble(mid+large) | 0.586 ± .049 | 577K | **−0.038** |

- **no-win:** 앙상블이 단일 large보다 **나쁨**(비용은 더 큼). 이유는 §8-2e/8-3의 상보성 부재 — 노이즈에서 약한 small/mid가 평균을 끌어내려, 약한 멤버를 섞을수록 손해(전형적 "weak-member drag"). 앙상블 이득은 멤버들이 서로 보완할 때만 나오는데 여기선 large가 전 구간 단독 우세라 부재.

**② Pruning + fine-tune (글로벌 마그니튜드 30/50/70%, clean fine-tune 12ep)** — "프루닝이 불필요한 부분을 없애 성능을 올린다"는 가설 검증. 저장 large 로드→프루닝→fine-tune→mixed-SNR.

| 구성 | mixed-SNR F1 | nonzero | vs dense |
|---|---:|---:|---|
| dense large | 0.624 ± .028 | 524K | (기준) |
| pruned 30% + ft | 0.624 ± .024 | 365K | +0.001 |
| pruned 50% + ft | 0.625 ± .032 | 260K | +0.001 |
| pruned 70% + ft | 0.630 ± .027 | 156K | +0.006 |

- **교란 통제(control):** pruned 모델은 12ep 추가 fine-tune을 받음 → 향상이 프루닝인지 추가학습인지 분리 필요. **dense large + 동일 12ep ft(프루닝 없음)** = 0.623±.029 (−0.001). 시드별 패턴이 pruned와 동일(s42 +0.035 / s1 −0.031 / s7 −0.006). → **±0.03 시드 출렁임은 프루닝이 아니라 추가학습/체크포인트 재선택 노이즈**.
- **no-win(정확도):** 프루닝은 dense 대비 정확도 향상 없음(평균 +0.006, std 이내, control 보정 후 ~0). 사용자 가설("프루닝이 성능 향상")은 이 과제에서 **미지지**.
- **그러나 의미있는 발견 = 중복성:** 70% 프루닝(nonzero 156K)에도 F1 유지 → large는 **중복 용량**을 가짐. 단, 처음부터 9K로 학습한 small은 0.49로 붕괴. **즉 "중복 용량"이 낭비가 아니라 노이즈 강건성을 사는 비용**(capacity 스토리 강화). 그림 `fig_pruning.png`.
- **압축 프론티어(고sparsity, unstructured magnitude, 3-seed):** dense 0.624 → 70%(156K) 0.624 / **80%(104K) 0.618(5×, −0.006)** / **90%(52K) 0.606(10×, −0.018)** / 95%(26K) 0.542(붕괴). **같은 ~52K 예산에서 pruned-large 0.606 ≫ from-scratch mid 0.547(+0.059)** → "train-big-deploy-small". 그림 `fig_compress.png`, 데이터 `imp_prune_hi_s*.csv`.
- **Structured(channel) pruning — 실제 MACs 감소(배포 가능, 3-seed):** ln_structured(L2, dim=0)로 출력 채널 통째 제거 → 실연산 감소. dense 9.96M MACs / 0.624. **30% 채널: 0.621 @ 5.00M MACs(2× 감소, −0.003 유지)** / 50%: 0.587 @ 2.62M(3.8×, −0.037) / 70%: 0.543 @ 1.00M(10×, 붕괴). **→ structured 2× MACs 절감에서 large 정확도 유지** = unstructured(파라미터/크기 압축)와 달리 **하드웨어 무관 실 latency 이득**. 데이터 `exp_structured_s*.csv`.
- **Hybrid(structured+unstructured 동시, 3-seed):** structured 30% 채널 → 남은 가중치에 unstructured magnitude. MACs는 structured대로 5.0M(2×) 고정 + 파라미터 추가 압축. **+unstr50%: 0.603 @ 2× MACs + 182K** / +unstr70%: 0.587 @ 2× MACs + 109K. → 두 압축축을 **동시에 쌓을 수 있으나** 정확도 손실이 ~가산적(structured-only 0.621 > hybrid 0.603 > 0.587). 용도별: 정확도 유지=structured-only 30%, MACs+크기 동시압축=hybrid u50. 데이터 `exp_hybrid_s*.csv`.

**③ Noise-augmented training of large (AWGN 증강 학습, clean 포함 SNR 범위)** — 표준 노이즈-강건성 기법. *프로토콜 변경*(clean-only 아님): 모델이 학습 중 노이즈를 봄.

| 구성 (체크포인트 선택) | mixed-SNR F1 | vs clean-large |
|---|---:|---|
| clean-trained large (기준) | 0.624 ± .028 | — |
| noise-aug large (**clean-val 선택**) | 0.633 ± .032 | +0.009 (std 이내; s42 .657/s1 .588/s7 .654) |

- **clean-val 선택은 marginal:** 평균 +0.009로 시드 노이즈 이내, seed1은 오히려 하락. 원인 가설: 학습은 noise인데 **체크포인트를 clean val로 선택**해 배포(노이즈) 분포와 불일치 → 핸디캡.

**③b Noise-aug training + noisy-val 체크포인트 선택 (표준-정확 프로토콜)** — val을 mixed-noise(seed 333, test 222와 분리)로 만들어 배포 분포에 맞춰 체크포인트 선택. ③의 핸디캡 교정. 사전 선언 1회 변형(fishing 아님).

| 구성 | mixed-SNR F1 | vs clean-large | per-seed (42/1/7) |
|---|---:|---|---|
| clean-trained large (기준) | 0.624 ± .028 | — | .594/.616/.661 |
| noise-aug + clean-val | 0.633 ± .032 | +0.009 | .657/.588/.654 |
| **noise-aug + noisy-val** | **0.651 ± .007** | **+0.027** | .657/.640/.655 |

- **✅ WIN(정직한 positive):** noisy-val 선택 noise-aug가 clean-large를 **+0.027** 능가, 2/3 시드 명확 향상(s42 +0.063, s1 +0.024)·1개 무승부(s7 −0.006). 게다가 **시드 분산을 1/4로 축소**(±0.028→±0.007) = 안정성↑.
- **이 과제에서 정확도를 실제로 올린 유일한 기법.** 앙상블·프루닝·용량 라우팅(비용만↓)과 대조.
- **정직한 caveat:** (1) **프로토콜 변경** — 모델이 학습 중 노이즈를 봄(clean-only 아님). (2) 배포 노이즈를 알/시뮬레이션 가능해야 하며, val도 그 분포로 선택해야 효과(clean-val 선택은 +0.009로 약함). (3) test 윈도우와 val 노이즈 seed를 분리(222 vs 333)해 누설 없음. 저장모델 `cnn_large_noiseaug_nv_s*.pt`, 데이터 `imp_noiseaug_nv.csv`.

**종합 (성능-향상 기법, mixed-SNR 3-seed, clean-large 0.624 기준):**

| 기법 | F1 | Δ vs large | 정확도 향상? |
|---|---:|---:|---|
| Tier ensemble (s+m+l) | 0.569 ± .042 | −0.055 | ❌ (weak-member drag) |
| Pruning 70% + ft | 0.630 ± .027 | +0.006 | ❌ (control=노이즈; 중복성만 확인) |
| Noise-aug (clean-val) | 0.633 ± .032 | +0.009 | △ (std 이내) |
| **Noise-aug (noisy-val)** | **0.651 ± .007** | **+0.027** | ✅ (일관·분산↓) |

- **결론:** 추론-시점 기법(앙상블·프루닝·용량 라우팅)은 단일 large의 정확도 천장을 못 넘음(상보성 부재 재확인). 정확도를 실제로 올리려면 **학습 분포에 노이즈를 넣어야** 함(noise-aug, 배포-정합 val 선택) → +0.027 & 분산↓. 단 이는 clean-only 프로토콜을 벗어남. 그림 `fig_improve.png`.

### 8-2n. 라우터를 다르게 — oracle headroom은 있으나 값싼 신호로 포착 불가 (GPU0/1 병렬, 3-seed)
"라우팅 방식을 바꾸면 성능/비용 이득을 더 볼 수 있나?"를 codex 합의 하에 검증. 먼저 **oracle 감사**(저장 tier logit, 학습 0)로 go/no-go:

| 지표 (mixed-SNR, 3-seed) | 값 |
|---|---|
| always-large macro-F1 | 0.624 ± .028 |
| **oracle (any-tier-correct) F1** | **0.736 ± .055 (+0.112 상한)** |
| oracle accuracy − large accuracy | +0.080 |
| HEADROOM (large 틀림 & small/mid 맞음) | **8.0%** (0~5 dB 집중: +0.149/+0.109) |

→ **상보성은 실재**(이전 "상보성 부재"는 *SNR 라우터* 한정 결론이었음). headroom이 있으니 더 똑똑한 라우터를 시도할 근거 → 두 기법 검증:

| 라우터 | F1 | 비용 | vs large |
|---|---:|---:|---|
| SNR 추정 라우팅(기존) | 0.611 | 398K | −0.013 |
| **불일치/마진 캐스케이드** (small+mid→large) | 0.584 ± .046 | 366K | −0.040 |
| **공유 backbone early-exit** (multi-exit 1망) | 0.563 ± .013 | 127K | −0.061 |
| └ early-exit deep만 (참고) | 0.627 ± .015 | 388K | +0.003 (large 동급, 1.35× 싸게) |

- **정확도 이득 없음:** 시도한 3개 라우터 신호(SNR / margin·불일치 / shared-backbone exit) **모두 large를 못 넘고, 캐스케이드는 기존 SNR 라우팅보다도 낮음.** 노이즈에서 얕은 모델이 **틀린 답에 확신하며 합의** → 조기 채택이 정확도를 깎음.
- **결정적 진단:** **상보성(+0.112)은 존재하나 어떤 값싼 신호(SNR·confidence·margin)와도 정렬되지 않음** → 실용 라우터가 8% headroom을 못 집음(라벨 oracle만 가능). 이는 "상보성 부재"보다 정밀한 결론: *라우팅이 정확도를 못 올리는 건 상보성이 없어서가 아니라, 있는 상보성을 입력만으로 식별할 수 없어서.*
- **비용 측면:** early-exit deep exit가 large 동급을 1.35× 싸게(388K), 캐스케이드는 4× 싸게(127K)지만 −0.06 손실. 기존 SNR 라우팅(0.611@398K)이 여전히 large-근접 최고 비용점.
- **codex 합의 결론(가장 방어적):** "노이즈/난이도 라우팅은 정확도 상보성을 *포착*하지 못한다; 조건부 연산은 비용만 절감; robust 학습(noise-aug)·압축이 라우터 복잡도보다 강하다." 데이터: oracle_audit.csv, exp_cascade.csv, exp_earlyexit.csv. MoE는 소데이터 붕괴·p-hacking 위험으로 양측 비추 → 미수행.

**압축 tier 위에서의 라우팅 (structured-pruned MACs-tier {1.0M/5.0M/9.96M} 노이즈 추정 라우팅, 3-seed):** 라우팅이 단일 압축 모델을 MACs 프론티어에서 넘는지 검증.

| 구성 | F1 | 평균 MACs |
|---|---:|---:|
| cheap(struct70%) | 0.533 | 1.0M |
| **mid(struct30%) 단일** | **0.621** | **5.0M** |
| big(dense) | 0.624 | 9.96M |
| 라우팅 accmax | 0.626 | 6.17M |
| 라우팅 cost-match | 0.609 | 5.42M |
| oracle(상한, acc) | 0.813 | — |

- **단일 structured-30%가 라우팅을 dominate:** 라우팅 accmax(0.626)는 struct30%(0.621)와 정확도 동급이나 **MACs 더 큼**(6.17M vs 5.0M); cost-match(0.609@5.42M)는 양축 모두 밀림. oracle 0.813로 상보성 여지는 또 있으나 추정기 미포착(일관).
- **결론:** 압축 tier로 라우팅해도 **단일 압축 모델보다 나을 게 없음** → "train-big-deploy-small"은 **라우팅 없이 단일 structured-pruned 모델로 최선**. 라우팅은 복잡도+추정기 오버헤드만 추가. 데이터 `exp_route_compressed_s*.csv`.

### 8-2o. 압축 하이퍼파라미터 sweep + quality-weighted pruning (3-seed)
train-big-deploy-small을 다각도 검증. 저장 large + fine-tune, structured=실 MACs, unstructured=파라미터.

**(a) Structured 미세 프론티어(clean FT, 채널 L2):** dense 0.624@9.96M → 10% 0.632@8.10M / 20% 0.614@6.45M / 30% 0.621@5.00M / 40% 0.618@3.72M / 50% 0.587@2.62M / 70% 0.543@1.00M. **40%(2.7× MACs)까지 ~0.62 유지**, 이후 하락.

**(b) Random vs Magnitude (control, unstructured, 3-seed):** magnitude가 random을 명확히 능가 — 70%: mag 0.620 vs rand 0.565(+0.055) / 90%: mag 0.606 vs rand 0.576(+0.030). → **weighted(magnitude) 기준이 실제로 중요**(fine-tune만으로 되는 게 아님). 중요도-기반 프루닝의 전제 검증.

**(c) Prune + noise-aug fine-tune(saug):** 압축+강건학습 결합. struct30%: clean-FT 0.621 vs noiseaug-FT 0.594(−0.027) / struct50%: 0.587 vs 0.608(+0.021). **비일관(시드 노이즈 큼)** — 짧은 12ep FT엔 noise-aug 이득 약함(from-scratch 50ep noise-aug의 +0.027과 대조).

**(d) ★Quality-weighted pruning:** quality estimator의 noise score로 채널 중요도 가중(노이즈 윈도에서 활성화되는 채널 보존). 같은 sparsity=같은 MACs, "어떤 채널 남기나"만 차이.

| 채널중요도 | struct30%(5.0M) | struct50%(2.62M) |
|---|---:|---:|
| mag (L2) | 0.626 | **0.611** |
| act (균등 활성) | 0.612 | 0.586 |
| **qweight (quality)** | **0.636** | 0.572 |
| (dense large 참조) | 0.624 | 0.624 |

- **저sparsity(30%)에서 qweight 최고:** 0.636 @ 5.0M MACs(2× 압축), mag(0.626) +0.010·dense(0.624) 상회, 3/3 중 2시드 우위 → **전 압축 실험 통틀어 최고 단일 배포점**. 단 +0.010은 시드 std(~0.03) 이내 → "동급+α, 최고 단일점"으로 표현.
- **고sparsity(50%)에선 역전:** qweight 0.572 < mag 0.611(−0.039). 강한 프루닝 시 quality 가중이 다수(clean) 채널을 과제거. → quality-weighted pruning은 **저sparsity 한정 유효**.
- 데이터: `exp_sweep_{sfine,rand,saug}_s*.csv`, `exp_quality_prune_s*.csv`.

**(e) qweight 저sparsity 세밀 sweep(10~40%, 3-seed) + 성능유지 라우팅:**

| sparsity(MACs) | mag | qweight | qw−mag |
|---|---:|---:|---:|
| 10%(8.10M) | 0.636 | 0.645 | +0.009 |
| 20%(6.45M) | 0.646 | 0.646 | +0.000 |
| 30%(5.00M) | 0.625 | 0.628 | +0.003 |
| 40%(3.72M) | 0.605 | 0.611 | +0.006 |

- **qweight가 10~40% 전 구간 mag 이상(never worse), +0.000~+0.009 일관 소폭 우위** — 단 시드 std 이내라 "동급+α". qw20%=0.646@6.45M이 dense large(0.624)를 1.5× 적은 MACs로 상회.
- **성능유지 라우팅(qw tier {dense,qw20,qw40}, val F1≥dense−0.01서 MACs 최소):** 3-seed 0.611@3.72M vs dense 0.609@9.96M → 정확도 유지·2.7× MACs↓. **그러나 τ=(0.8,0.85)로 ~100% 윈도를 qw40로 보냄 = 단일 qw40 모델 그 자체** → 라우팅의 단일-압축 대비 추가 이득 0(일관). 데이터 `exp_qweight_route_s*.csv`.
- **종합 배포 권장:** qweight-pruned 단일 모델 — 40%(0.611@3.72M, 2.7× MACs) 또는 20%(0.646@6.45M, dense 상회 @1.5×). 라우팅 불필요.

### 8-2p. quality-weighted-tier 라우팅 "win"의 정체 = BatchNorm 평가 아티팩트 (철회)
qweight-pruned tier {dense,qw20,qw40}를 quality score로 라우팅하면 mixed-SNR에서 단일을 +0.05 능가하는 듯 보였고(accmax 0.698 vs single 0.646, val≈test) per-SNR에선 0dB 0.73(단일 large eval 0.46)로 극적. **그러나 디버그 결과 dense tier가 `.eval()` 누락으로 train 모드 추론 → BatchNorm이 test 배치 통계로 정규화(test-time BN adaptation)된 아티팩트**였음. 라우팅이 노이즈를 그 BN-adapted dense로 보내 높게 나온 것 — 라우팅·qweight 공로 아님.

**Test-time BN adaptation 정밀 검증 (단일 large, eval vs train-mode BN, 3-seed):**

| SNR | eval | BN-adapt(동질배치 bs256) | BN-adapt(bs32) |
|---|---:|---:|---:|
| 0 dB | 0.460 | 0.732 (+0.272) | 0.701 |
| −10 dB | 0.191 | 0.404 (+0.213) | 0.400 |
| 7-SNR avg | 0.611 | 0.741 (+0.130) | 0.722 |

- **그러나 동질-SNR 배치 아티팩트:** 위는 한 배치가 전부 같은 SNR이라 BN이 그 레벨에 완벽 적응한 경우. **현실(per-window 혼합 SNR 배치)에선 BN-adapt가 오히려 손해:** mixed-SNR 3-seed eval 0.624 vs BN256 0.609(−0.014) vs BN32 0.591(−0.033).
- **결론:** 라우팅·qweight-tier·BN-adapt 모두 **현실(혼합 노이즈) 평가에서 단일 large(eval)를 못 넘음.** 앞선 negative 라우팅 결과 유지. 캐노니컬 large=0.624(eval)도 재확인.
- **방법론적 경고(부수 기여):** **per-SNR 동질배치 + BatchNorm(train 모드 또는 대배치) 평가는 노이즈 강건성을 크게 부풀린다**(0dB +0.27). 노이즈-강건성 벤치마크는 (a) BN을 eval 모드로 고정, (b) 혼합-SNR/소배치 평가를 병행해야 함. 데이터 `exp_bn_adapt.py`, `exp_bn_mixed.py`, `exp_qweight_*.py`.

**(정정·버그제거) qw20-anchored 라우팅 (모든 tier .eval() 강제, mixed-SNR, 3-seed):** BN 버그 제거 후 재실험. tier sanity 통과(qw20 0.635/qw40 0.611/dense 0.624 = 캐노니컬 일치).

| 구성 | F1 | 평균 MACs |
|---|---:|---:|
| qw40 단일 | 0.611 | 3.72M |
| qw20 단일 | 0.635 | 6.45M |
| **qw20+라우팅(iso)** | **0.634** | **5.26M** |
| dense | 0.624 | 9.96M |

- **작지만 진짜 비용 win:** 라우팅이 qw20 정확도 유지(0.634 vs 0.635, −0.001 시드노이즈 내)하며 **평균 MACs 1.23× 절감**(6.45→5.26M). 쉬운 윈도→qw40, 어려운→qw20(dense 미사용, 실질 2-tier). 정확도 향상 아님(유지), frontier 상 중간점일 뿐 dominate 아님. 이전 BN-버그 "+0.05"와 달리 버그 없는 결과. 데이터 `exp_qw20_route_s*.csv`.

### 8-2q. Dynamic pruning-ratio (quality score → sparsity 동적 선택, 3-seed)
사용자 아이디어: quality estimator score로 **윈도별 프루닝 비율을 동적 결정**. sparsity 사다리 {70/50/30/0%}(qweight, MACs 1.0/2.6/5.0/9.96M)를 q로 라우팅(clean→고sparsity 싼 tier, 노이즈→dense). eval 강제·iso 보정(best single −0.005, min MACs).

| | F1 | 평균 MACs |
|---|---:|---:|
| best single (qw30%) | 0.636 | 5.0M |
| **dynamic-sparsity 라우팅** | 0.617 | 5.51M |

- per-seed: 42) 0.642@4.23M(−0.002) / 7) 0.624@6.36M(−0.001) / 1) 0.584@5.95M(−0.055, val 임계값 선택 실패).
- **단일 못 넘음(−0.019, 비용도 안 쌈).** 라우팅류(SNR/캐스케이드/early-exit/압축-tier/qweight-tier/dynamic-sparsity) **6종 모두 현실 평가서 단일 압축 모델 못 넘음** 재확인. 데이터 `qsparsity_s*.csv`, 스크립트 `exp_qsparsity.py`.

### 8-3. 종합 결론
이 과제(노이즈 + Paderborn 진단)에서 **모델 스케일·용량 라우팅은 단일 적정모델(base)을 능가하지 못한다.** SNRGate의 "노이즈에 따라 다른 용량으로 라우팅" 전제는 재현 가능한 실험에서 지지되지 않음. 정직한 기여는 (a) 재현성 검증, (b) 용량-강건성 비단조성/포화, (c) A2R 도메인 갭이 진짜 병목.

## 부록: 실행 환경·검증 메모
- 버전 일치 env: Python 3.11, torch 2.5.1+cu121, transformers 4.33.3, numpy 1.26.4.
- 모든 학습은 seed 고정, best-on-val checkpoint 복원 사용.
- 데이터: 25개 A2R 베어링(KA08의 손상 .mat 1개 격리), n_test≈9,997.
