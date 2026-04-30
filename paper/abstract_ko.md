# 한국PHM학회 2026 — 초록 (한글, 200~300자)

## 제목

**SNRGate: SNR 기반 동적 시계열 모델 라우팅을 통한 베어링 결함 진단**
(SNRGate: SNR-Aware Dynamic Routing of Time-Series Models for Bearing Fault Diagnosis)

---

## 본 초록

초록: 산업 환경의 베어링 진동 신호는 부하, 회전 속도, 측정 환경, 센서 잡음 등에 따라 입력 품질이 달라지며, 이는 신호 대 잡음비(Signal-to-Noise Ratio, SNR)의 차이로 나타날 수 있다. 기존 결함 진단 모델은 입력 품질과 관계없이 동일한 모델을 사용하므로, 경량 모델은 저품질 신호에서 성능 저하가 발생할 수 있고 대형 모델은 고품질 신호에서도 불필요한 연산을 수행한다. 본 논문은 입력 신호의 품질을 추정하고, 그 결과에 따라 서로 다른 연산 비용과 표현력을 갖는 시계열 모델을 동적으로 선택하는 SNRGate를 제안한다. SNRGate는 입력 신호의 SNR 수준을 정규화된 점수로 추정하는 입력 품질 추정기, 후보 모델의 선택 확률을 산출하는 SNR 인지 게이트, 서로 다른 연산 비용과 표현력을 갖는 복수의 시계열 모델, 그리고 모델별 분류 헤드로 구성된다. 학습 단계에서는 soft gate로 각 모델의 출력을 미분 가능하게 결합하고, 추론 단계에서는 가장 높은 게이트 점수를 갖는 단일 모델만 실행하여 샘플별 연산 비용을 조절한다. 또한 결함 특징과 노이즈 성분이 주파수 영역에서 혼재될 수 있는 문제를 고려하여, SNR 추정 손실과 결함 분류 손실을 함께 최적화한다. Paderborn University Bearing Dataset 기반 4-class 베어링 결함 진단 태스크에서 노이즈 주입을 통해 다양한 SNR 조건을 구성하고, 제안 방법을 단일 모델 및 고정 임계값 기반 라우팅과 비교하여 정확도와 연산 비용 간 균형을 평가한다.
---

## 키워드 (5개, 영문 병기)

- 베어링 결함 진단 (Bearing Fault Diagnosis)
- 시계열 모델 (Time-Series Foundation Models / TSFM)
- SNR 인지 게이팅 (SNR-Aware Gating)
- 신호 대 잡음비 (Signal-to-Noise Ratio / SNR)
- 정확도-비용 균형 (Accuracy-Cost Trade-off)

Bearing Fault Diagnosis, Dynamic Model Routing, SNR-Aware Gating, Signal-to-Noise Ratio, Computational Efficiency

---

## 작성 메모

- **톤 (v2)**: "분석·실험 보고" 톤 → "새로운 메소드·모델 제안" 톤. 첫 문장 제안 방법 명시 → 3-구성 요소 → 핵심 기여 → 평가 검증.
- **모델 파라미터 (실측)**: 35M/110M/341M (MOMENT-S/B/L). plan 추정치 40M/125M/385M와 차이는 momentfm의 attention head 포함 여부 차이. 본문에서는 모델 이름만 쓰고 수치는 생략 (분량 절약).
- **수치 placeholder**: Phase 1 sanity 통과 후 마지막 문장에 정량 결과 삽입 예정 — "MOMENT-L 단독 대비 FLOPs ~X% 절감, -5dB F1 0.X 보존".
- **Fallback 시나리오**: sanity가 실패하면 핵심 기여 두 줄 중 「추정기 설계」를 메인으로 격상한 버전 별도 작성.

## 수정 이력

- 2026-04-29 v1: 초안 (sanity placeholder)
- 2026-04-29 v2: 톤 변경 (메소드 제안), 키워드 영문 병기, 파라미터 정정
- 2026-04-29 v3: 자수 압축 (200~300자 적합), 키워드 separator 정리
- 2026-04-29 v4: 메소드 명명 (SNRGate), 라우터 격상 (rule → learnable temperature-annealed gate), 백본 무관 framework 포지셔닝, 키워드 1개 교체 (라우팅 → SNR 인지 게이팅)
- 2026-04-29 v5: "기초모델" → "시계열 모델"로 한국어 자연화 (영문 TSFM은 유지), LoRA Phase 3 → Phase 1 격상으로 단별 저랭크 적응 명시, MOMENT는 메소드 정의에서 빼고 실험 instantiation으로만 언급 (backbone-agnostic 강화)
- 2026-04-29 v6: contribution claim 재정리. Li et al. 2025 (arXiv:2511.23177)이 MOMENT를 모터 fault diagnosis에 이미 적용했음을 확인 → "MOMENT 적용 첫 시도" 주장 폐기. 대신 "베어링 진단에서 입력 SNR 적응 다중 단 라우팅을 처음 도입"으로 contribution (ii) 강화. manuscript Related Work 차별화 단락은 paper/related_work_draft.md에 별도 작성.
- 2026-04-29 v7: abstract 주장과 일대일 매칭되는 핵심 인용 5개를 본 파일 하단에 정리 (추후 manuscript 작성 시 그대로 차용).

---

## 핵심 인용 (5개, 제출 전 필수)

abstract의 각 주장에 일대일 매칭되어 미인용 시 reviewer 방어가 불가능한 reference. manuscript 본문 Related Work에서 추가 참고문헌은 별도로 늘어남.

### [1] Lessmeier et al. (2016) — Paderborn-PU 데이터셋

> Lessmeier, C., Kimotho, J. K., Zimmer, D., & Sextro, W. (2016). Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set. **PHM Society European Conference (PHME 2016)**.

- **매칭**: "Paderborn University Bearing Dataset 기반 4-class 베어링 결함 진단 태스크"
- **필수성**: 데이터셋 라이센스가 인용을 명시 요구. 미인용 시 학회 제출 결격 위험.

### [2] Goswami et al. (2024) — MOMENT (TSFM 백본)

> Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024). MOMENT: A Family of Open Time-Series Foundation Models. **ICML 2024**. arXiv:2402.03885.

- **매칭**: "복수의 시계열 모델"의 실험 instantiation
- **필수성**: 실험에서 MOMENT-S/B/L을 backbone으로 사용 → manuscript Methods에서 필수. abstract에 모델명을 노출 안 해도 본문 인용은 불가피.

### [3] Li et al. (2025) — MOMENT의 회전기계 적용 직접 prior art

> Li, et al. (2025). Data-Efficient Motor Condition Monitoring with Time Series Foundation Models. **arXiv:2511.23177**.

- **매칭**: TSFM을 회전기계 진단에 적용한 선행 연구 (SNRGate가 차별화해야 할 가장 가까운 작품)
- **필수성**: MOMENT를 모터 fault diagnosis에 이미 적용. 미인용 시 "이게 첫 적용이냐?" reviewer 질문 방어 불가. paper/related_work_draft.md에 차별화 단락 초안 있음.

### [4] Antoni & Randall (2006) — Spectral Kurtosis (fault/noise 분리)

> Antoni, J. & Randall, R. B. (2006). The spectral kurtosis: application to the vibratory surveillance and diagnostics of rotating machines. **Mechanical Systems and Signal Processing**, 20(2), 308–331.

- **매칭**: "결함 특징과 노이즈 성분이 주파수 영역에서 혼재될 수 있는 문제"
- **필수성**: 베어링 분야 fault/noise 분광 비식별성 주장의 유일한 학술적 anchor. 미인용 시 도메인 전문성 결여로 보임.

### [5] Shazeer et al. (2017) — Sparsely-Gated Mixture-of-Experts

> Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. **ICLR 2017**. arXiv:1701.06538.

- **매칭**: "후보 모델의 선택 확률을 산출하는 SNR 인지 게이트 + 가장 높은 게이트 점수를 갖는 단일 모델만 실행"
- **필수성**: SNRGate의 게이팅 메커니즘은 sparse MoE의 직계 후손. 미인용 시 "MoE와 차이?" 질문 방어 불가. SNRGate 차별점 = "input SNR이 gate input"임을 manuscript에서 명시 필요.

---

### 인용 분포 도식

| abstract 주장 | 인용 |
|---|---|
| Paderborn 데이터 | [1] Lessmeier 2016 |
| 시계열 모델(MOMENT 인스턴스화) | [2] Goswami 2024 |
| 회전기계 TSFM 적용 차별화 | [3] Li 2025 |
| fault/noise 주파수 혼재 | [4] Antoni & Randall 2006 |
| 게이트 점수 + 단일 모델 실행 | [5] Shazeer 2017 |
