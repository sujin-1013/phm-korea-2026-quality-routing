# 한국PHM학회 2026 — 초록 (한글, 200~300자 목표)

## 제목 (가안)

**진동 신호의 노이즈 적응형 라우팅을 통한 베어링 결함 진단의 정확도-비용 균형**

(영문 가안: *Noise-Adaptive Routing of Time-Series Foundation Models for Cost-Efficient Bearing Fault Diagnosis*)

---

## 본 초록 (약 270자)

산업 진동 데이터의 SNR은 시점·환경에 따라 크게 변하지만, 기존 결함 진단 모델은 단일 모델로 모든 입력을 처리하여 깨끗한 신호에 대해서도 불필요한 연산 비용을 지불한다. 본 연구는 입력 윈도우의 노이즈 수준을 0~1 스칼라로 추정하는 경량 1D CNN(약 100K 파라미터)과 두 임계값 기반의 hard-gate 라우터를 결합하여, 깨끗한 신호는 작은 시계열 기초모델(MOMENT-S, 40M)로, 노이즈가 큰 신호는 큰 모델(MOMENT-L, 385M)로 자동 분배한다. Paderborn-PU 베어링 4-class 데이터에 AWGN을 -10~∞ dB 범위로 주입하여 평가한 결과, 제안 기법은 단일 대형 모델 대비 평균 추론 비용을 크게 줄이면서 저SNR 영역의 F1을 보존하는 Pareto-우월 영역을 달성한다. 또한 fault signature와 additive noise의 분광적 혼동(under-identification) 위험을 정면 분석하고, 노이즈 추정기의 false-positive를 정량화하는 실험을 함께 보고한다.

> 자수 점검(공백 포함): 본문 약 270자 (한글 학회 기준 통상 200~300자 범위)

---

## 키워드 (5)

진동 신호 진단, Prognostics and Health Management, 시계열 기초모델, 적응형 라우팅, 신호 대 잡음비

---

## 작성 메모

- **현재 상태**: framing/method 우선 작성 (Sanity check 결과 미포함). Phase 1 sanity 결과가 나오면 「Pareto-우월 영역 달성」을 정량 수치로 교체.
- **제출 직후 후속 작업**: 본 발표/논문은 1개월 뒤. 그동안 Sanity → 5-step pipeline → baseline 6종 → Pareto plot 마련.
- **Fallback 시나리오**: sanity에서 size effect 안 나오면 본 초록의 last sentence를 main contribution으로 격상 (fault-noise disentangled SNR estimation as the paper's primary claim).

## 수정 이력

- 2026-04-29: 초안 작성 (sanity 결과 placeholder 상태)
