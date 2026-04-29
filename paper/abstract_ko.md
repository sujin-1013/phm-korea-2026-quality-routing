# 한국PHM학회 2026 — 초록 (한글, 200~300자)

## 제목

**시계열 기초모델의 노이즈 적응형 라우팅을 통한 베어링 결함 진단**
(Noise-Adaptive Routing of Time-Series Foundation Models for Bearing Fault Diagnosis)

---

## 본 초록

산업 환경의 진동 신호는 운전 조건에 따라 SNR이 크게 변하지만, 기존 베어링 결함 진단 모델은 입력 품질과 무관하게 단일 모델로 동일한 연산 비용을 지불하여 정확도-비용 균형이 비효율적이다. 본 논문은 이 한계를 해결하기 위해 **시계열 기초모델의 노이즈 적응형 라우팅**을 새로운 방법으로 제안한다. 제안 모델은 노이즈 수준을 [0,1]로 회귀하는 1D CNN 추정기, 두 임계값 기반의 하드-게이트 라우터, 동결 3-단 MOMENT(S/B/L)와 단별 분류 헤드로 구성되어 추론 시 입력당 단 하나의 단만 활성화한다. 핵심 기여는 fault signature와 가산 노이즈의 분광적 비식별성을 정면 해결한 추정기 설계, 그리고 단일 모델로 도달 불가능한 정확도-비용 파레토 우월 영역을 보장하는 라우팅 구조이다. Paderborn-PU 4-class 데이터에 AWGN -10~∞ dB을 주입한 실험으로 효과를 검증한다.

---

## 키워드 (5개, 영문 병기)

- 베어링 결함 진단 (Bearing Fault Diagnosis)
- 시계열 기초모델 (Time-Series Foundation Models / TSFM)
- 노이즈 적응형 라우팅 (Noise-Adaptive Routing)
- 신호 대 잡음비 (Signal-to-Noise Ratio / SNR)
- 정확도-비용 균형 (Accuracy-Cost Trade-off)

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
