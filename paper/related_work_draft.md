# Related Work — Differentiation Drafts (Korean)

본 문서는 SNRGate manuscript의 Related Work 섹션에 들어갈 차별화 단락 초안을 모아둔다. 검증 완료된 prior art와의 비교 문장을 미리 작성해두어 manuscript 작성 시 직접 인용 가능.

---

## 1. Li et al. (2025) — TSFM을 모터 fault diagnosis에 적용

**Reference**: Li et al. (2025). *Data-Efficient Motor Condition Monitoring with Time Series Foundation Models*. arXiv:2511.23177 (제출 2025-11-28).

**핵심 내용**: MOMENT와 Mantis를 모터 fault diagnosis에 적용. 1% labeled data만으로 기존 DL 대비 약 2배 성능 달성. Transfer learning 기반 fine-tuning.

**SNRGate와의 차별화 (manuscript 본문 초안)**:

> 시계열 기초모델을 회전기계 진단에 적용한 선행 연구로 Li et al. (2025)는 MOMENT를 모터 fault diagnosis에 fine-tune하여 적은 라벨 환경에서의 데이터 효율성을 입증하였다. 본 연구는 이러한 backbone 적합성 결과를 기반으로 출발하되, 입력 신호의 noise 수준에 무관하게 단일 모델만 사용하는 기존 접근의 한계를 정면 해결한다. SNRGate는 (i) 운전 환경별 SNR 변동성을 명시적 회귀 신호로 노출하고, (ii) 학습 가능한 게이트로 backbone 크기 자체를 입력 SNR에 적응시켜 정확도-비용 파레토 우월 영역을 달성한다. 이는 단일 backbone fine-tuning이 다루지 못하는 cost-aware 적응 차원을 새로 도입한 것이다.

---

## 2. BearingFM (2024) — 베어링 전용 Foundation Model

**Reference**: BearingFM (2024). *BearingFM: Towards a foundation model for bearing fault diagnosis by domain knowledge and contrastive learning*. International Journal of Production Economics.

**핵심 내용**: 베어링 fault diagnosis를 위한 자체 foundation model 구축 (MOMENT 사용 아님). Contrastive learning + domain knowledge 통합.

**SNRGate와의 차별화 (manuscript 본문 초안)**:

> BearingFM (2024)은 contrastive learning과 도메인 지식을 결합한 베어링 전용 사전학습 모델을 제안하였다. 이는 단일 backbone에 도메인 prior를 주입하는 방식인 반면, SNRGate는 직교적 차원에서 접근한다 — 도메인 무관(generic) TSFM 백본을 그대로 활용하면서, 입력 신호의 SNR 추정과 다중 단 라우팅을 통해 운전 환경 변화에 적응한다. 따라서 SNRGate는 BearingFM과 같은 도메인 특화 backbone과도 결합 가능한 framework-level contribution이며, 두 접근은 보완적이다.

---

## 3. UniFault (Eldele et al. 2025) — bearing-specific FM 트렌드

**Reference**: Eldele et al. (2025). *UniFault: A Fault Diagnosis Foundation Model from Bearing Data*. arXiv:2504.01373.

**핵심 내용**: 베어링 데이터 기반 fault diagnosis foundation model. MOMENT 미사용.

**SNRGate에서의 위치**: bearing-specific FM 흐름의 맥락 논문으로 한 줄 인용. 차별화는 BearingFM과 동일한 논리.

---

## Action items (manuscript 작성 시)

- Related Work 섹션을 (a) classical SP, (b) DL fault diagnosis, (c) TSFM, (d) routing/MoE/cost-aware로 4개 sub-section 구성
- Li et al. 2025를 (c) TSFM sub-section의 직접 prior art로 인용 후 위 차별화 단락 삽입
- BearingFM/UniFault는 (c) sub-section의 도메인 특화 contrast로 한 단락에 묶음
- (d) routing sub-section에서 BranchyNet, MoE, ACT 등 일반 dynamic routing literature 언급 후 SNRGate의 SNR-aware + cost-aware 차별화

## Reference 정확성 검증 메모

- arXiv:2511.23177은 2025-11-28 제출로 매우 최신. 제출 전 arXiv에서 다시 한번 최신 버전/published venue 확인 필요.
- PHM Xi'an 2025 논문 (DOI: 10.1109/PHM-Xian66756.2025.11427729) — IEEE Xplore 유료라 제목/저자 미확인. 도서관 접근 가능하면 본문 확인 후 위 차별화 단락에 추가.
