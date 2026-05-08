# SNRGate: SNR 지도 기반 동적 모델 라우팅을 통한 베어링 결함 진단

**최수진, 양성진, 최종락, 신용우†**
*차세대동력시스템연구센터, 한국전자기술연구원*
†교신저자: credit@keti.re.kr

---

**SNRGate: SNR-Supervised Dynamic Routing of Time-Series Models for Bearing Fault Diagnosis**

Sujin Choi, Sungjin Yang, Jongrak Choi, Yongwoo Shin†
*Advanced Electrification System Research Center, Korea Electronics Technology Institute*

**Key Words**: Bearing Fault Diagnosis, Vibration Signal Analysis, Noisy Environments, Dynamic Model Selection, Computational Efficiency

---

## 초록

산업 환경에서 수집되는 베어링 진동 신호는 부하·회전 속도·측정 환경·센서 잡음 등에 따라 입력 품질이 달라지며, 이는 결함 진단 모델의 성능과 연산 효율에 영향을 미친다. 다수의 기존 결함 진단 모델은 입력 품질 변화와 관계없이 동일한 추론 경로를 사용하므로, 경량 모델은 저품질 신호에서 성능이 저하될 수 있고 대형 모델은 고품질 신호에서도 불필요한 연산을 수행한다. 본 논문은 학습 단계에서 노이즈 주입 시 설정한 신호 대 잡음비(SNR) 조건을 정규화된 소프트 라벨로 변환하여 입력 품질 추정기를 학습하고, 추론 단계에서는 원시 진동 신호만으로 예측된 품질 점수를 기반으로 서로 다른 연산 비용과 표현력을 갖는 시계열 모델을 동적으로 선택하는 **SNRGate**를 제안한다. SNRGate는 입력 신호의 품질 점수를 산출하는 입력 품질 추정기와, 해당 점수를 바탕으로 후보 시계열 모델의 선택 확률을 계산하는 SNR 인지 게이트로 구성된다. 학습 단계에서는 소프트 게이트를 통해 각 후보 모델의 출력을 미분 가능하게 결합하고, 추론 단계에서는 가장 높은 게이트 점수를 갖는 단일 모델만 실행하여 샘플별 연산 비용을 조절한다. Paderborn University Bearing Dataset의 인공-실 손상 전이 (Artificial-to-Real, A2R) 프로토콜에서 8개 SNR 조건 (clean ~ -20 dB) 으로 평가한 결과, SNRGate는 가장 강한 단일 tier baseline 대비 **clean ~ -10 dB 영역에서 평균 +21.7 pp Macro-F1 향상** (개별 SNR 별 +9.9 ~ +25.2 pp) 을 달성하면서, MOMENT-L only 대비 평균 effective parameter 를 **약 62% 절감** (341.2 M → 130 M) 하였다. 학습 가능한 게이트는 사전 정의 임계값 기반 라우팅 대비 평균 **+4.9 pp** 추가 향상을 보였다.

---

## 1. 서론 (Introduction)

회전기계의 베어링은 대표적인 산업 핵심 부품이며, 베어링 결함의 조기 진단은 비계획 정지 (unplanned downtime) 손실을 줄이는 가장 효과적인 예지보전(PHM) 수단의 하나이다 [1]. 진동 가속도 센서로 측정된 시계열 신호로부터 결함의 종류 (정상/내륜/외륜) 를 분류하는 데이터 기반 진단 기법은 최근 깊은 신경망과 시계열 기초모델 (Time-Series Foundation Model, TSFM) 의 도입으로 정확도 측면에서 큰 진보를 이루었다 [2,3].

그러나 산업 현장에서 수집되는 진동 신호는 실험실 측정과 달리 부하 변동, 회전 속도 변화, 인접 기계의 간섭, 센서 자체의 잡음 등 다양한 요인으로 **입력 신호 대 잡음비 (Signal-to-Noise Ratio, SNR) 가 큰 폭으로 변동**한다. 동일한 결함 패턴이라도 SNR 이 충분히 높은 신호에서는 단순한 분류기로도 신뢰성 있게 식별 가능하지만, 저SNR 환경에서는 결함 특징과 잡음 성분이 주파수 영역에서 혼재하여 [4] 더 큰 표현력을 갖는 모델이 요구된다. 다시 말해 진단 난이도가 입력별로 변동함에도 불구하고, 기존의 진단 시스템은 *모든 입력에 대해 동일한 모델·동일한 추론 경로*를 사용한다.

이러한 일률적 추론은 두 방향의 비효율을 동시에 일으킨다. 경량 모델만 사용하는 시스템은 저SNR 입력에서 성능이 저하되고, 반대로 고용량 모델만 사용하는 시스템은 고SNR 입력에서도 항상 최대 연산을 수행한다. 산업 배포 환경에서 처리 자원과 정확도 사이의 균형이 중요한 점을 고려할 때, **입력 신호의 품질에 따라 추론 경로를 동적으로 조절**하는 기제가 필요하다.

본 논문은 이 문제를 정면으로 다루는 새로운 베어링 결함 진단 프레임워크 **SNRGate (SNR-Supervised Dynamic Routing of Time-Series Models)** 를 제안한다. 핵심 아이디어는 두 가지이다. (i) 학습 단계에서 합성 노이즈 주입 시 설정한 SNR 값을 정규화된 소프트 라벨로 변환하여, 원시 진동 신호로부터 입력 품질 점수를 회귀하는 *입력 품질 추정기 (Quality Estimator)* 를 학습한다. (ii) 학습 가능한 *SNR 인지 게이트 (SNR-aware Gate)* 가 추정된 품질 점수를 입력으로 받아, 서로 다른 연산 비용과 표현력을 갖는 복수의 시계열 모델 중 하나를 샘플별로 선택한다. 학습 시에는 소프트 mixture 로 미분 가능하게 학습하고, 추론 시에는 가장 가까운 게이트 중심에 해당하는 단일 모델만 실행하여 샘플별 연산 비용을 조절한다.

본 논문의 기여는 다음과 같다.
1. **베어링 결함 진단에서 입력 SNR 적응 다중 단 라우팅을 처음 도입.** 기존 단일 backbone fine-tuning [3] 또는 도메인 특화 사전학습 [5,6] 과 직교적인, *cost-aware 적응* 차원을 새로 정의한다.
2. **결함과 잡음을 함께 다루는 SNR 지도 게이트.** 추정된 SNR 점수를 학습 가능한 prototype 중심에 대한 거리 기반 soft-to-hard 게이트로 활용함으로써, 기존 rule-based 임계값 라우팅과 sparse Mixture-of-Experts (MoE) [7] 사이의 결합 형태를 구성한다.
3. **인공-실 손상 전이 (A2R) 평가.** Paderborn-PU 베어링 데이터셋의 표준 PU-C1 프로토콜 [1,8] 위에서 8개 SNR 조건 평가를 수행하여, 본 프레임워크가 도메인 갭과 잡음 변동성을 동시에 다룰 수 있음을 보인다.

---

## 2. 관련 연구 (Related Work)

### 2.1 베어링 결함 진단

베어링 결함 진단의 고전적 접근은 진동 신호의 envelope 분석, 스펙트럼 분석, 통계적 시간/주파수 도메인 특징 추출에 기반한다 [4]. 최근 깊은 신경망의 도입으로 1D-CNN, ResNet, Transformer 계열 모델이 표준 baseline 으로 자리잡았으며 [9], 합성/실제 손상이 모두 포함된 Paderborn University Bearing Dataset (PU) [1] 이 가장 널리 인용되는 벤치마크이다. PU 데이터셋의 *PU-C1 인공-실 손상 전이 프로토콜* [1,8] 은 인공 손상 (drilled, EDM-cut) 베어링으로 학습한 모델이 가속수명시험으로 발생한 *real* 손상 베어링에 일반화되는지 평가하는 표준 셋업으로, 본 연구도 이를 그대로 따른다.

### 2.2 시계열 기초모델 (TSFM)

대규모 시계열 사전학습 모델인 MOMENT [2] 는 분류·이상탐지·회귀 등 범용 task 에 transfer 가능함을 입증하였다. Li et al. [3] 는 MOMENT 와 Mantis 를 모터 fault diagnosis 에 fine-tune 하여 1% 라벨 환경에서 기존 DL 대비 약 2 배 성능 향상을 보고하였다. 본 연구는 Li et al. [3] 의 backbone 적합성을 출발점으로 하되, 단일 backbone fine-tuning 이 다루지 못하는 *입력별 적응* 차원에서 차별화된다 — SNRGate 는 입력 SNR 에 따라 backbone 크기 자체를 동적으로 선택하여 정확도-비용 파레토에서 우월한 영역을 달성한다.

이와 직교적인 흐름으로, 베어링 도메인에 특화된 foundation model 도 개발되고 있다. BearingFM [5] 은 contrastive learning 과 도메인 지식을 결합한 베어링 전용 사전학습 모델이며, UniFault [6] 도 베어링 데이터 자체로 사전학습한 fault diagnosis FM 을 제안한다. 이들은 *단일 backbone* 에 도메인 prior 를 주입하는 방식인 반면, SNRGate 는 backbone 무관 (backbone-agnostic) framework-level 기여로서, 도메인 특화 backbone 과도 결합 가능하다.

### 2.3 잡음 환경에서의 진단

저SNR 환경에서 베어링 결함을 진단하기 위해 다양한 신호 처리 기반 전처리가 제안되어 왔다. Spectral Kurtosis [4] 는 결함의 충격성 (impulsivity) 이 잡음과 다른 분포 특성을 갖는다는 사실을 활용해 결함 신호와 잡음을 분리한다. 본 연구는 이러한 신호 처리 통찰을 직접 prior 로 사용하지는 않으나, 결함과 잡음의 분광적 비분리성 [4] 이라는 *문제 정의* 를 공유하며, 이를 학습 기반 라우팅 메커니즘으로 다룬다.

### 2.4 동적 라우팅 / 적응형 연산

입력에 따라 모델 capacity 를 조절하는 메커니즘은 BranchyNet [10], Mixture-of-Experts (MoE) [7] 등으로 일반화되어 있다. Sparsely-Gated MoE [7] 는 입력별로 가장 점수가 높은 *k* 개 expert 만 활성화하여 효율적인 추론을 수행한다. SNRGate 의 게이트 메커니즘은 sparse MoE 의 직계 후손이지만 두 가지 점에서 차별화된다. 첫째, 게이트의 입력이 학습된 임베딩이 아니라 *명시적 SNR 추정값* 이라는 점, 둘째, 후보 모델들이 같은 도메인의 *서로 다른 capacity* 를 갖는 시계열 모델로 구성된다는 점이다. 이를 통해 paper 에서 다루는 진단 정확도-비용 trade-off 를 입력 SNR 변동에 직접 연결한다.

---

## 3. 제안 방법 (Method)

### 3.1 전체 구조

SNRGate 는 (Fig. 1) 4 개의 stage 로 구성된다.

```
Stage 1: 입력             Stage 2: SNR 인지 라우팅                 Stage 3: 후보 모델               Stage 4: 예측
─────────             ────────────────────────────              ─────────────────────             ───────────────
진동 윈도우 x ──┬─→ 입력 품질 추정기 q ────┬─→ SNR 인지 게이트 ─→ {S, B, L} 중 1 개 ─→ 결함 분류
              │                          │     (3 prototype centers)
              └──────────────────────────┴────────────────────────────┘
```

입력 진동 윈도우 $x \in \mathbb{R}^{T}$ ($T=2048$, 64 kHz 샘플링 기준 32 ms) 는 두 분기로 동시 처리된다. (a) 입력 품질 추정기는 $x$ 로부터 정규화된 SNR 점수 $q = f_\theta(x) \in [0,1]$ 를 회귀한다. (b) 학습 가능한 SNR 인지 게이트 $G$ 는 $q$ 로부터 후보 모델 $\{M_S, M_B, M_L\}$ 의 선택 확률 $p \in \Delta^2$ 를 계산한다. 학습 단계에서는 soft mixture $\hat{y} = \sum_k p_k M_k(x)$ 으로 미분 가능하게 학습하며, 추론 단계에서는 가장 높은 점수를 갖는 단일 모델만 실행하여 샘플별 연산 비용을 조절한다.

### 3.2 입력 품질 추정기 (Quality Estimator)

입력 품질 추정기는 1D CNN 회귀기로, 4개의 stride-2 convolution block + global average pooling + linear-sigmoid 헤드로 구성된 약 93K 파라미터 모델이다. 학습 시에는 깨끗한 윈도우 $x$ 에 SNR 조건 $\sigma \in \mathcal{S}_{aug}$ 의 가산 백색 가우시안 잡음(AWGN) 을 주입하여 $\tilde{x}$ 를 생성하고, 이 $\tilde{x}$ 의 SNR 정답을 다음과 같이 정규화된 소프트 라벨로 변환한다:

$$q^{\ast}(\sigma) = \mathrm{sigmoid}\!\left(-\sigma_{\text{dB}} / 10\right)$$

이 변환은 $\sigma_{\text{dB}} \to \infty$ 에서 $q^{\ast} \to 0$ (잡음 없음), $\sigma_{\text{dB}} \to -\infty$ 에서 $q^{\ast} \to 1$ (매우 잡음) 의 범위를 가지며 학습 데이터의 SNR 분포를 부드럽게 매핑한다. 학습 손실은 회귀 평균제곱오차이다:

$$\mathcal{L}_q = \frac{1}{B} \sum_{i=1}^{B} \left(f_\theta(\tilde{x}_i) - q^{\ast}(\sigma_i)\right)^2$$

추론 시에는 *원시 진동 신호 $x$* 만 입력으로 받아 $q$ 를 출력한다. 즉, 학습 시 사용된 SNR 라벨은 자기지도 (self-supervised) 신호로만 활용되며, 배포 시에는 추가 라벨 없이 작동한다.

**설계 한계 (signal energy vs noise level confound).** 본 추정기는 AWGN 주입 신호로 학습되므로, 입력 *진폭(amplitude)* 의 변화를 *노이즈 수준* 의 변화로 학습할 가능성이 있다. 즉 *low-energy fault signature* (예: broadband 저충격성 손상) 와 *clean signal* 모두 추정기에게 "낮은 잡음 점수" 로 보일 수 있다. §6 (Limitations) 에서 이 confound 가 KA15/KA22 베어링의 분류 실패로 직접 이어지는 사례를 보고한다.

### 3.3 SNR 인지 게이트 (SNR-aware Gate)

게이트 $G$ 는 후보 모델 수 $K=3$ 에 대응하는 *학습 가능한 prototype 중심* $\boldsymbol{\mu} = (\mu_S, \mu_B, \mu_L) \in [0,1]^3$ 와 온도 $\tau$ 로 매개변수화된다. 입력 품질 점수 $q$ 가 주어지면, 각 prototype 까지의 거리에 비례하여 negatively scaled softmax 로 mixture weight 를 계산한다:

$$p_k(q) = \frac{\exp\!\left(-|q - \mu_k|/\tau\right)}{\sum_{j=1}^{K} \exp\!\left(-|q - \mu_j|/\tau\right)}, \quad k \in \{S, B, L\}$$

학습 초기에는 $\tau = 1$ 로 시작하여 모든 후보 모델에 거의 균등한 mixture weight 가 할당되며 (모든 tier 로 gradient 가 흐름), 학습이 진행되면서 $\tau$ 를 지수 스케줄로 $\tau_{\min}=0.05$ 까지 anneal 하여 mixture 가 거의 one-hot 분포로 수렴하도록 유도한다. 추론 단계에서는 다음과 같이 hard argmin 라우팅을 수행한다:

$$k^{\ast}(q) = \arg\min_k |q - \mu_k|$$

이는 $\tau \to 0$ 의 극한과 일치하며, 단일 후보 모델만 실행되므로 샘플별 추론 비용이 보존된다.

### 3.4 단조성 강제 (Monotonicity Constraint)

게이트 의미를 보존하기 위해 prototype 중심들은 학습 전 과정에서 $\mu_S < \mu_B < \mu_L$ 의 순서를 유지해야 한다. 즉, 가장 작은 후보 모델 $M_S$ 가 가장 낮은 잡음 영역의 prototype 을 갖고, 가장 큰 모델 $M_L$ 이 가장 높은 잡음 영역의 prototype 을 가져야 한다. 그렇지 않으면 라우팅과 모델 capacity 사이의 의도된 매핑이 깨진다. 우리는 이를 다음 soft penalty 로 강제한다:

$$\mathcal{L}_{order} = \mathrm{ReLU}(\mu_S - \mu_B + \delta)^2 + \mathrm{ReLU}(\mu_B - \mu_L + \delta)^2$$

여기서 $\delta = 0.05$ 는 안전 margin 이다. 사전 실험에서 이 penalty 가 없을 때 게이트 중심들이 학습 도중 교차하여 binary collapse 가 발생함을 확인하였으며 (S=0.343, B=0.316, L=0.804 — S 가 B 보다 큰 값으로 수렴), 이는 의도된 3-tier 라우팅을 사실상 2-tier 로 축소시켰다. 본 penalty 적용 후에는 학습 전반에 걸쳐 단조성이 보존됨을 확인하였다.

### 3.5 후보 모델 (Candidate Models)

SNRGate framework 는 후보 모델의 구체적 선택과 직교적이다. 본 연구의 실험에서는 시계열 기초모델인 MOMENT [2] 의 세 가지 크기 (S=35.3M, B=109.6M, L=341.2M parameters) 를 사용한다. 백본은 frozen 으로 유지하고, 각 tier 에 대해 LoRA [11] (rank $r=8$) 어댑터와 분류 헤드만 학습 가능하게 둠으로써 단일 RTX 3070 8 GB GPU 에서 동시 fine-tuning 이 가능하도록 한다.

### 3.6 학습 손실 (Training Objective)

전체 학습 손실은 분류·품질 회귀·게이트 단조성의 가중합이다:

$$\mathcal{L} = \mathcal{L}_{cls} + \lambda_q \mathcal{L}_q + \lambda_o \mathcal{L}_{order}$$

여기서 $\mathcal{L}_{cls}$ 는 mixture logit $\hat{y}$ 와 정답 라벨 $y$ 사이의 weighted cross-entropy 이며, 가중치는 학습 데이터의 클래스별 inverse frequency 로 설정하여 PU-C1 train split 의 1:5:7 (N:IR:OR) 클래스 불균형을 보정한다. $\lambda_q = \lambda_o = 1.0$ 으로 고정한다.

---

## 4. 실험 셋업 (Experimental Setup)

### 4.1 데이터셋

본 연구는 **Paderborn University Bearing Dataset (PU)** [1] 의 vibration_1 채널 (가속도계, 64 kHz) 을 사용한다. 표준 *PU-C1 인공-실 손상 전이 프로토콜* [8] 을 따라 3 개 클래스 (Normal / Inner Race / Outer Race) 로 라벨링하며, 다음과 같이 split 한다.

- **Train (인공 손상 source domain)**: K002 (healthy) + KA01/03/04/05/06/07/09 (artificial OR, 7 베어링) + KI01/03/04/05/07 (artificial IR, 5 베어링), 총 13 베어링.
- **Within-train Validation**: KA08 + KI08 + K002 의 80/20 file-level split. 학습 중 best-on-val 체크포인트 선택용.
- **Test (실 손상 target domain)**: K001 (healthy) + KA15/KA16/KA22/KA30 (real OR, 4 베어링) + KI14/KI16/KI17/KI18/KI21 (real IR, 5 베어링), 총 10 베어링. 모든 본 논문 결과는 이 set 에서만 보고한다.

조합 손상 베어링 (KB23/24/27) 은 인공 손상 짝이 존재하지 않아 A2R 비대칭을 일으키므로 본 연구에서 제외한다. 한 베어링당 최대 8 개 .mat 파일을 사용하며 (`max_files_per_bearing=8`), 각 파일에서 길이 2048 의 비중복 윈도우 (stride=2048) 를 추출한다. 최종 데이터셋 크기는 train 12,747 / val 2,248 / test 10,000 윈도우이다.

### 4.2 잡음 주입 및 평가

학습 시 매 mini-batch 마다 $\mathcal{S}_{aug} = \{\infty, 20, 10, 5, 0, -5, -10, -20\}$ dB 에서 SNR 을 균등 샘플링하여 동일 batch 의 모든 윈도우에 동일 SNR 의 AWGN 을 주입한다. 정규화는 노이즈 주입 후 윈도우별 z-score 로 적용하여 SNR 이 신호 power 기준으로 정확하도록 한다. 평가 시에는 동일 8 개 SNR 조건 각각에 대해 별도의 sweep 을 수행하여 Macro-F1 을 측정한다.

### 4.3 학습 설정 (Training Configuration)

모든 모델은 단일 RTX 3070 GPU (8 GB) 에서 bf16 mixed-precision 으로 학습한다. Optimizer 는 AdamW (LoRA lr = $2 \times 10^{-4}$, head lr = $2 \times 10^{-4}$, gate center lr = $1 \times 10^{-2}$, weight decay = $1 \times 10^{-3}$) 를 사용한다. SNRGate 는 8 epoch, baseline 단일 tier 모델은 5 epoch 동안 학습하며 batch size = 8 로 통일한다. 게이트 온도 $\tau$ 는 $1.0 \to 0.05$ 로 학습 horizon 전체에 걸쳐 지수 anneal 한다. 시드 42 의 단일 실행으로 보고한다.

### 4.4 비교 기준 (Baselines)

다음 4 가지 비교 기준을 사용한다.

1. **MOMENT-S only / MOMENT-B only / MOMENT-L only** — SNRGate 와 동일한 데이터·동일한 SNR augmentation·동일한 LoRA 설정에서 단일 tier 만 독립 학습한 모델. SNRGate 의 핵심 contribution 이 *라우팅* 임을 격리하기 위한 비교이다.
2. **HardGate (rule-based ablation)** — SNRGate 와 동일한 후보 모델 + 학습된 weights 를 사용하되, 학습 가능한 게이트 대신 규칙 기반 임계값 게이트 ($q < 0.33 \to S$, $0.33 \le q < 0.66 \to B$, $q \ge 0.66 \to L$) 로 라우팅을 결정하는 ablation. 학습 가능한 게이트의 효과를 평가한다.

### 4.5 평가 지표

본 연구의 1차 metric 은 **SNR 별 Macro-F1** 이다. 클래스 불균형 (특히 healthy 가 train 의 5.9%, test 의 10%) 을 고려하여 per-class F1 도 함께 보고한다. 추가로 cost 측면에서는 (i) sample 당 *effective parameter count* (각 SNR 의 라우팅 분포에 가중) 와 (ii) MOMENT-L only 대비 평균 절감률을 함께 보고한다.

---

## 5. 결과 (Results)

### 5.1 SNR 별 Macro-F1 (Main Table)

Table 1 은 5 개 method 의 8 개 SNR 조건별 Macro-F1 (test set, PU-C1 real-damage bearings) 을 보고한다.

**Table 1.** SNR-stratified Macro-F1 on Paderborn PU-C1 A2R test split. Best per column in **bold**.

| Method | clean (∞) | 20 dB | 10 dB | 5 dB | 0 dB | -5 dB | -10 dB | -20 dB | 평균* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MOMENT-S only | 0.325 | 0.326 | 0.324 | 0.322 | 0.311 | 0.296 | 0.288 | **0.293** | 0.313 |
| MOMENT-B only | 0.404 | 0.402 | 0.383 | 0.372 | 0.368 | 0.335 | 0.297 | 0.297 | 0.366 |
| MOMENT-L only | 0.389 | 0.385 | 0.382 | 0.357 | 0.284 | 0.257 | 0.241 | 0.250 | 0.328 |
| HardGate (rule, ablation) | 0.600 | 0.600 | 0.589 | 0.572 | 0.522 | 0.486 | 0.363 | 0.281 | 0.533 |
| **SNRGate (ours)** | **0.636** | **0.635** | **0.633** | **0.624** | **0.591** | **0.563** | **0.396** | 0.254 | **0.583** |

*평균은 clean ~ -10 dB 의 7 개 SNR 조건 산술평균 (-20 dB 제외; §6 한계 참조).

핵심 관찰은 다음과 같다.

(i) **SNRGate 가 clean ~ -10 dB 의 모든 SNR 에서 최고 성능을 달성한다.** 가장 강한 단일 tier baseline 인 MOMENT-base 대비 평균 **+21.7 pp** 향상이며, 개별 SNR 별로는 clean (+23.2 pp), 0 dB (+22.3 pp), -5 dB (+22.8 pp), -10 dB (+9.9 pp) 의 격차를 보인다.

(ii) **단일 tier baseline 에서 가장 큰 모델인 MOMENT-L 가 0 dB 이하 영역에서 더 작은 -B/-S 보다 떨어지는** 비단조 패턴을 보인다 (예: 0 dB 에서 L=0.284 vs B=0.368). 이는 large 모델 + 단일 tier + LoRA fine-tuning + 작은 학습 데이터 + mixed-SNR augmentation 의 결합에서 발생하는 학습 불안정성으로, *단순히 큰 모델을 사용하는 것이 답이 아니라는* SNRGate 의 라우팅 동기를 직접적으로 입증한다.

(iii) **-20 dB 에서 SNRGate (0.254) 가 best baseline (B=0.297) 보다 -4.3 pp 떨어진다.** 본 한계는 다음 §5.4 와 §6 에서 routing 붕괴 현상으로 자세히 논의한다.

### 5.2 라우팅 분포 (SNR vs Tier Share)

Fig. 2 (`figures/F3_routing_dist.pdf`) 는 8 개 SNR 조건에서 SNRGate 가 각 tier 로 input 을 라우팅한 비율을 stacked bar 로 표시한다.

**Table 2.** Per-SNR routing share (test set).

| SNR (dB) | S share | B share | L share | dominant |
|---:|---:|---:|---:|:---|
| ∞ (clean) | 65.4% | 4.3% | 30.2% | S |
| 20 | 65.4% | 4.4% | 30.2% | S |
| 10 | 65.3% | 4.6% | 30.1% | S |
| 5 | 65.2% | 4.3% | 30.4% | S |
| 0 | 64.4% | 5.2% | 30.4% | S |
| -5 | 61.7% | 7.7% | 30.6% | S |
| -10 | 52.2% | 27.1% | 20.8% | S→B 전환 |
| -20 | 76.3% | 20.3% | 3.3% | S (collapse) |

핵심 관찰: clean ~ -5 dB 에서 학습된 라우팅은 *S* 로의 dominant 라우팅 (65% 부근) 과 *L* 로의 일관된 30% 부근 라우팅을 결합한다. 즉 일부 input 은 SNR 과 무관하게 high-capacity tier 를 요구하는 *어려운 sample* 임을 시사한다. -10 dB 에서 가장 두드러진 변화 (S 65→52%, B 7→27%) 가 발생하며, 이는 학습된 게이트가 noise 가 강한 영역에서 high-capacity tier 의 필요성을 인식함을 보여준다.

**Final gate centers** $\boldsymbol{\mu} = (\mu_S, \mu_B, \mu_L) = (0.491, 0.583, 0.687)$ 는 8 epoch 전 과정에 걸쳐 $\mu_S < \mu_B < \mu_L$ 의 단조성을 유지하였다. §3.4 에서 도입한 ordering penalty $\mathcal{L}_{order}$ 가 없을 때 사전 실험에서 관찰된 $\mu_S = 0.343 > \mu_B = 0.316$ 의 binary collapse 는 본 학습에서 발생하지 않았다.

### 5.3 비용-정확도 파레토

Fig. 3 (`figures/F4_pareto.pdf`) 은 sample 당 effective parameter count (라우팅 share 가중) vs 평균 Macro-F1 (clean ~ -10 dB) 산점도이다.

**Table 3.** Cost-accuracy summary (평균은 Table 1 과 동일하게 clean ~ -10 dB 7-SNR 산술평균).

| Method | Effective params (M, 평균) | MOMENT-L 대비 | 평균 Macro-F1 (clean ~ -10 dB) |
|---|---:|---:|---:|
| MOMENT-S only | 35.3 | 10.3% | 0.313 |
| MOMENT-B only | 109.6 | 32.1% | 0.366 |
| MOMENT-L only | 341.2 | 100.0% | 0.328 |
| HardGate (rule) | ~140 | ~41% | 0.533 |
| **SNRGate (ours)** | **130** | **38.1%** | **0.583** |

SNRGate 는 MOMENT-L only 대비 **약 62% 연산 절감 (effective param 341.2 M → 130 M) + 25.5 pp 정확도 향상** 을 동시에 달성하며, 단일 tier baseline 어떤 점도 SNRGate 의 trade-off 를 dominate 하지 못하는 *Pareto-dominant* 영역에 위치한다.

### 5.4 학습 가능한 게이트의 효과 (HardGate ablation)

Table 1 의 *HardGate (rule, ablation)* 행은 SNRGate 와 동일한 후보 모델 weights + 동일한 quality estimator 를 사용하되, 학습된 게이트 대신 사전 정의 임계값 ($q < 0.33 \to S, 0.33 \le q < 0.66 \to B, q \ge 0.66 \to L$) 으로 라우팅을 결정하는 ablation 이다.

**Table 4.** SNRGate vs HardGate Δ Macro-F1 (pp). 평균 행은 Table 1 과 동일한 clean ~ -10 dB 7-SNR 산술평균.

| SNR | HardGate | SNRGate | Δ pp |
|---|---:|---:|---:|
| clean | 0.600 | 0.636 | **+3.6** |
| 20 dB | 0.600 | 0.635 | **+3.6** |
| 10 dB | 0.589 | 0.633 | **+4.4** |
| 5 dB | 0.572 | 0.624 | **+5.2** |
| 0 dB | 0.522 | 0.591 | **+6.9** |
| -5 dB | 0.486 | 0.563 | **+7.7** |
| -10 dB | 0.363 | 0.396 | **+3.3** |
| -20 dB | 0.281 | 0.254 | -2.7 |
| **평균 (clean ~ -10 dB)** | 0.533 | **0.583** | **+4.9** |

학습된 게이트는 7 개 SNR 조건 (clean ~ -10 dB) 모두에서 rule-based 임계값 대비 우월하며, 평균 **+4.9 pp** 향상을 보인다. 이는 prototype 중심 $\boldsymbol{\mu}$ 의 *학습 가능성* 이 단순한 사전 정의 임계값으로 대체될 수 없음을 의미한다.

흥미롭게도 -20 dB 에서는 두 방법이 모두 collapse 한다 (HardGate 0.281, SNRGate 0.254). 이는 routing 메커니즘 자체보다는 **공통의 quality estimator 가 -20 dB 에서 saturate 되는** 구조적 문제임을 시사한다. §6 에서 자세히 논의한다.

### 5.5 학습 가능한 게이트의 단조성 (보충 결과)

§3.4 에서 도입한 ordering penalty 의 효과를 직접 검증하기 위해, 본 세션 이전에 동일 셋업에서 penalty 를 제외하고 학습한 결과 (final centers $\boldsymbol{\mu} = (0.343, 0.316, 0.804)$) 와 비교하였다. Penalty 없을 때 $\mu_S > \mu_B$ 의 단조성 위반이 발생하여, 게이트의 hard-gate 라우팅 시 의도된 "small model = low noise" 매핑이 깨지고 effective 2-tier 라우팅으로 collapse 하였다. Penalty 적용 후에는 8 epoch 전체에서 $\mu_S < \mu_B < \mu_L$ 가 유지되며 최종 centers 는 $(0.491, 0.583, 0.687)$ 로 수렴하였다.

### 5.6 Per-class F1 (N / IR / OR 분해)

aggregate Macro-F1 만으로는 라우팅 효과의 클래스별 분포를 알 수 없다. Table 5 는 5 method 의 클래스별 F1 (clean ~ -10 dB 7-SNR 산술평균) 을 보고한다.

**Table 5.** Per-class F1 by method (7-SNR average, test set).

| Method | F1_N | F1_IR | F1_OR | Macro-F1 |
|---|---:|---:|---:|---:|
| MOMENT-S only | 0.001 | 0.457 | 0.485 | 0.314 |
| MOMENT-B only | 0.105 | 0.518 | 0.478 | 0.367 |
| MOMENT-L only | 0.276 | 0.275 | 0.441 | 0.331 |
| HardGate (rule) | 0.438 | 0.669 | 0.504 | 0.537 |
| **SNRGate (ours)** | **0.474** | **0.768** | **0.520** | **0.587** |

핵심 관찰:

(i) **MOMENT-S 단독은 F1_N = 0.001** — 단일 healthy 베어링 (K001) 의 영역에서 거의 완전히 실패한다. PU-C1 의 1-bearing-per-class healthy 제약이 small-capacity 단일 모델의 학습을 압도함을 시사한다.

(ii) **MOMENT-L 의 F1_OR = 0.441** 은 baseline 중 최고 — sharp impulsive fault 패턴 (KA01-09 artificial OR) 의 학습은 high capacity 가 필요. 그러나 동일 모델의 F1_IR = 0.275 는 baseline 중 최저로, IR 의 미세 패턴은 over-parameterized model 에서 오히려 collapse 한다.

(iii) **SNRGate 는 모든 클래스에서 baseline 의 최고치를 능가한다** (F1_N +0.198, F1_IR +0.250, F1_OR +0.035). 특히 IR 향상폭 +0.250 은 multi-tier joint training 의 효과로 해석된다 — 라우팅이 IR 의 미세 패턴을 학습하기 적합한 capacity 영역으로 input 을 보낸다.

### 5.7 Per-bearing accuracy 와 KA15/KA22 사례

aggregate per-class F1 도 베어링 단위 분포를 가린다. Table 6 은 test set 의 10 개 베어링 별 정확도 (7-SNR 평균) 를 5 method 에 대해 보고한다.

**Table 6.** Per-bearing accuracy (7-SNR average, test set, 10 bearings).

| Bearing | Class | M-S | M-B | M-L | HardGate | **SNRGate** |
|---|:---:|---:|---:|---:|---:|---:|
| K001 | N | 0.000 | 0.066 | 0.211 | 0.331 | **0.341** |
| KA15 | OR | 0.637 | 0.579 | 0.728 | 0.097 | **0.004** |
| KA16 | OR | 0.947 | 0.935 | 0.995 | 0.911 | 0.896 |
| KA22 | OR | 0.473 | 0.410 | 0.314 | 0.176 | **0.006** |
| KA30 | OR | 0.344 | 0.331 | 0.416 | 0.865 | 0.878 |
| KI14 | IR | 0.424 | 0.531 | 0.182 | 0.815 | 0.987 |
| KI16 | IR | 0.322 | 0.381 | 0.169 | 0.516 | 0.787 |
| KI17 | IR | 0.504 | 0.491 | 0.448 | 0.681 | 0.886 |
| KI18 | IR | 0.226 | 0.331 | 0.121 | 0.555 | 0.845 |
| KI21 | IR | 0.611 | 0.709 | 0.119 | 0.903 | 0.985 |
| **평균** | — | 0.449 | 0.476 | 0.370 | 0.585 | **0.661** |

**KA15 / KA22 catastrophic failure** 는 SNRGate 의 가장 두드러진 약점이다. 두 베어링 모두 *real-damage OR* 로 라벨링되어 있으나, SNRGate 의 정확도는 각각 0.004 / 0.006 — 사실상 *모든 윈도우를 잘못 분류*. 그러나 동일한 학습 데이터로 학습된 baseline 단독 모델은 동일 베어링을 정상 분류한다 — MOMENT-L 단독은 KA15 를 0.728, MOMENT-S 단독은 KA22 를 0.473 의 정확도로 분류한다.

이 결과는 **KA15/KA22 의 실패가 데이터 분포의 문제 (OOD) 가 아닌 SNRGate 의 라우팅 의사결정 자체의 결함** 임을 시사한다. 진단 분석은 다음과 같다.

(a) **Confusion mode**: SNRGate 는 KA15 의 99.5%, KA22 의 99.0% 를 IR 클래스로 오분류한다.

(b) **Routing mode**: KA15 의 99.2%, KA22 의 97.6% 가 가장 작은 모델 MOMENT-S 로 라우팅된다. 즉 quality estimator 가 두 베어링을 "고품질 / 저잡음 입력" 으로 판단한다.

(c) **Fault signature**: 두 실패 베어링의 *kurtosis* 는 각각 4.21 과 4.18 로, Gaussian baseline (3.0) 에 매우 근접하다. 반면 정상 분류된 KA16 의 kurtosis 는 13.11 — *전형적 베어링 결함의 sharp impulsive ringing* 을 보인다. 신호 표준편차 또한 KA15/KA22 (0.20, 0.18) 가 KA16/KA30 (0.73, 0.36) 대비 낮다 (Fig. T1-T4 참조).

§3.2 에서 명시한 *signal energy ↔ noise level confound* 가 직접 발현된 사례다. KA15/KA22 의 broadband low-impulsivity 손상은 신호 진폭이 작아 quality estimator 에게 "clean signal" 로 보이고, 그 결과 MOMENT-S 로 라우팅되어 학습 분포 (sharp impulsive OR) 와 일치하지 않는 fault signature 가 IR 로 오분류된다. 이 실패의 책임 소재 — 데이터 분포 (OOD) 가 아니라 라우팅 메커니즘 — 는 §6 Limitations 에 정직하게 기록한다.

#### Figures (보충 자료)

본 분석을 시각적으로 확인할 수 있는 figure 는 다음과 같다 (`paper/figures/`).

- **Fig. T1** (`T1_time_envelope_KA15_KA22.pdf`): 5 베어링의 40 ms 시간-도메인 신호 + Hilbert envelope. KA15/KA22 의 평탄한 envelope 와 KA16/KA30 의 sharp impulse train 이 시각적으로 대비된다.
- **Fig. T2** (`T2_fft_envelope_spectrum.pdf`): 0–8 kHz 주파수 스펙트럼 (log-y) + 0–1 kHz envelope spectrum (fault frequency 영역). KA15/KA22 의 envelope spectrum 은 fault 특성 주파수 피크가 약하다.
- **Fig. T3** (`T3_kurtosis.pdf`): 4096-sample sliding window kurtosis. KA15/KA22 (≈ 4) 와 KA16/K001 (≈ 13) 두 그룹이 명확히 분리된다.
- **Fig. T4** (`T4_combined_fault_signature.pdf`): 5×3 공동 figure (time / FFT / envelope spectrum) — paper main body 에 통합 후보.

---

## 6. 한계 (Limitations)

본 연구의 한계는 *방법론적 한계* (메커니즘 결함) 와 *셋업적 한계* (데이터·실험 환경) 로 구분하여 명시한다.

### 6.1 방법론적 한계

- **Quality estimator 의 signal-energy / noise-level confound (KA15/KA22 catastrophic failure)**. §5.7 에서 보고한 바와 같이, SNRGate 는 real-damage OR 베어링 KA15, KA22 를 각각 0.004 / 0.006 의 정확도로 분류하며 — 사실상 전 윈도우를 IR 로 오분류한다. 이는 데이터 분포 (OOD) 의 문제가 아니다: 동일 학습 데이터로 학습된 baseline 단독 모델 (MOMENT-L) 은 KA15 를 0.728 의 정확도로 정상 분류한다. **실패의 책임 소재는 quality estimator 의 설계** 에 있다. 두 베어링의 broadband low-impulsivity 손상 (kurtosis ≈ 4.2, std ≈ 0.20) 은 *신호 진폭은 낮지만 잡음은 적은* 입력이며, AWGN-supervised 추정기는 이를 "고품질 / clean" 으로 잘못 해석하여 가장 작은 모델 (MOMENT-S) 로 라우팅한다 (KA15 의 99.2%, KA22 의 97.6%). MOMENT-S 는 학습 분포 (sharp impulsive OR) 와 다른 fault signature 를 IR 로 오분류한다. 향후 연구로 *class-aware difficulty estimator* 또는 *impulsivity-aware proxy* (예: spectral kurtosis [4]) 를 quality estimator 에 통합하는 방향이 자연스러운 보완책이다.

- **-20 dB 에서의 routing collapse**. 본 연구의 학습은 SNR ∈ {clean, ..., -20 dB} 의 균등 augmentation 을 사용하나, 추정기가 -20 dB 에서 saturate 하여 L 라우팅 비율이 3.3% 로 무너진다 (§5.2). HardGate 도 동일 한계를 보이므로 (-20 dB 0.281 vs 0.254) 라우팅 메커니즘 자체보다 추정기의 calibration 한계로 판단된다. -20 dB sample 비중을 학습에서 늘리거나, 추정기 출력을 logit 공간으로 확장하면 완화 가능할 것으로 기대된다.

### 6.2 셋업적 한계

- **합성 AWGN 학습**. 실 산업 잡음은 종종 비가우시안 (impulsive EMI from VFD PWM, 기계 간섭 등) 이며 [12] 본 연구의 학습 잡음은 이를 모두 다루지 못한다. 차후 연구에서 pink noise·structured noise 추가 평가가 필요하다.
- **단일 데이터셋**. 본 연구는 PU 데이터셋만으로 평가하였다. CWRU [13], MAFAULDA 등 cross-machine 일반화는 향후 과제이다.
- **단일 healthy 베어링**. PU-C1 프로토콜의 표준 셋업으로 인해 train/test 모두에 healthy 가 1 베어링 (K002, K001) 만 포함되며, healthy 클래스의 결과는 이 1 베어링의 특성에 좌우된다. §5.6 의 F1_N 결과 (모든 method 에서 0.001 ~ 0.474) 는 이 제약의 직접적 표현이다. 본 protocol 에서 미사용된 K003-K006 을 healthy augmentation 에 추가하면 완화할 수 있으나, 이는 PU-C1 의 정합성을 깨뜨리므로 별도 protocol 로 보고할 필요가 있다.
- **OR/IR 비대칭** (train: 7 OR, 5 IR). KAt 서버에서 일부 인공 손상 베어링 (KA02/KI02/KI06/KI09) 이 deprecation 으로 다운로드 불가하여 발생한 데이터셋 부산물이다. weighted cross-entropy 로 자동 보정되지만 명시한다.
- **Edge HW 측정 부재**. 본 연구는 cost 절감을 *parameter count proxy* 로만 보고하며, Jetson Nano/RPi 등 edge 장비에서의 실 측정은 향후 과제이다.
- **단일 시드 (42)**. 분산성을 줄이기 위해 시드 1, 2 의 추가 실행이 권장되나 본 paper 에서는 단일 시드로 보고한다.

---

## 7. 결론 (Conclusion)

본 논문은 산업 환경의 베어링 진동 신호가 입력별로 큰 SNR 변동성을 가짐에도 불구하고 기존 진단 시스템이 일률적 추론 경로를 사용하는 한계를 다루기 위해, **SNRGate** — SNR 지도 기반 동적 시계열 모델 라우팅 프레임워크 — 를 제안하였다. 입력 품질 추정기는 합성 노이즈 주입 시 설정한 SNR 을 정규화된 소프트 라벨로 활용해 자기지도 학습되며, 학습 가능한 SNR 인지 게이트는 추정된 품질 점수와 학습된 prototype 중심 사이의 거리에 기반한 단조 라우팅을 수행한다. 학습 단계의 soft mixture 와 추론 단계의 hard 라우팅을 결합하여 미분 가능한 학습과 sample 별 단일 tier 추론을 동시에 달성하였다. Paderborn-PU 데이터셋의 인공-실 손상 전이 (PU-C1 A2R) 프로토콜에서 8 개 SNR 조건 평가 결과, SNRGate 는 가장 강한 단일 tier baseline 대비 clean ~ -10 dB 영역에서 평균 **+21.7 pp** Macro-F1 향상을 달성하면서, MOMENT-L only 대비 평균 effective parameter 를 **약 62%** (341.2 M → 130 M) 절감하였다. 학습 가능한 게이트가 사전 정의 임계값 라우팅 대비 평균 **+4.9 pp** 추가 향상을 보임으로써, prototype 중심의 *학습 가능성* 이 핵심 설계 결정임을 입증하였다.

본 연구의 한계 (§6) 는 *방법론적* 한계와 *셋업적* 한계로 구분된다. 방법론적으로 가장 중요한 한계는 quality estimator 의 *signal-energy / noise-level confound* — KA15/KA22 의 broadband low-impulsivity 손상이 추정기에게 "clean signal" 로 보여 작은 모델로 라우팅되는 catastrophic failure 다. 셋업적 한계로는 합성 AWGN 학습, 단일 데이터셋, 단일 healthy 베어링, 단일 시드가 있다. 이를 바탕으로 향후 연구 방향은 다음과 같다. 첫째, **class-aware 또는 impulsivity-aware difficulty estimator** — spectral kurtosis [4] 같은 신호처리 prior 를 quality estimator 에 결합하여 KA15/KA22 류의 misrouting 을 차단. 둘째, 비가우시안 산업 잡음 (impulsive EMI, 기계 간섭) 에 대한 robustness 평가 — pink noise / structured noise augmentation. 셋째, cross-machine 일반화 — CWRU [13] 등 추가 데이터셋. 넷째, edge 장비 (Jetson Nano, Raspberry Pi) 실측 latency / energy 보고. 다섯째, -20 dB 에서 관찰된 quality estimator saturation 의 calibration 강화. 여섯째, 제안 framework 의 backbone 무관성 활용 — BearingFM [5] 등 도메인 특화 backbone 과의 결합. 본 연구의 framework-level 기여는 이러한 후속 확장과 직교적이며, 나아가 다른 시계열 PHM 도메인 (모터, 풍력 발전기, 배터리 등) 에도 적용 가능할 것으로 기대된다.

---

## 참고문헌 (References)

[1] Lessmeier, C., Kimotho, J. K., Zimmer, D., and Sextro, W., 2016, "Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification," PHM Society European Conference, Vol. 3, No. 1.

[2] Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., and Dubrawski, A., 2024, "MOMENT: A Family of Open Time-Series Foundation Models," Proc. of the 41st International Conference on Machine Learning.

[3] Li, et al., 2025, "Data-Efficient Motor Condition Monitoring with Time Series Foundation Models," arXiv:2511.23177.

[4] Antoni, J., and Randall, R. B., 2006, "The spectral kurtosis: application to the vibratory surveillance and diagnostics of rotating machines," Mechanical Systems and Signal Processing, 20(2), 308–331.

[5] BearingFM, 2024, "BearingFM: Towards a foundation model for bearing fault diagnosis by domain knowledge and contrastive learning," International Journal of Production Economics.

[6] Eldele, et al., 2025, "UniFault: A Fault Diagnosis Foundation Model from Bearing Data," arXiv:2504.01373.

[7] Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J., 2017, "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," ICLR 2017.

[8] Bayer, et al., 2023, "FaultFormer: Pretraining Transformers for Adaptable Bearing Fault Classification," arXiv:2312.02380v3.

[9] Zhang, W., Peng, G., Li, C., Chen, Y., and Zhang, Z., 2017, "A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals," Sensors, 17(2), 425.

[10] Teerapittayanon, S., McDanel, B., and Kung, H. T., 2016, "BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks," Proc. of ICPR.

[11] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W., 2022, "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.

[12] Łuczak, D., 2024, "Mechanical Vibrations Analysis in Direct Drive Using CWT with Complex Morlet Wavelet — A Survey," Mechanical Systems and Signal Processing, focusing on EMI / non-AWGN robustness in industrial bearing diagnostics.

[13] Smith, W. A., and Randall, R. B., 2015, "Rolling Element Bearing Diagnostics Using the Case Western Reserve University Data: A Benchmark Study," Mechanical Systems and Signal Processing, 64–65, 100–131.

---

## 작성 메모

- §1–§4 는 *수치 독립* (데이터·메소드·셋업) 으로 학습 전 작성, §5·§7 는 학습 종료 후 채움 (2026-05-07 evening).
- §3.2 에 *signal energy ↔ noise level confound* 명시 (2026-05-08 추가) — KA15/KA22 case study 의 sectional foundation.
- §5.6 (per-class F1) 와 §5.7 (per-bearing accuracy + KA15/KA22 case study) 는 5-method 비교 표 (`paper/PER_BEARING_ANALYSIS_2026-05-08.md` 의 분석 결과) 를 paper 본문에 반영.
- §6 는 *방법론적 한계* (KA15/KA22 routing failure, -20 dB collapse) 와 *셋업적 한계* 로 구분 작성. 가장 큰 paper-defining limitation 은 quality estimator 의 confound 이며, 이는 dataset OOD 가 아닌 method-specific 결함임을 정직 보고.
- 참고문헌 [9] (Zhang et al. 2017, anti-noise CNN), [12] (Łuczak 2024, EMI robustness), [13] (Smith & Randall 2015, CWRU benchmark) 확정.
- abstract_ko.md v8.1 (2026-05-07 갱신) 의 정량 수치 (+21.7 pp / 약 62% / +4.9 pp) 는 §5 / §7 과 일치.
- Figures: 기존 F2-F4 (SNR-stratified, routing, Pareto) + 신규 T1-T4 (fault signature). T4 가 main body 통합 후보.
