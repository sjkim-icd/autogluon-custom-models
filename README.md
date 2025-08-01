# AutoGluon Custom Deep Learning Models

> AutoGluon 프레임워크에 커스텀 딥러닝 모델(DeepFM, DCNv2, CustomNN)을 통합하고, Focal Loss를 활용한 불균형 데이터 처리 모델을 구현한 프로젝트입니다.

## 📋 목차

- [🚀 주요 기능](#-주요-기능)
- [📁 프로젝트 구조](#-프로젝트-구조)
- [🛠️ 설치 및 설정](#️-설치-및-설정)
- [🚀 사용 방법](#-사용-방법)
- [📊 모델 설명](#-모델-설명)
- [🔧 하이퍼파라미터 튜닝](#-하이퍼파라미터-튜닝)
- [📈 성능 결과](#-성능-결과)
- [🔍 주요 특징](#-주요-특징)
- [🤝 기여하기](#-기여하기)
- [📝 라이선스](#-라이선스)

## 🚀 주요 기능

### 🧠 커스텀 딥러닝 모델
- **DeepFM**: Factorization Machine과 Deep Neural Network 결합
- **DCNv2**: Cross Network에 Low-rank Factorization 적용
- **CustomNNTorchModel**: 일반적인 신경망 모델 (CrossEntropy Loss)
- **CustomFocalDLModel**: 클래스 불균형 문제 해결을 위한 Focal Loss 구현

### 🔧 AutoGluon 통합
- 커스텀 모델들을 AutoGluon의 하이퍼파라미터 튜닝 시스템과 완전히 통합
- 동적 차원 처리로 다양한 데이터셋에 자동 적용
- 앙상블 학습으로 최적 성능 보장
- 학습률 스케줄러 지원 (Cosine, OneCycle, Plateau 등)

## 📁 프로젝트 구조

```
autogluon_env_cursor/
├── 📄 README.md                           # 프로젝트 설명서
├── 📄 requirements.txt                    # 의존성 패키지 목록
├── 📄 LICENSE                            # MIT 라이선스
├── 📄 .gitignore                         # Git 제외 파일 목록
├── 📁 datasets/                          # 데이터셋 폴더
│   └── 📄 creditcard.csv                 # 신용카드 사기 탐지 데이터셋
├── 📁 custom_models/                     # 커스텀 모델 구현
│   ├── 📄 __init__.py
│   ├── 📄 tabular_deepfm_torch_model.py   # DeepFM AutoGluon 진입점
│   ├── 📄 tabular_dcnv2_torch_model.py    # DCNv2 AutoGluon 진입점
│   ├── 📄 custom_nn_torch_model.py        # CustomNN AutoGluon 진입점
│   ├── 📄 deepfm_block.py                 # DeepFM 네트워크 구현
│   ├── 📄 dcnv2_block.py                  # DCNv2 네트워크 구현
│   ├── 📄 focal_loss_implementation.py    # Focal Loss 구현 및 CustomFocalDLModel
│   └── 📄 focal_loss.py                   # Focal Loss 클래스 구현
├── 📁 experiments/                        # 실험 스크립트
│   ├── 📄 five_models_combined.py         # 5개 모델 하이퍼파라미터 튜닝
│   ├── 📄 five_hyper.py                   # 5개 모델 하이퍼파라미터 검색
│   ├── 📄 hyperparameter_search.py        # 하이퍼파라미터 검색 유틸리티
│   ├── 📄 hyperparameter_search_autogluon.py  # AutoGluon 하이퍼파라미터 검색
│   └── 📄 test_deepfm_simple.py           # DeepFM 간단 테스트
├── 📁 tutorials/                          # 사용 예제 및 튜토리얼
│   ├── 📄 deepfm_tutorial.py              # DeepFM 단독 학습 예제
│   ├── 📄 dcnv2_tutorial.py               # DCNv2 단독 학습 예제
│   ├── 📄 learning_rate_scheduler_tutorial.py  # 학습률 스케줄러 튜토리얼
│   └── 📄 simple_lr_scheduler_tutorial.py     # 간단한 LR 스케줄러 튜토리얼
└── 📁 models/                             # 학습된 모델 저장 폴더 (Git 제외)
    ├── 📁 five_models_experiment/         # 5개 모델 실험 결과
    ├── 📁 deepfm_tutorial/                # DeepFM 튜토리얼 결과
    ├── 📁 dcnv2_tutorial/                 # DCNv2 튜토리얼 결과
    ├── 📁 deepfm_no_scheduler/            # DeepFM (스케줄러 없음)
    ├── 📁 deepfm_onecycle_scheduler/      # DeepFM (OneCycle 스케줄러)
    ├── 📁 deepfm_cosine_scheduler/        # DeepFM (Cosine 스케줄러)
    └── 📁 deepfm_plateau_scheduler/       # DeepFM (Plateau 스케줄러)
```

## 🛠️ 설치 및 설정

### 1️⃣ 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv autogluon_env
source autogluon_env/bin/activate  # Linux/Mac
# 또는
autogluon_env\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2️⃣ 데이터 준비

`datasets/` 폴더에 `creditcard.csv` 파일을 위치시킵니다.

## 🚀 사용 방법

### 🎯 기본 사용법 (5개 모델 하이퍼파라미터 튜닝)

```bash
cd experiments
python five_models_combined.py
```

### 🔍 하이퍼파라미터 검색

```bash
# 5개 모델 하이퍼파라미터 검색
python experiments/five_hyper.py

# AutoGluon 하이퍼파라미터 검색
python experiments/hyperparameter_search_autogluon.py
```

### 📚 개별 모델 학습

```bash
# DeepFM 단독 학습
python tutorials/deepfm_tutorial.py

# DCNv2 단독 학습
python tutorials/dcnv2_tutorial.py
```

### 🔧 학습률 스케줄러 튜토리얼

```bash
# 학습률 스케줄러 튜토리얼
python tutorials/learning_rate_scheduler_tutorial.py

# 간단한 LR 스케줄러 튜토리얼
python tutorials/simple_lr_scheduler_tutorial.py
```

## 📊 모델 설명

### 🧠 DeepFM (Factorization-Machine based Neural Network)

| 항목 | 설명 |
|------|------|
| **특징** | Factorization Machine과 Deep Neural Network 결합 |
| **장점** | 저차원과 고차원 특성 상호작용을 모두 학습 |
| **적용** | 추천 시스템, CTR 예측 등 |

### 🔗 DCNv2 (Deep & Cross Network v2)

| 항목 | 설명 |
|------|------|
| **특징** | Cross Network에 Low-rank Factorization 적용 |
| **장점** | 효율적인 특성 상호작용 학습, 파라미터 수 감소 |
| **적용** | 추천 시스템, CTR 예측, 대규모 스파스 데이터 처리 |

### 🧠 CustomNNTorchModel

| 항목 | 설명 |
|------|------|
| **특징** | 일반적인 신경망 모델 (CrossEntropy Loss) |
| **장점** | 간단하고 안정적인 성능, 학습률 스케줄러 지원 |
| **적용** | 일반적인 분류 문제 |

### ⚖️ CustomFocalDLModel

| 항목 | 설명 |
|------|------|
| **특징** | Focal Loss를 사용한 클래스 불균형 처리 |
| **장점** | 소수 클래스에 대한 학습 성능 향상 |
| **적용** | 불균형 데이터셋 (사기 탐지, 의료 진단 등) |

## 🔧 하이퍼파라미터 튜닝

### 🧠 DeepFM 하이퍼파라미터

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `fm_dropout` | 0.1 ~ 0.3 | FM 레이어 드롭아웃 |
| `fm_embedding_dim` | 8 ~ 16 | 임베딩 차원 |
| `deep_output_size` | 32 ~ 128 | 딥 네트워크 출력 크기 |
| `deep_hidden_size` | 32 ~ 128 | 딥 네트워크 은닉층 크기 |
| `deep_dropout` | 0.1 ~ 0.3 | 딥 네트워크 드롭아웃 |
| `deep_layers` | 1 ~ 3 | 딥 네트워크 레이어 수 |

### 🔗 DCNv2 하이퍼파라미터

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `num_cross_layers` | 1 ~ 3 | 크로스 네트워크 레이어 수 |
| `cross_dropout` | 0.1 ~ 0.3 | 크로스 네트워크 드롭아웃 |
| `low_rank` | 8 ~ 32 | 저차원 분해 크기 |
| `deep_output_size` | 32 ~ 128 | 딥 네트워크 출력 크기 |
| `deep_hidden_size` | 32 ~ 128 | 딥 네트워크 은닉층 크기 |
| `deep_dropout` | 0.1 ~ 0.3 | 딥 네트워크 드롭아웃 |
| `deep_layers` | 1 ~ 3 | 딥 네트워크 레이어 수 |

### 🧠 CustomNNTorchModel 하이퍼파라미터

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `learning_rate` | 0.0001 ~ 0.01 | 학습률 |
| `weight_decay` | 0.00001 ~ 0.001 | 가중치 감쇠 |
| `dropout_prob` | 0.1 ~ 0.3 | 드롭아웃 확률 |
| `layers` | [100,50], [200,100], [300,150] | 네트워크 구조 |
| `activation` | relu, tanh, leaky_relu | 활성화 함수 |
| `optimizer` | adam, sgd, adamw | 최적화 알고리즘 |

### ⚖️ CustomFocalDLModel 하이퍼파라미터

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `learning_rate` | 0.0001 ~ 0.01 | 학습률 |
| `weight_decay` | 0.00001 ~ 0.001 | 가중치 감쇠 |
| `dropout_prob` | 0.1 ~ 0.3 | 드롭아웃 확률 |
| `layers` | [100,50], [200,100], [300,150] | 네트워크 구조 |
| `activation` | relu, tanh, leaky_relu | 활성화 함수 |
| `optimizer` | adam, sgd, adamw | 최적화 알고리즘 |

## 📈 성능 결과

### 🎯 신용카드 사기 탐지 데이터셋 결과

| 모델 | 검증 F1 | 테스트 F1 | 학습시간 | 특징 |
|------|---------|-----------|----------|------|
| **DCNV2_FUXICTR** | 0.8571 | 0.8148 | 188.52초 | 최고 성능 (Best Performance) |
| **DCNV2** | 0.8571 | 0.7143 | 119.29초 | 빠른 학습 (Fast Learning) |
| **CUSTOM_FOCAL_DL** | 0.7500 | 0.7979 | 243.59초 | Focal Loss |
| **WeightedEnsemble_L2** | 0.8571 | 0.7143 | 0.17초 | 앙상블 (Ensemble) |

### 🏆 주요 성과

- **최고 성능**: DCNV2_FUXICTR (검증 F1: 0.8571)
- **가장 빠른 학습**: WeightedEnsemble_L2 (0.17초)
- **안정적 성능**: DCNV2 (검증 F1: 0.8571)
- **불균형 처리**: CUSTOM_FOCAL_DL (Focal Loss 적용)

### 📊 데이터셋 정보

| 항목 | 값 |
|------|-----|
| **전체 데이터 크기** | 284,807개 샘플 |
| **정상 거래** | 284,315개 (99.83%) |
| **사기 거래** | 492개 (0.17%) |
| **평가 지표** | F1 Score |
| **데이터 불균형 비율** | 1:577 (매우 심한 불균형) |

## 🔍 주요 특징

### ✅ AutoGluon 완전 통합
- 커스텀 모델들이 AutoGluon의 모든 기능과 호환
- 하이퍼파라미터 튜닝, 앙상블 학습 자동화

### ✅ 동적 차원 처리
- 입력/출력 차원이 데이터셋에 따라 자동 조정
- 다양한 데이터셋에 즉시 적용 가능

### ✅ 클래스 불균형 처리
- Focal Loss를 통한 효과적인 불균형 데이터 학습
- 소수 클래스 성능 향상

### ✅ 모듈화된 구조
- 각 모델이 독립적으로 사용 가능
- 새로운 커스텀 모델 추가 용이

### ✅ 학습률 스케줄러 지원
- 다양한 학습률 스케줄러 구현
- OneCycle, Cosine, Plateau 등 지원

### ✅ 하이퍼파라미터 검색 도구
- 체계적인 하이퍼파라미터 검색 기능
- AutoGluon 통합 검색 도구 제공

## 🤝 기여하기

1. **Fork** the repository
2. **Create** your feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

## 📝 라이선스

이 프로젝트는 Apache License 2.0 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

**라이선스 관련 참고사항:**
- AutoGluon 프레임워크는 Apache License 2.0을 사용합니다


---



