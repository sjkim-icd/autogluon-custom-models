# AutoGluon Custom Models: DeepFM & DCNv2

이 프로젝트는 AutoGluon 프레임워크에 커스텀 딥러닝 모델(DeepFM, DCNv2)을 통합하고, Focal Loss를 활용한 불균형 데이터 처리 모델을 구현한 프로젝트입니다.

## 🚀 주요 기능

- **DeepFM (Factorization-Machine based Neural Network)**: Factorization Machine과 Deep Neural Network를 결합한 모델
- **DCNv2 (Deep & Cross Network v2)**: Cross Network에 Low-rank Factorization을 적용한 모델
- **CustomFocalDLModel**: 클래스 불균형 문제를 해결하기 위한 Focal Loss 구현
- **AutoGluon 통합**: 커스텀 모델들을 AutoGluon의 하이퍼파라미터 튜닝 시스템과 완전히 통합

## 📁 프로젝트 구조

```
autogluon_env_cursor/
├── README.md                           # 프로젝트 설명서
├── requirements.txt                    # 의존성 패키지 목록
├── LICENSE                            # MIT 라이선스
├── .gitignore                         # Git 제외 파일 목록
├── datasets/                           # 데이터셋 폴더
│   └── creditcard.csv                 # 신용카드 사기 탐지 데이터셋
├── custom_models/                      # 커스텀 모델 구현
│   ├── __init__.py
│   ├── tabular_deepfm_torch_model.py   # DeepFM AutoGluon 진입점
│   ├── tabular_dcnv2_torch_model.py    # DCNv2 AutoGluon 진입점
│   ├── deepfm_block.py                 # DeepFM 네트워크 구현
│   ├── dcnv2_block.py                  # DCNv2 네트워크 구현
│   ├── focal_loss_implementation.py    # Focal Loss 구현 및 CustomFocalDLModel
│   └── focal_loss.py                   # Focal Loss 클래스 구현
├── experiments/                        # 실험 스크립트
│   ├── three_models_combined.py        # 3개 모델 고정 하이퍼파라미터 학습
│   ├── four_models_combined.py         # 4개 모델 하이퍼파라미터 튜닝
│   └── focal_loss_experiment.py        # Focal Loss 실험 전용 스크립트
├── tutorials/                          # 사용 예제 및 튜토리얼
│   ├── deepfm_tutorial.py              # DeepFM 단독 학습 예제
│   └── dcnv2_tutorial.py               # DCNv2 단독 학습 예제
└── models/                             # 학습된 모델 저장 폴더 (Git 제외)
    ├── three_models_experiment/        # 3개 모델 실험 결과
    ├── four_models_experiment/         # 4개 모델 실험 결과
    ├── deepfm_tutorial/                # DeepFM 튜토리얼 결과
    └── dcnv2_tutorial/                 # DCNv2 튜토리얼 결과
```

## 🛠️ 설치 및 설정

### 1. 환경 설정
```bash
# 가상환경 생성 (권장)
python -m venv autogluon_env
source autogluon_env/bin/activate  # Linux/Mac
# 또는
autogluon_env\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비
`datasets/` 폴더에 `creditcard.csv` 파일을 위치시킵니다.

## 🚀 사용 방법

### 기본 사용법 (3개 모델 고정 하이퍼파라미터)
```bash
cd experiments
python three_models_combined.py
```

### 하이퍼파라미터 튜닝 (4개 모델)
```bash
cd experiments
python four_models_combined.py
```

### 개별 모델 학습
```bash
# DeepFM 단독 학습
python tutorials/deepfm_tutorial.py

# DCNv2 단독 학습
python tutorials/dcnv2_tutorial.py
```

## 📊 모델 설명

### 1. DeepFM (Factorization-Machine based Neural Network)
- **특징**: Factorization Machine과 Deep Neural Network 결합
- **장점**: 저차원과 고차원 특성 상호작용을 모두 학습
- **적용**: 추천 시스템, CTR 예측 등

### 2. DCNv2 (Deep & Cross Network v2)
- **특징**: Cross Network에 Low-rank Factorization 적용
- **장점**: 효율적인 특성 상호작용 학습, 파라미터 수 감소
- **적용**: 대규모 스파스 데이터 처리

### 3. CustomFocalDLModel
- **특징**: Focal Loss를 사용한 클래스 불균형 처리
- **장점**: 소수 클래스에 대한 학습 성능 향상
- **적용**: 불균형 데이터셋 (사기 탐지, 의료 진단 등)

## 🔧 하이퍼파라미터 튜닝

### DeepFM 하이퍼파라미터
- `fm_dropout`: 0.1 ~ 0.3
- `fm_embedding_dim`: 8 ~ 16
- `deep_output_size`: 32 ~ 128
- `deep_hidden_size`: 32 ~ 128
- `deep_dropout`: 0.1 ~ 0.3
- `deep_layers`: 1 ~ 3

### DCNv2 하이퍼파라미터
- `num_cross_layers`: 1 ~ 3
- `cross_dropout`: 0.1 ~ 0.3
- `low_rank`: 8 ~ 32
- `deep_output_size`: 32 ~ 128
- `deep_hidden_size`: 32 ~ 128
- `deep_dropout`: 0.1 ~ 0.3
- `deep_layers`: 1 ~ 3

### CustomFocalDLModel 하이퍼파라미터
- `learning_rate`: 0.0001 ~ 0.01
- `weight_decay`: 0.00001 ~ 0.001
- `dropout_prob`: 0.1 ~ 0.3
- `layers`: [100, 50], [200, 100], [300, 150]
- `activation`: relu, tanh, leaky_relu
- `optimizer`: adam, sgd, adamw

## 📈 성능 결과

### 신용카드 사기 탐지 데이터셋 결과
- **데이터셋**: 284,807개 샘플 (492개 사기, 284,315개 정상)
- **평가 지표**: F1 Score
- **최고 성능**: CustomFocalDLModel (F1: 0.8571)

## 🔍 주요 특징

1. **AutoGluon 완전 통합**: 커스텀 모델들이 AutoGluon의 모든 기능과 호환
2. **동적 차원 처리**: 입력/출력 차원이 데이터셋에 따라 자동 조정
3. **하이퍼파라미터 튜닝**: AutoGluon의 HPO 시스템과 완전 통합
4. **클래스 불균형 처리**: Focal Loss를 통한 효과적인 불균형 데이터 학습
5. **모듈화된 구조**: 각 모델이 독립적으로 사용 가능

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 Issues를 통해 문의해주세요.



