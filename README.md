# AutoGluon Custom Deep Learning Models with Optuna HPO

> AutoGluon 프레임워크에 커스텀 딥러닝 모델을 통합하고, Optuna를 활용한  하이퍼파라미터 최적화(HPO)와 실험 관리 시스템을 구현한 프로젝트입니다.

## 📋 목차

- [🚀 주요 기능](#-주요-기능)
- [📁 프로젝트 구조](#-프로젝트-구조)
- [🛠️ 설치 및 설정](#️-설치-및-설정)
- [🚀 사용 방법](#-사용-방법)
- [📊 모델 설명](#-모델-설명)
- [🔧 Optuna HPO 시스템](#-optuna-hpo-시스템)
- [📈 실험 관리](#-실험-관리)
- [📊 분석 대시보드](#-분석-대시보드)
- [🔍 주요 특징](#-주요-특징)
- [🤝 기여하기](#-기여하기)
- [📝 라이선스](#-라이선스)

## 🚀 주요 기능

### 🧠 커스텀 딥러닝 모델
- **DCNV2**: Deep & Cross Network v2 구현
- **DCNV2_FUXICTR**: DCNv2 with Mixture-of-Experts
- **CustomNNTorchModel**: 일반적인 신경망 모델 (CrossEntropy Loss)
- **CustomFocalDLModel**: 클래스 불균형 문제 해결을 위한 Focal Loss 구현
- **RandomForest**: 트리 기반 앙상블 모델

### 🔧 AutoGluon + Optuna 통합
- 커스텀 모델들을 AutoGluon의 하이퍼파라미터 튜닝 시스템과 통합
- **Optuna를 활용한 고급 HPO**: Bayesian Optimization, Random Search
- **통합 DB 시스템**: 모든 실험을 단일 SQLite DB에 저장
- **실험별 폴더 구조**: 각 실험의 결과를 독립적으로 관리
- **실시간 모니터링**: Optuna Dashboard로 실시간 진행 상황 확인

### OPTUNA 결과 분석

- **HTML 대시보드**: 인터랙티브 차트와 필터링 기능
- **Excel 보고서**: 상세한 분석 결과와 조건부 서식
- **파라미터 중요도 분석**: Optuna의 자동 중요도 계산
- **최적화 과정 분석**: 수렴성, 안정성 평가
- **상관관계 분석**: 하이퍼파라미터 간 상관관계 시각화

## 📁 프로젝트 구조

```
autogluon_env_cursor/
├── 📄 README.md                           # 프로젝트 설명서
├── 📄 requirements.txt                    # 의존성 패키지 목록
├── 📄 LICENSE                            # MIT 라이선스
├── 📄 .gitignore                         # Git 제외 파일 목록
├── 📁 datasets/                          # 데이터셋 폴더
│   ├── 📄 creditcard.csv                 # 신용카드 사기 탐지 데이터셋
│   └── 📄 titanic.csv                    # 타이타닉 생존 예측 데이터셋
├── 📁 custom_models/                     # 커스텀 모델 구현
│   ├── 📄 __init__.py
│   ├── 📄 tabular_dcnv2_torch_model.py    # DCNv2 AutoGluon 진입점
│   ├── 📄 tabular_dcnv2_fuxictr_torch_model_fixed.py  # DCNv2 FuxiCTR
│   ├── 📄 custom_nn_torch_model.py        # CustomNN AutoGluon 진입점
│   ├── 📄 focal_loss_implementation.py    # Focal Loss 구현 및 CustomFocalDLModel
│   ├── 📄 dcnv2_block.py                  # DCNv2 네트워크 구현
│   └── 📄 dcnv2_block_fuxictr.py         # DCNv2 FuxiCTR 네트워크 구현
├── 📁 experiments/                        # 실험 스크립트
│   ├── 📄 optuna_single_stage_hpo_unified_db.py  # Titanic 데이터 HPO
│   ├── 📄 optuna_single_stage_hpo_credit_card.py # Credit Card 데이터 HPO
│   ├── 📄 run_experiment.bat              # Windows 배치 스크립트
│   └── 📄 run_experiment.ps1              # PowerShell 스크립트
├── 📁 analysis/                           # 분석 스크립트
│   └── 📄 create_final_unified_dashboard_excel_fixed.py  # 대시보드 생성
├── 📁 optuna_studies/                     # Optuna 실험 DB
│   ├── 📁 titanic_5models_hpo_v1/        # Titanic 실험 DB
│   ├── 📁 credit_card_5models_hpo_v1/    # Credit Card 실험 DB
│   └── 📁 {experiment_name}/              # 실험별 DB 폴더
├── 📁 results/                            # 실험 결과 폴더
│   ├── 📁 titanic_5models_hpo_v1/        # Titanic 실험 결과
│   ├── 📁 credit_card_5models_hpo_v1/    # Credit Card 실험 결과
│   └── 📁 {experiment_name}/              # 실험별 결과 폴더
├── 📁 models/                             # 학습된 모델 저장 폴더
└── 📁 backup/                             # 백업 파일들
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

### 2️⃣ 추가 패키지 설치

```bash
# Optuna 관련 패키지
pip install optuna kaleido

# Excel 파일 생성용
pip install openpyxl

# 대시보드 시각화용
pip install plotly
```

### 3️⃣ 데이터 준비

`datasets/` 폴더에 다음 파일들을 위치시킵니다:
- `creditcard.csv`: 신용카드 사기 탐지 데이터셋
- `titanic.csv`: 타이타닉 생존 예측 데이터셋

## 🚀 사용 방법

### 🎯 1단계: HPO 실험 실행

#### Titanic 데이터 실험
```bash
# 직접 실행
python experiments/optuna_single_stage_hpo_unified_db.py "titanic_5models_hpo_v1"

# 또는 배치 스크립트 사용
run_experiment.bat "titanic_5models_hpo_v1"
```

#### Credit Card 데이터 실험
```bash
# 직접 실행
python experiments/optuna_single_stage_hpo_credit_card.py "credit_card_5models_hpo_v1"

# 또는 배치 스크립트 사용
run_experiment.bat "credit_card_5models_hpo_v1"
```

### 🔄 2단계: 연속 실행 (HPO + 분석)

```bash
# 방법 1: 직접 연결
python experiments/optuna_single_stage_hpo_unified_db.py "titanic_5models_hpo_v1" ; python analysis/create_final_unified_dashboard_excel_fixed.py "titanic_5models_hpo_v1"

# 방법 2: 배치 스크립트
run_experiment.bat "titanic_5models_hpo_v1"

# 방법 3: PowerShell 스크립트
.\run_experiment.ps1 "titanic_5models_hpo_v1"
```

### 📊 3단계: 분석 대시보드 생성

```bash
python analysis/create_final_unified_dashboard_excel_fixed.py "experiment_name"
```

## 📊 모델 설명

### 🧠 커스텀 모델들

#### DCNV2 (Deep & Cross Network v2)/DCNV2_FUXICTR
- **특징**: Cross Network와 Deep Network의 결합
- **장점**: 고차원 특성 상호작용 학습, 효율적인 계산
- **적용**: 범주형 + 수치형 데이터 혼합


#### CustomFocalDLModel
- **특징**: Focal Loss를 사용한 불균형 데이터 처리
- **장점**: 클래스 불균형 문제 해결, 소수 클래스 성능 향상
- **적용**: 사기 탐지, 의료 진단 등 불균형 데이터

#### CustomNNTorchModel
- **특징**: 일반적인 신경망 (CrossEntropy Loss)
- **장점**: 안정적인 학습, 다양한 데이터에 적용 가능
- **적용**: 일반적인 분류 문제

#### RandomForest
- **특징**: 트리 기반 앙상블 모델
- **장점**: 해석 가능성, 과적합 방지
- **적용**: 모든 분류 문제

## 🔧 Optuna HPO 시스템

### 🎯 HPO 구성
- **각 모델당 15 trials**: 총 75 trials (5개 모델)
- **HPO 방법**: Bayesian Optimization
- **메트릭**: F1 Score (불균형 데이터에 적합)
- **시간 제한**: 모델당 10분, 전체 20분

### 📊 하이퍼파라미터 검색 공간

#### 딥러닝 모델들 (DCNV2, DCNV2_FUXICTR, CUSTOM_FOCAL_DL, CUSTOM_NN_TORCH)
```python
{
    'learning_rate': [1e-4, 1e-2],  # 로그 스케일
    'weight_decay': [1e-6, 1e-3],   # 로그 스케일
    'dropout_prob': [0.1, 0.2, 0.3],
    'num_layers': [3, 4, 5],
    'hidden_size': [128, 256, 512],
    'num_epochs': [15, 20, 25]
}
```

#### Focal Loss 모델 추가 파라미터
```python
{
    'focal_alpha': [0.25, 0.5, 0.75],
    'focal_gamma': [1.0, 2.0, 3.0]
}
```

#### RandomForest
```python
{
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2]
}
```

### 🔄 통합 DB 시스템
- **단일 SQLite DB**: `optuna_studies/{experiment_name}/all_studies.db`
- **실험별 분리**: 각 실험의 DB가 독립적으로 관리
- **지속성**: 실험 중단 후 재시작 가능
- **확장성**: 새로운 실험 추가 용이

## 📈 실험 관리

### 📁 실험별 폴더 구조
```
optuna_studies/
├── titanic_5models_hpo_v1/
│   └── all_studies.db
├── credit_card_5models_hpo_v1/
│   └── all_studies.db
└── {experiment_name}/
    └── all_studies.db

results/
├── titanic_5models_hpo_v1/
│   ├── optuna_advanced_report_*.xlsx
│   └── optuna_unified_dashboard_*.html
├── credit_card_5models_hpo_v1/
│   ├── optuna_advanced_report_*.xlsx
│   └── optuna_unified_dashboard_*.html
└── {experiment_name}/
    ├── optuna_advanced_report_*.xlsx
    └── optuna_unified_dashboard_*.html
```

### 🔍 실험 모니터링
```bash
# Optuna Dashboard 실행
optuna-dashboard sqlite:///optuna_studies/{experiment_name}/all_studies.db

# 웹 브라우저에서 접속
http://localhost:8080
```

## 📈 성능 결과

### 🏆 Titanic 데이터셋 실험 결과 (`titanic_5models_hpo_v1`)

| 모델 | 최고 성능 | 평균 성능 | 표준편차 | 특징 |
|------|-----------|-----------|----------|------|
| **DCNV2_FUXICTR** | 0.9811 | 0.9679 | 0.0086 | 🥇 **최고 성능, 안정적** |
| **CUSTOM_NN_TORCH** | 0.9811 | 0.9782 | 0.0039 | 🥈 **가장 안정적, 일관적** |
| **CUSTOM_FOCAL_DL** | 0.9811 | 0.9183 | 0.1438 | 🥉 **최고 성능, 변동성 있음** |
| **RF** | 0.9682 | 0.9620 | 0.0063 | **안정적, 해석 가능** |
| **DCNV2** | 0.9744 | 0.8728 | 0.1663 | **최고 성능, 높은 변동성** |

### 📊 주요 발견사항

#### 🏅 **최고 성능 모델들**
- **DCNV2_FUXICTR, CUSTOM_NN_TORCH, CUSTOM_FOCAL_DL**: 모두 0.9811의 최고 성능
- **CUSTOM_NN_TORCH**: 가장 안정적 (표준편차 0.0039)
- **DCNV2_FUXICTR**: 높은 성능 + 안정성 (표준편차 0.0086)

#### ⚠️ **변동성이 큰 모델들**
- **DCNV2**: 높은 최고 성능이지만 변동성 큼 (표준편차 0.1663)
- **CUSTOM_FOCAL_DL**: 최고 성능이지만 불안정 (표준편차 0.1438)

#### 🎯 **권장 모델**
- **CUSTOM_NN_TORCH**: 일관성과 성능의 최적 균형
- **DCNV2_FUXICTR**: 높은 성능과 안정성
- **RF**: 해석 가능성과 안정성

### 🔍 실험 설정
- **데이터셋**: Titanic 생존 예측 (이진 분류)
- **데이터 분할**: 80% 학습, 20% 테스트 (Stratified)
- **평가 메트릭**: F1 Score
- **HPO 설정**: 각 모델당 15 trials, 총 75 trials
- **실험 시간**: 약 1시간 이내

## 📊 분석 대시보드

### 🌐 HTML 대시보드 기능
- **최적화 과정 차트**: 실시간 성능 변화 추이
- **파라미터 중요도**: 각 하이퍼파라미터의 영향도
- **상관관계 분석**: 하이퍼파라미터 간 상관관계
- **Parallel Coordinate Plot**: 다차원 파라미터 공간 시각화
- **Contour Plot**: 2차원 파라미터 공간 최적화 영역
- **Slice Plot**: 개별 파라미터 영향 분석
- **필터링 기능**: 각 차트별 독립적인 필터
- **사용자 지정 권장사항**: 다음 실험을 위한 제안사항

### 📊 Excel 보고서 기능
- **요약 시트**: 실험 개요 및 주요 결과
- **개별 모델 시트**: 각 모델의 상세 분석
- **파라미터 중요도**: 정렬된 중요도 차트
- **최적화 과정**: 수렴성 및 안정성 분석
- **권장사항**: 다음 실험을 위한 구체적 제안
- **조건부 서식**: 성능별 색상 구분

---

## 🚀 빠른 시작

```bash
# 1. 환경 설정
python -m venv autogluon_env
autogluon_env\Scripts\activate  # Windows

# 2. 의존성 설치
pip install -r requirements.txt
pip install optuna kaleido openpyxl plotly

# 3. Titanic 실험 실행
python experiments/optuna_single_stage_hpo_unified_db.py "titanic_5models_hpo_v1"

# 4. 분석 대시보드 생성
python analysis/create_final_unified_dashboard_excel_fixed.py "titanic_5models_hpo_v1"

# 5. 결과 확인
# - HTML: results/titanic_5models_hpo_v1/optuna_unified_dashboard_*.html
# - Excel: results/titanic_5models_hpo_v1/optuna_advanced_report_*.xlsx
```



