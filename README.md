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

### 🎯 Optuna HPO 시스템
- **AutoGluon + Optuna 완벽 통합**: 커스텀 모델들의 자동 하이퍼파라미터 최적화
- **실시간 실험 모니터링**: Optuna Dashboard를 통한 실시간 진행 상황 추적
- **HTML/Excel 대시보드**: 인터랙티브 차트와 상세 분석 보고서 자동 생성
- **통합 DB 실험 관리**: 모든 실험을 체계적으로 관리하는 SQLite 기반 시스템
- **다중 데이터셋 지원**: Titanic, Credit Card 등 다양한 데이터셋에 적용 가능

### 📊 IV/WOE 분석 시스템

- **자동 변수 선택**: Information Value (IV) 기반 특성 선택
- **WOE 계산**: Weight of Evidence 값 자동 계산 및 시각화
- **데이터 전처리**: 결측치 처리, 범주형 변수 인코딩, 연속형 변수 이산화
- **Excel 출력**: 상세한 분석 결과를 Excel 파일로 저장 (여러 시트)
- **HTML 대시보드**: Plotly 기반 인터랙티브 차트 (확대/축소, 호버 정보)
- **검증 시스템**: WOE/IV 계산 과정을 Excel 수식으로 검증 가능
- **유연한 입력**: CSV, Excel, Parquet 파일 지원, 임계값 조정 가능

### 🔍 범용 EDA 도구

- **다양한 EDA 패키지**: ydata_profiling, Sweetviz, Autoviz, Klib, D-Tale 통합
- **선택적 실행**: `--packages` 옵션으로 원하는 EDA 도구만 선택 실행
- **자동 데이터 처리**: CSV, Excel, Parquet 파일 자동 인식 및 로드
- **체계적 결과 저장**: 데이터셋별로 체계적인 폴더 구조 생성
- **한글 지원**: Malgun Gothic 폰트로 한글 데이터 완벽 지원
- **샘플링 옵션**: 대용량 데이터를 위한 계층화 샘플링 지원
- **웹 기반 분석**: D-Tale을 통한 웹 브라우저 기반 인터랙티브 분석

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
├── 📁 experiments/                        # 실험 스크립트 (최신 Optuna HPO 시스템)
│   ├── 📄 optuna_single_stage_hpo_multi_dataset.py      # 최신 HPO 시스템 (Titanic + Credit Card)
│   ├── 📄 optuna_single_stage_hpo_unified_db.py         # 통합 DB HPO 시스템
│   ├── 📄 run_experiment.bat              # Windows 배치 스크립트
│   └── 📄 run_experiment.ps1              # PowerShell 스크립트
├── 📁 analysis/                           # 분석 스크립트
│   └── 📄 create_final_unified_dashboard_excel_fixed.py  # 대시보드 생성
├── 📁 EDA/                                # 탐색적 데이터 분석 도구
│   ├── 📄 iv_woe_analysis.py             # IV/WOE 분석 메인 스크립트
│   └── 📄 universal_eda_tool.py          # 범용 EDA 도구 (최신 버전)
├── 📁 outputs/                            # 분석 결과 출력 폴더
│   ├── 📁 iv_woe_analysis/               # IV/WOE 분석 결과
│   └── 📁 notebooks/                      # Jupyter 노트북
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
    ├── 📁 backup_optuna/                  # Optuna 중간 버전 파일들
    └── 📁 iv_analysis_versions/           # IV 분석 중간 버전 파일들
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

#### 최신 Optuna HPO 시스템 사용 (권장)
```bash
# 다중 데이터셋 지원 HPO 시스템
python experiments/optuna_single_stage_hpo_multi_dataset.py

# 통합 DB HPO 시스템
python experiments/optuna_single_stage_hpo_unified_db.py
```

#### 기존 실험 (레거시)
```bash
# Titanic 데이터 실험
python experiments/optuna_single_stage_hpo_unified_db.py "titanic_5models_hpo_v1"

# Credit Card 데이터 실험
python experiments/optuna_single_stage_hpo_credit_card.py "credit_card_5models_hpo_v1"
```

#### 배치 스크립트 사용
```bash
# Windows 배치 스크립트
run_experiment.bat "titanic_5models_hpo_v1"

# PowerShell 스크립트
.\run_experiment.ps1 "titanic_5models_hpo_v1"
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

### 🔍 4단계: IV/WOE 분석 실행

#### 기본 사용법
```bash
# Titanic 데이터 분석 (기본 임계값 0.02)
python EDA/iv_woe_analysis.py --datapath datasets/titanic.csv --target_col survived --eda_name titanic_analysis

# Credit Card 데이터 분석
python EDA/iv_woe_analysis.py --datapath datasets/creditcard.csv --target_col Class --eda_name credit_analysis

# 임계값 조정 (0.2로 설정)
python EDA/iv_woe_analysis.py --datapath datasets/creditcard.csv --target_col Class --threshold 0.2 --eda_name credit_threshold_02

# 특정 변수만 분석
python EDA/iv_woe_analysis.py --datapath datasets/titanic.csv --target_col survived --feature_cols pclass fare --eda_name titanic_selected
```

#### 출력 파일
- **Excel 파일**: `{eda_name}_results.xlsx` (IV 요약, WOE 상세, 검증 데이터)
- **HTML 대시보드**: `{eda_name}_complete_analysis.html` (인터랙티브 차트)
- **저장 위치**: `outputs/iv_woe_analysis/` 폴더

#### Excel 시트 구성
1. **IV_Summary**: 변수별 IV 값과 선택/제거 상태
2. **WOE_Details**: 각 변수의 bin별 WOE 값과 통계
3. **Verification_Details**: WOE/IV 계산 과정 검증 (Excel 수식 포함)
4. **Data_Summary**: 데이터셋 기본 정보

### 🔍 5단계: 범용 EDA 도구 실행

#### 기본 사용법
```bash
# Titanic 데이터 EDA (ydata_profiling만 실행)
python EDA/universal_eda_tool.py --data_path datasets/titanic.csv --dataset_name titanic_eda --packages ydata_profiling

# Credit Card 데이터 EDA (여러 패키지 실행)
python EDA/universal_eda_tool.py --data_path datasets/creditcard.csv --dataset_name credit_eda --packages ydata_profiling sweetviz autoviz

# 모든 EDA 패키지 실행
python EDA/universal_eda_tool.py --data_path datasets/titanic.csv --dataset_name titanic_full_eda
```

### 🎯 6단계: Optuna HPO 시스템 실행

#### 최신 HPO 시스템 (권장)
```bash
# 다중 데이터셋 지원 HPO 시스템
python experiments/optuna_single_stage_hpo_multi_dataset.py

# 통합 DB HPO 시스템
python experiments/optuna_single_stage_hpo_unified_db.py
```

#### 실시간 모니터링
```bash
# Optuna Dashboard 실행
optuna-dashboard sqlite:///optuna_studies/{experiment_name}/all_studies.db

# 웹 브라우저에서 접속
http://localhost:8080
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

### 🎯 최신 HPO 시스템 특징

#### 1. **optuna_single_stage_hpo_multi_dataset.py** (최신 권장)
- **다중 데이터셋 지원**: Titanic, Credit Card 등 다양한 데이터셋에 적용
- **통합 모델 관리**: 5개 커스텀 모델의 일괄 HPO
- **자동 데이터 로딩**: 데이터셋별 자동 전처리 및 로딩
- **최적화된 하이퍼파라미터**: 각 모델별 맞춤형 검색 공간

#### 2. **optuna_single_stage_hpo_unified_db.py** (통합 DB)
- **단일 SQLite DB**: 모든 실험을 하나의 DB에 통합 관리
- **실험 지속성**: 중단 후 재시작 가능
- **효율적인 저장**: 중복 데이터 제거 및 최적화된 저장 구조

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

### 📊 실시간 모니터링
- **Optuna Dashboard**: `optuna-dashboard` 명령어로 실시간 모니터링
- **웹 기반 시각화**: http://localhost:8080에서 실시간 실험 진행 상황 확인
- **Parallel Coordinate Plot**: 다차원 파라미터 공간 시각화
- **Contour Plot**: 2차원 파라미터 공간 최적화 영역 시각화

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

## 📊 IV/WOE 분석 시스템

### 🔍 주요 기능
- **자동 데이터 전처리**: 결측치 처리, 범주형 변수 인코딩, 연속형 변수 이산화
- **IV 기반 변수 선택**: Information Value를 통한 예측력 있는 변수 자동 선택
- **WOE 계산**: 각 bin별 Weight of Evidence 값 자동 계산
- **시각화**: Plotly 기반 인터랙티브 차트 (IV 비교, WOE 분포, IV 기여도)
- **검증 시스템**: Excel 수식을 통한 WOE/IV 계산 과정 검증

### 📈 IV 해석 가이드
- **IV < 0.02**: 예측력 없음 (제거 권장)
- **0.02 ≤ IV < 0.1**: 약한 예측력
- **0.1 ≤ IV < 0.3**: 중간 예측력
- **0.3 ≤ IV < 0.5**: 강한 예측력
- **IV ≥ 0.5**: 매우 강한 예측력 (과적합 위험)

### 📊 WOE 해석 가이드
- **WOE > 0**: 해당 bin에서 타겟 변수 값이 높음 (예: 생존, 사기)
- **WOE < 0**: 해당 bin에서 타겟 변수 값이 낮음 (예: 사망, 정상)
- **WOE = 0**: 해당 bin에서 타겟 변수 값이 동일

### 🔢 IV Contribution 해석
- **높은 IV Contribution**: 해당 bin이 변수의 예측력에 크게 기여
- **낮은 IV Contribution**: 해당 bin이 변수의 예측력에 적게 기여
- **음수 IV Contribution**: 해당 bin이 예측력을 저하시킴

### 💡 사용 시나리오
- **특성 선택**: 모델 학습 전 예측력 있는 변수 선별
- **데이터 품질 검증**: 변수의 예측력과 분포 확인
- **비즈니스 인사이트**: 각 변수가 타겟에 미치는 영향 분석
- **모델 해석**: 변수별 기여도와 중요도 파악

---

## 🚀 빠른 시작

```bash
# 1. 환경 설정
python -m venv autogluon_env
autogluon_env\Scripts\activate  # Windows

# 2. 의존성 설치
pip install -r requirements.txt
pip install optuna kaleido openpyxl plotly

# 3. 최신 Optuna HPO 시스템 실행 (권장)
python experiments/optuna_single_stage_hpo_multi_dataset.py

# 4. 분석 대시보드 생성
python analysis/create_final_unified_dashboard_excel_fixed.py "experiment_name"

# 5. 결과 확인
# - HTML: results/{experiment_name}/optuna_unified_dashboard_*.html
# - Excel: results/{experiment_name}/optuna_advanced_report_*.xlsx

# 6. IV/WOE 분석 실행 (선택사항)
python EDA/iv_woe_analysis.py --datapath datasets/titanic.csv --target_col survived --eda_name titanic_iv_analysis

# 7. 범용 EDA 도구 실행 (선택사항)
python EDA/universal_eda_tool.py --data_path datasets/titanic.csv --dataset_name titanic_eda --packages ydata_profiling

# 8. 결과 확인
# - IV/WOE Excel: outputs/iv_woe_analysis/titanic_iv_analysis_results.xlsx
# - IV/WOE HTML: outputs/iv_woe_analysis/titanic_iv_analysis_complete_analysis.html
# - EDA 결과: EDA/titanic_eda/ 폴더
```

## 🔍 주요 특징

### 🎯 **Optuna HPO 시스템**
- **AutoGluon + Optuna 완벽 통합**: 커스텀 모델들의 자동 하이퍼파라미터 최적화
- **실시간 모니터링**: Optuna Dashboard를 통한 실시간 진행 상황 추적
- **통합 DB 관리**: 모든 실험을 체계적으로 관리하는 SQLite 기반 시스템
- **다중 데이터셋 지원**: 다양한 데이터셋에 적용 가능한 범용 시스템

### 📊 **IV/WOE 분석 시스템**
- **자동 특성 선택**: Information Value 기반 예측력 있는 변수 자동 선별
- **상세한 검증**: Excel 수식을 통한 WOE/IV 계산 과정 검증
- **인터랙티브 시각화**: Plotly 기반 확대/축소, 호버 정보 지원
- **유연한 입력**: 다양한 파일 형식과 임계값 조정 지원

### 🔍 **범용 EDA 도구**
- **선택적 실행**: 원하는 EDA 도구만 선택하여 실행
- **체계적 관리**: 데이터셋별 체계적인 결과 저장
- **한글 지원**: 한글 데이터 완벽 지원
- **웹 기반 분석**: D-Tale을 통한 인터랙티브 분석



