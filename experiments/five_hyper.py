import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_deepfm_torch_model import TabularDeepFMTorchModel
from custom_models.tabular_dcnv2_torch_model import TabularDCNv2TorchModel
from custom_models.focal_loss_implementation import CustomFocalDLModel
from custom_models.custom_nn_torch_model import CustomNNTorchModel
from sklearn.model_selection import train_test_split
from autogluon.common import space

# 모델 등록
ag_model_registry.add(TabularDeepFMTorchModel)
ag_model_registry.add(TabularDCNv2TorchModel)
ag_model_registry.add(CustomFocalDLModel)
ag_model_registry.add(CustomNNTorchModel)

# 데이터 로드
df = pd.read_csv("datasets/creditcard.csv")
df["Class"] = df["Class"].astype("category")  # AutoGluon에서 분류로 인식하게

# 학습/검증 분리
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)

print(f"학습 데이터 크기: {train_data.shape}")
print(f"테스트 데이터 크기: {test_data.shape}")
print(f"학습 데이터 클래스 분포:\n{train_data['Class'].value_counts()}")
print(f"학습 데이터 클래스 비율:\n{train_data['Class'].value_counts(normalize=True)}")

print("\n=== 4개 모델 동시 학습 (DCNv2, CustomFocalDLModel, RandomForest, CustomNNTorchModel) ===")

# 5개 모델을 한 번에 학습
predictor = TabularPredictor(label="Class", problem_type="binary", eval_metric="f1", path="models/five_models_experiment").fit(
    train_data,
    hyperparameters={
        # DeepFM - 수치형 특성만 있는 경우 부적합 (주석처리)
        # "DEEPFM": {
        #     "fm_dropout": 0.1,
        #     "fm_embedding_dim": 16,  # 논문에서 권장하는 임베딩 차원
        #     "deep_output_size": 128,  # 논문에서 권장하는 출력 크기
        #     "deep_hidden_size": 128,  # 논문에서 권장하는 히든 크기
        #     "deep_dropout": 0.1,
        #     "deep_layers": 3,  # 논문에서 권장하는 레이어 수
        #     # 조기 종료 설정
        #     'epochs_wo_improve': 15,  # 충분한 학습 시간
        #     'num_epochs': 25,         # 충분한 학습 시간
        #     # Cosine Annealing Learning Rate Scheduler 설정
        #     "lr_scheduler": True,
        #     "scheduler_type": "cosine",
        #     "lr_scheduler_min_lr": 1e-6,
        # },
        # DCNv2 - 수치형 특성에 적합한 모델 (아키텍처 확인용 작은 설정)
        "DCNV2": {
       "num_cross_layers": space.Int(1, 3, default=2),
            "cross_dropout": space.Real(0.1, 0.3, default=0.2),
         
        },
        # CustomFocalDLModel (Focal Loss 사용) - Cosine Annealing 스케줄러 포함
        # 'CUSTOM_FOCAL_DL': [{
        #     'max_batch_size': 512,
        #     'num_epochs': 5,  # 아키텍처 확인용 작은 설정
        #     'epochs_wo_improve': 3,  # 아키텍처 확인용 작은 설정
        #     'optimizer': 'adam',
        #     'learning_rate': 0.001,
        #     'weight_decay': 0.0001,
        #     'dropout_prob': 0.1,
        #     'num_layers': 4,
        #     'hidden_size': 128,
        #     'activation': 'relu',
        #     'lr_scheduler': True,
        #     'scheduler_type': 'cosine',
        #     'lr_scheduler_min_lr': 1e-6,
        # }],
        # RandomForest - 기본 설정 (주석 처리)
        # "RF": {
        #     "n_estimators": 100,
        #     "max_depth": 10,
        #     "min_samples_split": 5,
        #     "min_samples_leaf": 2,
        #     "criterion": "gini",
        # },
        # CustomNNTorchModel (일반 CrossEntropy) - 아키텍처 확인용 작은 설정
        # 'CUSTOM_NN_TORCH': [{
        #     'max_batch_size': 512,
        #     'num_epochs': 5,  # 아키텍처 확인용 작은 설정
        #     'epochs_wo_improve': 3,  # 아키텍처 확인용 작은 설정
        #     'optimizer': 'adam',
        #     'learning_rate': 0.001,
        #     'weight_decay': 0.0001,
        #     'dropout_prob': 0.1,
        #     'num_layers': 4,
        #     'hidden_size': 128,
        #     'activation': 'relu',
        #     'lr_scheduler': True,
        #     'scheduler_type': 'cosine',
        #     'lr_scheduler_min_lr': 1e-6,
        # }],
    },
    time_limit=900,  # 15분
    verbosity=4,  # 최고 verbosity (가장 자세한 로그)
    hyperparameter_tune_kwargs={
            'scheduler': 'local',
            'searcher': 'random',
            'num_trials': 2,  # 빠른 테스트를 위해 2번 시도로 제한
        }
)

print("\n=== 학습 완료! 결과 확인 ===")
print("리더보드:")
print(predictor.leaderboard())

print("\n=== 모델별 상세 정보 ===")
leaderboard = predictor.leaderboard()
for idx, row in leaderboard.iterrows():
    model_name = row['model']
    score = row['score_val']
    fit_time = row['fit_time_marginal']
    print(f"\n{model_name}:")
    print(f"  - 성능 점수: {score:.4f}")
    print(f"  - 학습 시간: {fit_time:.2f}초")

print("\n=== 최고 성능 모델 ===")
# get_model_best() 대신 leaderboard에서 최고 성능 모델 찾기
best_row = leaderboard.loc[leaderboard['score_val'].idxmax()]
best_model = best_row['model']
print(f"최고 성능 모델: {best_model}")
print(f"성능 점수: {best_row['score_val']:.4f}")

print("\n=== 모델 성능 비교 ===")
print("모델별 성능 순위:")
for idx, row in leaderboard.iterrows():
    if 'WeightedEnsemble' not in row['model']:  # 앙상블 제외하고 개별 모델만
        print(f"{idx+1}. {row['model']}: F1 = {row['score_val']:.4f}")

print("\n=== Focal Loss vs 일반 CrossEntropy 비교 ===")
focal_score = None
nn_torch_score = None

for idx, row in leaderboard.iterrows():
    if 'CUSTOM_FOCAL_DL' in row['model']:
        focal_score = row['score_val']
    elif 'CUSTOM_NN_TORCH' in row['model']:
        nn_torch_score = row['score_val']

if focal_score is not None and nn_torch_score is not None:
    print(f"CUSTOM_FOCAL_DL (Focal Loss): {focal_score:.4f}")
    print(f"CUSTOM_NN_TORCH (일반 CrossEntropy): {nn_torch_score:.4f}")
    if focal_score > nn_torch_score:
        print("✅ Focal Loss가 더 우수한 성능을 보였습니다!")
    elif focal_score < nn_torch_score:
        print("❌ 일반 CrossEntropy가 더 우수한 성능을 보였습니다.")
    else:
        print("🤝 두 모델의 성능이 동일합니다.")
else:
    print("Focal Loss와 일반 CrossEntropy 비교를 위한 데이터가 부족합니다.")

print("\n=== 예측 테스트 ===")
# 테스트 데이터로 예측
predictions = predictor.predict(test_data)
probabilities = predictor.predict_proba(test_data)

print(f"예측 결과 형태: {predictions.shape}")
print(f"확률 예측 형태: {probabilities.shape}")
print(f"예측값 샘플: {predictions.head()}")
print(f"확률값 샘플:\n{probabilities.head()}")

print("\n=== 5개 모델 학습 완료! ===") 