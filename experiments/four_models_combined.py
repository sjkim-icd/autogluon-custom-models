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
from sklearn.model_selection import train_test_split
from autogluon.common import space

# 모델 등록
ag_model_registry.add(TabularDeepFMTorchModel)
ag_model_registry.add(TabularDCNv2TorchModel)
ag_model_registry.add(CustomFocalDLModel)

# 데이터 로드 - 경로 수정
df = pd.read_csv("../datasets/creditcard.csv")
df["Class"] = df["Class"].astype("category")  # AutoGluon에서 분류로 인식하게

# 학습/검증 분리
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)

print(f"학습 데이터 크기: {train_data.shape}")
print(f"테스트 데이터 크기: {test_data.shape}")
print(f"학습 데이터 클래스 분포:\n{train_data['Class'].value_counts()}")
print(f"학습 데이터 클래스 비율:\n{train_data['Class'].value_counts(normalize=True)}")

print("\n=== 4개 모델 Hyperparameter Tuning (DeepFM, DCNv2, CustomFocalDLModel, RandomForest) ===")

# 4개 모델을 Hyperparameter Tuning과 함께 학습
predictor = TabularPredictor(label="Class", problem_type="binary", eval_metric="f1", path="models/four_models_experiment").fit(
    train_data,
    hyperparameters={
        # DeepFM - Hyperparameter Space
        "DEEPFM": {
            "fm_dropout": space.Real(0.1, 0.3, default=0.2),
            "fm_embedding_dim": space.Int(8, 16, default=10),
            "deep_output_size": space.Int(32, 128, default=64),
            "deep_hidden_size": space.Int(32, 128, default=64),
            "deep_dropout": space.Real(0.1, 0.3, default=0.2),
            "deep_layers": space.Int(1, 3, default=2),
        },
        # DCNv2 - Hyperparameter Space
        "DCNV2": {
            "num_cross_layers": space.Int(1, 3, default=2),
            "cross_dropout": space.Real(0.1, 0.3, default=0.2),
            "low_rank": space.Int(8, 32, default=16),
            "deep_output_size": space.Int(32, 128, default=64),
            "deep_hidden_size": space.Int(32, 128, default=64),
            "deep_dropout": space.Real(0.1, 0.3, default=0.2),
            "deep_layers": space.Int(1, 3, default=2),
        },
        # CustomFocalDLModel (Focal Loss 사용) - Hyperparameter Space
        "CustomFocalDLModel": {
            "num_dataloading_workers": space.Int(0, 2, default=0),
            "max_batch_size": space.Int(256, 1024, default=512),
            "num_epochs": space.Int(5, 15, default=10),
            "epochs_wo_improve": space.Int(3, 7, default=5),
            "optimizer": space.Categorical("adam", "sgd", "adamw"),
            "learning_rate": space.Real(0.0001, 0.01, default=0.001),
            "weight_decay": space.Real(0.00001, 0.001, default=0.0001),
            "dropout_prob": space.Real(0.1, 0.3, default=0.2),
            "layers": space.Categorical([100, 50], [200, 100], [300, 150]),
            "activation": space.Categorical("relu", "tanh", "leaky_relu"),
        },
        # RandomForest - Hyperparameter Space
        "RF": {
            "n_estimators": space.Int(50, 200, default=100),
            "max_depth": space.Int(5, 15, default=10),
            "min_samples_split": space.Int(2, 10, default=5),
            "min_samples_leaf": space.Int(1, 5, default=2),
            "criterion": space.Categorical("gini", "entropy"),
        },
    },
    time_limit=600,  # 10분
    verbosity=4,  # 최고 verbosity (가장 자세한 로그)
    num_trials=2,  # 각 모델당 2번 trial
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

print("\n=== 예측 테스트 ===")
# 테스트 데이터로 예측
predictions = predictor.predict(test_data)
probabilities = predictor.predict_proba(test_data)

print(f"예측 결과 형태: {predictions.shape}")
print(f"확률 예측 형태: {probabilities.shape}")
print(f"예측값 샘플: {predictions.head()}")
print(f"확률값 샘플:\n{probabilities.head()}")

print("\n=== 4개 모델 Hyperparameter Tuning 완료! ===") 