import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_deepfm_torch_model import TabularDeepFMTorchModel
from custom_models.tabular_dcnv2_torch_model import TabularDCNv2TorchModel
from custom_models.tabular_dcnv2_fuxictr_torch_model_fixed import TabularDCNv2FuxiCTRTorchModel
from custom_models.focal_loss_implementation import CustomFocalDLModel
from custom_models.custom_nn_torch_model import CustomNNTorchModel
from sklearn.model_selection import train_test_split
from autogluon.common import space

# 모델 등록
ag_model_registry.add(TabularDeepFMTorchModel)
ag_model_registry.add(TabularDCNv2TorchModel)
ag_model_registry.add(TabularDCNv2FuxiCTRTorchModel)
ag_model_registry.add(CustomFocalDLModel)
ag_model_registry.add(CustomNNTorchModel)

# 데이터 로드
df = pd.read_csv("datasets/creditcard.csv")
df["Class"] = df["Class"].astype("category")  # AutoGluon에서 분류로 인식하게
# 학습 데이터만 분리 (AutoGluon이 자동으로 holdout 분할 수행)
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)

print(f"학습 데이터 크기: {train_data.shape}")
print(f"학습 데이터 클래스 분포:\n{train_data['Class'].value_counts()}")
print(f"학습 데이터 클래스 비율:\n{train_data['Class'].value_counts(normalize=True)}")

print("\n=== 4개 모델 동시 학습 (DCNv2, CustomFocalDLModel, RandomForest, CustomNNTorchModel) ===")

# 5개 모델을 한 번에 학습 (AutoGluon 자동 holdout 분할 사용)
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
        # DCNv2 - 수치형 특성에 적합한 모델 (최고 성능 하이퍼파라미터 적용)
        "DCNV2": {
            "num_cross_layers": 2,
            "cross_dropout": 0.1181,
            "low_rank": 29,
            "deep_output_size": 98,
            "deep_hidden_size": 91,
            "deep_dropout": 0.1583,
            "deep_layers": 3,
            'epochs_wo_improve': 5,
            'num_epochs': 20,
            "lr_scheduler": True,
            "scheduler_type": "cosine",
            "lr_scheduler_min_lr": 1e-6,
            "learning_rate": 0.000629,
            "weight_decay": 5.68e-12,
            "dropout_prob": 0.5,
            "activation": "relu",
            "optimizer": "adam",
            "hidden_size": 128,
            "use_batchnorm": True,
        },
        # FuxiCTR DCNv2 - MoE 구조의 DCNv2 (최고 성능 하이퍼파라미터 적용)
        "DCNV2_FUXICTR": {
            "num_cross_layers": 2,
            "cross_dropout": 0.1181,
            "low_rank": 29,
            "deep_output_size": 98,
            "deep_hidden_size": 91,
            "deep_dropout": 0.1583,
            "deep_layers": 3,
            'epochs_wo_improve': 5,
            'num_epochs': 20,
            "lr_scheduler": True,
            "scheduler_type": "cosine",
            "lr_scheduler_min_lr": 1e-6,
            "learning_rate": 0.000629,
            "weight_decay": 5.68e-12,
            "dropout_prob": 0.5,
            "activation": "relu",
            "optimizer": "adam",
            "hidden_size": 128,
            "use_batchnorm": True,
            # FuxiCTR 특화 파라미터
            "use_low_rank_mixture": True,
            "num_experts": 4,
            "model_structure": "parallel",
        },
        # CustomFocalDLModel (Focal Loss 사용) - Cosine Annealing 스케줄러 포함
        'CUSTOM_FOCAL_DL': [{
            'max_batch_size': 512,
            'num_epochs': 20,
            'epochs_wo_improve': 5,
            'optimizer': 'adam',
            'learning_rate': 0.0008,  # Focal Loss에 적합한 LR
            'weight_decay': 0.0001,
            'dropout_prob': 0.1,
            'num_layers': 4,
            'hidden_size': 128,
            'activation': 'relu',
            'lr_scheduler': True,
            'scheduler_type': 'cosine',
            'lr_scheduler_min_lr': 1e-6,
        }],
        # RandomForest - 기본 설정
        "RF": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "criterion": "gini",
        },
        # CustomNNTorchModel (일반 CrossEntropy) - 스케줄러 포함
        'CUSTOM_NN_TORCH': [{
            'max_batch_size': 512,
            'num_epochs': 20,
            'epochs_wo_improve': 5,
            'optimizer': 'adam',
            'learning_rate': 0.0005,  # 더 안정적인 LR
            'weight_decay': 0.0001,
            'dropout_prob': 0.1,
            'num_layers': 4,
            'hidden_size': 128,
            'activation': 'relu',
            'lr_scheduler': True,
            'scheduler_type': 'cosine',
            'lr_scheduler_min_lr': 1e-6,
        }],
    },
    time_limit=900,  # 15분
    verbosity=4,  # 최고 verbosity (가장 자세한 로그)
    # holdout_frac=0.2,  # 검증 데이터를 20%로 늘림
)

print("\n=== 학습 완료! 결과 확인 ===")
print("리더보드 (검증 데이터 기준):")
print(predictor.leaderboard())

print("\n=== 리더보드 (테스트 데이터 기준) ===")
print(predictor.leaderboard(data=test_data))

print("\n=== 모델별 상세 정보 (검증 데이터 기준) ===")
leaderboard_val = predictor.leaderboard()
for idx, row in leaderboard_val.iterrows():
    model_name = row['model']
    score = row['score_val']
    fit_time = row['fit_time_marginal']
    print(f"\n{model_name}:")
    print(f"  - 검증 성능 점수: {score:.4f}")
    print(f"  - 학습 시간: {fit_time:.2f}초")

print("\n=== 모델별 상세 정보 (테스트 데이터 기준) ===")
leaderboard_test = predictor.leaderboard(data=test_data)
for idx, row in leaderboard_test.iterrows():
    model_name = row['model']
    score = row['score_val']
    print(f"\n{model_name}:")
    print(f"  - 테스트 성능 점수: {score:.4f}")

print("\n=== 최고 성능 모델 (검증 데이터 기준) ===")
best_row_val = leaderboard_val.loc[leaderboard_val['score_val'].idxmax()]
best_model_val = best_row_val['model']
print(f"최고 성능 모델: {best_model_val}")
print(f"검증 성능 점수: {best_row_val['score_val']:.4f}")

print("\n=== 최고 성능 모델 (테스트 데이터 기준) ===")
best_row_test = leaderboard_test.loc[leaderboard_test['score_val'].idxmax()]
best_model_test = best_row_test['model']
print(f"최고 성능 모델: {best_model_test}")
print(f"테스트 성능 점수: {best_row_test['score_val']:.4f}")

print("\n=== 모델 성능 비교 (검증 데이터 기준) ===")
print("모델별 성능 순위:")
for idx, row in leaderboard_val.iterrows():
    if 'WeightedEnsemble' not in row['model']:  # 앙상블 제외하고 개별 모델만
        print(f"{idx+1}. {row['model']}: F1 = {row['score_val']:.4f}")

print("\n=== 모델 성능 비교 (테스트 데이터 기준) ===")
print("모델별 성능 순위:")
for idx, row in leaderboard_test.iterrows():
    if 'WeightedEnsemble' not in row['model']:  # 앙상블 제외하고 개별 모델만
        print(f"{idx+1}. {row['model']}: F1 = {row['score_val']:.4f}")

print("\n=== Focal Loss vs 일반 CrossEntropy 비교 (검증 데이터 기준) ===")
focal_score_val = None
nn_torch_score_val = None

for idx, row in leaderboard_val.iterrows():
    if 'CUSTOM_FOCAL_DL' in row['model']:
        focal_score_val = row['score_val']
    elif 'CUSTOM_NN_TORCH' in row['model']:
        nn_torch_score_val = row['score_val']

if focal_score_val is not None and nn_torch_score_val is not None:
    print(f"CUSTOM_FOCAL_DL (Focal Loss): {focal_score_val:.4f}")
    print(f"CUSTOM_NN_TORCH (일반 CrossEntropy): {nn_torch_score_val:.4f}")
    if focal_score_val > nn_torch_score_val:
        print("✅ Focal Loss가 더 우수한 성능을 보였습니다!")
    elif focal_score_val < nn_torch_score_val:
        print("❌ 일반 CrossEntropy가 더 우수한 성능을 보였습니다.")
    else:
        print("🤝 두 모델의 성능이 동일합니다.")
else:
    print("Focal Loss와 일반 CrossEntropy 비교를 위한 데이터가 부족합니다.")

print("\n=== Focal Loss vs 일반 CrossEntropy 비교 (테스트 데이터 기준) ===")
focal_score_test = None
nn_torch_score_test = None

for idx, row in leaderboard_test.iterrows():
    if 'CUSTOM_FOCAL_DL' in row['model']:
        focal_score_test = row['score_val']
    elif 'CUSTOM_NN_TORCH' in row['model']:
        nn_torch_score_test = row['score_val']

if focal_score_test is not None and nn_torch_score_test is not None:
    print(f"CUSTOM_FOCAL_DL (Focal Loss): {focal_score_test:.4f}")
    print(f"CUSTOM_NN_TORCH (일반 CrossEntropy): {nn_torch_score_test:.4f}")
    if focal_score_test > nn_torch_score_test:
        print("✅ Focal Loss가 더 우수한 성능을 보였습니다!")
    elif focal_score_test < nn_torch_score_test:
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