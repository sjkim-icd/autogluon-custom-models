"""
Learning Rate Scheduler Tutorial: AutoGluon 내장 Learning Rate Scheduling 사용 예제

이 튜토리얼은 AutoGluon에서 내장된 learning rate scheduling 기능을 사용하는 방법을 보여줍니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_deepfm_torch_model import TabularDeepFMTorchModel
from sklearn.model_selection import train_test_split

# DeepFM 모델 등록
ag_model_registry.add(TabularDeepFMTorchModel)

print("=== AutoGluon 내장 Learning Rate Scheduler Tutorial ===")
print("AutoGluon에서 내장된 Learning Rate Scheduling 기능을 사용하는 예제입니다.")

# 데이터 로드
print("\n1. 데이터 로드 중...")
df = pd.read_csv("datasets/creditcard.csv")
df["Class"] = df["Class"].astype("category")

# 학습/검증 분리
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)

print(f"학습 데이터 크기: {train_data.shape}")
print(f"테스트 데이터 크기: {test_data.shape}")
print(f"클래스 분포:\n{train_data['Class'].value_counts()}")

# 1. ReduceLROnPlateau 스케줄러로 DeepFM 모델 학습
print("\n2. ReduceLROnPlateau 스케줄러로 DeepFM 모델 학습...")
predictor_plateau = TabularPredictor(
    label="Class", 
    problem_type="binary", 
    eval_metric="f1", 
    path="models/deepfm_plateau_scheduler"
).fit(
    train_data,
    hyperparameters={
        "DEEPFM": {
            "fm_dropout": 0.1,
            "fm_embedding_dim": 10,
            "deep_output_size": 64,
            "deep_hidden_size": 64,
            "deep_dropout": 0.1,
            "deep_layers": 2,
            # Learning Rate Scheduler 설정
            "lr_scheduler": True,
            "scheduler_type": "plateau",
            "lr_scheduler_patience": 3,
            "lr_scheduler_factor": 0.2,
            "lr_scheduler_min_lr": 1e-6,
        }
    },
    time_limit=300,
    verbosity=4,
)

print("ReduceLROnPlateau 결과:")
print(predictor_plateau.leaderboard())

# 2. Cosine Annealing 스케줄러로 DeepFM 모델 학습
print("\n3. Cosine Annealing 스케줄러로 DeepFM 모델 학습...")
predictor_cosine = TabularPredictor(
    label="Class", 
    problem_type="binary", 
    eval_metric="f1", 
    path="models/deepfm_cosine_scheduler"
).fit(
    train_data,
    hyperparameters={
        "DEEPFM": {
            "fm_dropout": 0.1,
            "fm_embedding_dim": 10,
            "deep_output_size": 64,
            "deep_hidden_size": 64,
            "deep_dropout": 0.1,
            "deep_layers": 2,
            # Learning Rate Scheduler 설정
            "lr_scheduler": True,
            "scheduler_type": "cosine",
            "lr_scheduler_min_lr": 1e-6,
        }
    },
    time_limit=300,
    verbosity=4,
)

print("Cosine Annealing 결과:")
print(predictor_cosine.leaderboard())

# 3. One Cycle 스케줄러로 DeepFM 모델 학습
print("\n4. One Cycle 스케줄러로 DeepFM 모델 학습...")
predictor_onecycle = TabularPredictor(
    label="Class", 
    problem_type="binary", 
    eval_metric="f1", 
    path="models/deepfm_onecycle_scheduler"
).fit(
    train_data,
    hyperparameters={
        "DEEPFM": {
            "fm_dropout": 0.1,
            "fm_embedding_dim": 10,
            "deep_output_size": 64,
            "deep_hidden_size": 64,
            "deep_dropout": 0.1,
            "deep_layers": 2,
            # Learning Rate Scheduler 설정
            "lr_scheduler": True,
            "scheduler_type": "onecycle",
            "lr_scheduler_min_lr": 1e-6,
        }
    },
    time_limit=300,
    verbosity=4,
)

print("One Cycle 결과:")
print(predictor_onecycle.leaderboard())

# 4. 스케줄러 없이 학습 (비교용)
print("\n5. 스케줄러 없이 DeepFM 모델 학습 (비교용)...")
predictor_no_scheduler = TabularPredictor(
    label="Class", 
    problem_type="binary", 
    eval_metric="f1", 
    path="models/deepfm_no_scheduler"
).fit(
    train_data,
    hyperparameters={
        "DEEPFM": {
            "fm_dropout": 0.1,
            "fm_embedding_dim": 10,
            "deep_output_size": 64,
            "deep_hidden_size": 64,
            "deep_dropout": 0.1,
            "deep_layers": 2,
            # Learning Rate Scheduler 비활성화
            "lr_scheduler": False,
        }
    },
    time_limit=300,
    verbosity=4,
)

print("스케줄러 없음 결과:")
print(predictor_no_scheduler.leaderboard())

# 결과 비교
print("\n=== 스케줄러별 성능 비교 ===")
print("1. ReduceLROnPlateau:")
leaderboard_plateau = predictor_plateau.leaderboard()
if not leaderboard_plateau.empty:
    best_plateau = leaderboard_plateau.loc[leaderboard_plateau['score_val'].idxmax()]
    print(f"   최고 성능: {best_plateau['score_val']:.4f}")

print("2. Cosine Annealing:")
leaderboard_cosine = predictor_cosine.leaderboard()
if not leaderboard_cosine.empty:
    best_cosine = leaderboard_cosine.loc[leaderboard_cosine['score_val'].idxmax()]
    print(f"   최고 성능: {best_cosine['score_val']:.4f}")

print("3. One Cycle:")
leaderboard_onecycle = predictor_onecycle.leaderboard()
if not leaderboard_onecycle.empty:
    best_onecycle = leaderboard_onecycle.loc[leaderboard_onecycle['score_val'].idxmax()]
    print(f"   최고 성능: {best_onecycle['score_val']:.4f}")

print("4. 스케줄러 없음:")
leaderboard_no_scheduler = predictor_no_scheduler.leaderboard()
if not leaderboard_no_scheduler.empty:
    best_no_scheduler = leaderboard_no_scheduler.loc[leaderboard_no_scheduler['score_val'].idxmax()]
    print(f"   최고 성능: {best_no_scheduler['score_val']:.4f}")

print("\n=== Learning Rate Scheduler Tutorial 완료! ===") 