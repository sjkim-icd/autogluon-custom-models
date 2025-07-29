"""
Simple Learning Rate Scheduler Tutorial: AutoGluon 내장 기능 사용

AutoGluon에서 내장된 learning rate scheduling 기능을 간단하게 사용하는 예제입니다.
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

print("=== AutoGluon 내장 Learning Rate Scheduler 간단 예제 ===")

# 데이터 로드
print("\n1. 데이터 로드 중...")
df = pd.read_csv("datasets/creditcard.csv")
df["Class"] = df["Class"].astype("category")

# 학습/검증 분리
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)

print(f"학습 데이터 크기: {train_data.shape}")
print(f"테스트 데이터 크기: {test_data.shape}")

# AutoGluon 내장 기능으로 학습
print("\n2. AutoGluon 내장 Learning Rate Scheduling으로 학습...")
predictor = TabularPredictor(
    label="Class", 
    problem_type="binary", 
    eval_metric="f1", 
    path="models/simple_lr_scheduler"
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
            # AutoGluon 내장 learning rate 관련 파라미터
            "learning_rate": 0.001,  # 초기 learning rate
            "epochs_wo_improve": 5,  # validation loss 개선 없을 때 early stopping
            "num_epochs": 50,        # 최대 epoch 수
        }
    },
    time_limit=300,
    verbosity=4,
)

# 결과 확인
print("\n3. 학습 결과 확인...")
print("리더보드:")
print(predictor.leaderboard())

# 예측 및 평가
print("\n4. 예측 및 평가...")
predictions = predictor.predict(test_data)
probabilities = predictor.predict_proba(test_data)

from sklearn.metrics import classification_report, confusion_matrix
print("분류 리포트:")
print(classification_report(test_data["Class"], predictions))

print("\n=== AutoGluon 내장 Learning Rate Scheduler 예제 완료! ===")
print("\n참고: AutoGluon은 자동으로 다음과 같은 기능을 제공합니다:")
print("- Early stopping: validation loss가 개선되지 않으면 학습 중단")
print("- Learning rate 조정: 내부적으로 적절한 learning rate scheduling 적용")
print("- 모델 앙상블: 여러 모델의 결과를 조합하여 최종 예측") 