"""
DeepFM Tutorial: AutoGluon 커스텀 모델 사용 예제

이 튜토리얼은 DeepFM 모델을 AutoGluon에 통합하고 사용하는 방법을 보여줍니다.
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

print("=== DeepFM Tutorial ===")
print("DeepFM 모델을 AutoGluon에 통합하고 학습하는 예제입니다.")

# 데이터 로드
print("\n1. 데이터 로드 중...")
df = pd.read_csv("datasets/creditcard.csv")
df["Class"] = df["Class"].astype("category")

# 학습/검증 분리
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)

print(f"학습 데이터 크기: {train_data.shape}")
print(f"테스트 데이터 크기: {test_data.shape}")
print(f"클래스 분포:\n{train_data['Class'].value_counts()}")

# DeepFM 모델 학습 (Cosine Annealing 스케줄러 포함)
print("\n2. DeepFM 모델 학습 중...")
predictor = TabularPredictor(label="Class", problem_type="binary", eval_metric="f1", path="models/deepfm_tutorial").fit(
    train_data,
    hyperparameters={
        "DEEPFM": {
            "fm_dropout": 0.1,
            "fm_embedding_dim": 10,
            "deep_output_size": 64,
            "deep_hidden_size": 64,
            "deep_dropout": 0.1,
            "deep_layers": 2,
            # Cosine Annealing Learning Rate Scheduler 설정
            "lr_scheduler": True,
            "scheduler_type": "cosine",
            "lr_scheduler_min_lr": 1e-6,
        }
    },
    time_limit=300,  # 5분
    verbosity=4,  # 최고 verbosity (가장 자세한 로그)
)

# 결과 확인
print("\n3. 학습 결과 확인...")
print("리더보드:")
print(predictor.leaderboard())

# 예측
print("\n4. 예측 수행...")
predictions = predictor.predict(test_data)
probabilities = predictor.predict_proba(test_data)

print(f"예측 결과 형태: {predictions.shape}")
print(f"확률 예측 형태: {probabilities.shape}")

# 성능 평가
print("\n5. 성능 평가...")
from sklearn.metrics import classification_report, confusion_matrix
print("분류 리포트:")
print(classification_report(test_data["Class"].astype(int), predictions.astype(int)))

print("\n=== DeepFM Tutorial 완료! ===") 