import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_dcnv2_fuxictr_torch_model_fixed import TabularDCNv2FuxiCTRTorchModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

# 모델 등록
ag_model_registry.add(TabularDCNv2FuxiCTRTorchModel)

# 데이터 로드
df = pd.read_csv("datasets/creditcard.csv")
df["Class"] = df["Class"].astype("category")  # AutoGluon에서 분류로 인식하게

# 학습/검증/테스트 분리
train_data, temp_data = train_test_split(df, test_size=0.3, stratify=df["Class"], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data["Class"], random_state=42)

print(f"학습 데이터 크기: {train_data.shape}")
print(f"검증 데이터 크기: {val_data.shape}")
print(f"테스트 데이터 크기: {test_data.shape}")
print(f"학습 데이터 클래스 분포:\n{train_data['Class'].value_counts()}")

# 0.8235 성능을 달성한 하이퍼파라미터 (FuxiCTR 스타일에 맞게 조정)
best_params = {
    'num_cross_layers': 2,
    'cross_dropout': 0.1181,
    'low_rank': 29,
    'deep_output_size': 98,
    'deep_hidden_size': 91,
    'deep_dropout': 0.1583,
    'deep_layers': 3,
    'learning_rate': 0.000629,
    'weight_decay': 5.68e-12,
    'dropout_prob': 0.5,
    'activation': 'relu',
    'optimizer': 'adam',
    'lr_scheduler': False,
    'scheduler_type': 'cosine',
    'num_epochs': 20,
    'hidden_size': 128,
    'use_batchnorm': True,
    # FuxiCTR 특화 파라미터
    'use_low_rank_mixture': True,
    'num_experts': 4,
    'model_structure': 'parallel'
}

print("\n=== FuxiCTR DCNv2 모델 테스트 (최고 성능 하이퍼파라미터 적용) ===")
print("하이퍼파라미터:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# 예측기 생성 및 훈련
predictor = TabularPredictor(
    label="Class", 
    problem_type="binary", 
    eval_metric="f1", 
    path="models/dcnv2_fuxictr_best_params_test"
)

print("\n=== 모델 훈련 시작 ===")
predictor.fit(
    train_data,
    hyperparameters={
        "DCNV2_FUXICTR": best_params
    },
    time_limit=600,  # 10분 제한
    verbosity=3
)

# 성능 평가
print("\n=== 성능 평가 ===")
val_pred = predictor.predict(val_data)
test_pred = predictor.predict(test_data)

# 데이터 타입 변환
val_true = val_data['Class'].astype(int)
test_true = test_data['Class'].astype(int)
val_pred = val_pred.astype(int)
test_pred = test_pred.astype(int)

print("검증 성능:")
print(f"  F1 Score: {f1_score(val_true, val_pred):.4f}")
print(f"  Accuracy: {accuracy_score(val_true, val_pred):.4f}")

print("테스트 성능:")
print(f"  F1 Score: {f1_score(test_true, test_pred):.4f}")
print(f"  Accuracy: {accuracy_score(test_true, test_pred):.4f}")

print("\n=== 분류 리포트 ===")
print("검증 데이터:")
print(classification_report(val_true, val_pred))

print("테스트 데이터:")
print(classification_report(test_true, test_pred))

# 리더보드 확인
print("\n=== 리더보드 ===")
print(predictor.leaderboard()) 