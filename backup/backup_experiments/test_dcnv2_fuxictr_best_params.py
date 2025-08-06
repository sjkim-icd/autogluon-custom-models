import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from autogluon.tabular import TabularPredictor
import autogluon.core as ag
from autogluon.common import space

# 데이터 로드
print("=== 데이터 로드 ===")
data = pd.read_csv('datasets/creditcard.csv')
print(f"데이터 크기: {data.shape}")
print(f"클래스 분포:\n{data['Class'].value_counts()}")

# 데이터 분할
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"훈련: {X_train.shape}, 검증: {X_val.shape}, 테스트: {X_test.shape}")

# 0.8235 성능을 달성한 하이퍼파라미터
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
    'use_batchnorm': True
}

print("\n=== FuxiCTR DCNv2 모델 테스트 (최고 성능 하이퍼파라미터 적용) ===")

# FuxiCTR DCNv2 모델 import
import sys
sys.path.append('custom_models')
from tabular_dcnv2_fuxictr_torch_model_fixed import TabularDCNv2FuxiCTRTorchModelFixed

# AutoGluon 모델 등록 (기존 방식 사용)
from autogluon.tabular.registry import ag_model_registry
ag_model_registry.add(TabularDCNv2FuxiCTRTorchModelFixed)

# 하이퍼파라미터 설정 (FuxiCTR 스타일에 맞게 조정)
hyperparameters = {
    'DCNV2_FUXICTR': {
        'num_cross_layers': best_params['num_cross_layers'],
        'cross_dropout': best_params['cross_dropout'],
        'low_rank': best_params['low_rank'],
        'deep_output_size': best_params['deep_output_size'],
        'deep_hidden_size': best_params['deep_hidden_size'],
        'deep_dropout': best_params['deep_dropout'],
        'deep_layers': best_params['deep_layers'],
        'learning_rate': best_params['learning_rate'],
        'weight_decay': best_params['weight_decay'],
        'dropout_prob': best_params['dropout_prob'],
        'activation': best_params['activation'],
        'optimizer': best_params['optimizer'],
        'lr_scheduler': best_params['lr_scheduler'],
        'scheduler_type': best_params['scheduler_type'],
        'num_epochs': best_params['num_epochs'],
        'hidden_size': best_params['hidden_size'],
        'use_batchnorm': best_params['use_batchnorm'],
        # FuxiCTR 특화 파라미터
        'use_low_rank_mixture': True,
        'num_experts': 4,
        'model_structure': 'parallel'
    }
}

print("하이퍼파라미터:")
for key, value in best_params.items():
    print(f"  {key}: {value}")
print("  use_low_rank_mixture: True")
print("  num_experts: 4")
print("  model_structure: parallel")

# 예측기 생성 및 훈련
predictor = TabularPredictor(
    label='Class',
    eval_metric='f1',
    verbosity=3
)

print("\n=== 모델 훈련 시작 ===")
predictor.fit(
    train_data=pd.concat([X_train, y_train], axis=1),
    hyperparameters=hyperparameters,
    time_limit=600,  # 10분 제한
    verbosity=3
)

# 성능 평가
print("\n=== 성능 평가 ===")
val_pred = predictor.predict(X_val)
test_pred = predictor.predict(X_test)

print("검증 성능:")
print(f"  F1 Score: {f1_score(y_val, val_pred):.4f}")
print(f"  Accuracy: {accuracy_score(y_val, val_pred):.4f}")

print("테스트 성능:")
print(f"  F1 Score: {f1_score(y_test, test_pred):.4f}")
print(f"  Accuracy: {accuracy_score(y_test, test_pred):.4f}")

print("\n=== 분류 리포트 ===")
print("검증 데이터:")
print(classification_report(y_val, val_pred))

print("테스트 데이터:")
print(classification_report(y_test, test_pred))

# 리더보드 확인
print("\n=== 리더보드 ===")
print(predictor.leaderboard()) 