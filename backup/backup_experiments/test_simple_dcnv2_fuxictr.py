# 간단한 FuxiCTR 스타일 DCNv2 테스트 (HPO 없이)
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_dcnv2_fuxictr_torch_model import TabularDCNv2FuxiCTRTorchModel
from sklearn.model_selection import train_test_split

# 모델 등록
ag_model_registry.add(TabularDCNv2FuxiCTRTorchModel)

print("=== 간단한 FuxiCTR 스타일 DCNv2 테스트 ===")

# 데이터 로드
df = pd.read_csv("datasets/creditcard.csv")
df["Class"] = df["Class"].astype("category")

print(f"데이터셋 크기: {df.shape}")
print(f"클래스 분포: {df['Class'].value_counts()}")

# 데이터 스플릿
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Class'])
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Class'])

print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

# 단일 모델 학습 (HPO 없이)
print("\n=== 단일 FuxiCTR 스타일 DCNv2 모델 학습 ===")
predictor = TabularPredictor(
    label="Class", 
    problem_type="binary", 
    eval_metric="f1", 
    path="models/dcnv2_fuxictr_simple"
).fit(
    train_df,
    hyperparameters={
        "DCNV2_FUXICTR": {
            "num_cross_layers": 2,
            "cross_dropout": 0.1,
            "low_rank": 32,
            "use_low_rank_mixture": False,
            "num_experts": 4,
            "model_structure": "parallel",
            "deep_layers": 3,
            "deep_hidden_size": 128,
            "deep_dropout": 0.1,
            "learning_rate": 0.001,
            "num_epochs": 10,
            "epochs_wo_improve": 5,
        }
    },
    time_limit=600,  # 10분
    presets="best_quality",
    raise_on_no_models_fitted=False
)

# 결과 확인
print("\n=== 결과 확인 ===")
print("Leaderboard:")
print(predictor.leaderboard())

# Test 성능 평가
try:
    test_predictions = predictor.predict(test_df)
    from sklearn.metrics import f1_score, accuracy_score
    test_f1 = f1_score(test_df['Class'], test_predictions)
    test_accuracy = accuracy_score(test_df['Class'], test_predictions)
    
    print(f"\nTest 성능:")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
except Exception as e:
    print(f"Test 예측 실패: {e}")

print("\n=== 간단한 테스트 완료! ===") 