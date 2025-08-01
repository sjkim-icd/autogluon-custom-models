import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_deepfm_torch_model import TabularDeepFMTorchModel
from sklearn.model_selection import train_test_split

# 모델 등록
ag_model_registry.add(TabularDeepFMTorchModel)

# 간단한 테스트 데이터 생성 (범주형 특성 포함)
np.random.seed(42)
n_samples = 1000

# 범주형 특성들
data = {
    'category1': np.random.randint(0, 5, n_samples),
    'category2': np.random.randint(0, 3, n_samples),
    'category3': np.random.randint(0, 4, n_samples),
    'continuous1': np.random.normal(0, 1, n_samples),
    'continuous2': np.random.normal(0, 1, n_samples),
}

df = pd.DataFrame(data)

# 타겟 생성 (범주형 특성들의 조합으로)
df['target'] = ((df['category1'] + df['category2'] + df['category3']) % 2).astype(int)
df['target'] = df['target'].astype('category')

print("테스트 데이터 정보:")
print(f"데이터 크기: {df.shape}")
print(f"특성 타입:\n{df.dtypes}")
print(f"타겟 분포:\n{df['target'].value_counts()}")

# 학습/검증 분리
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)

print(f"\n학습 데이터 크기: {train_data.shape}")
print(f"테스트 데이터 크기: {test_data.shape}")

print("\n=== DeepFM 테스트 ===")

# DeepFM 모델 학습
predictor = TabularPredictor(label="target", problem_type="binary", eval_metric="f1", path="models/deepfm_test").fit(
    train_data,
    hyperparameters={
        "DEEPFM": {
            "fm_dropout": 0.1,
            "fm_embedding_dim": 8,
            "deep_output_size": 64,
            "deep_hidden_size": 64,
            "deep_dropout": 0.1,
            "deep_layers": 2,
            'epochs_wo_improve': 10,
            'num_epochs': 15,
            "lr_scheduler": True,
            "scheduler_type": "cosine",
            "lr_scheduler_min_lr": 1e-6,
        },
    },
    time_limit=300,  # 5분
    verbosity=4,
)

print("\n=== 학습 완료! 결과 확인 ===")
print("리더보드:")
print(predictor.leaderboard())

print("\n=== 예측 테스트 ===")
predictions = predictor.predict(test_data)
probabilities = predictor.predict_proba(test_data)

print(f"예측 결과 형태: {predictions.shape}")
print(f"확률 예측 형태: {probabilities.shape}")
print(f"예측값 샘플: {predictions.head()}")
print(f"확률값 샘플:\n{probabilities.head()}")

print("\n=== DeepFM 테스트 완료! ===") 