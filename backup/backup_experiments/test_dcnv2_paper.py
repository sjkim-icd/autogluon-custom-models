# 논문과 동일한 DCNv2 테스트
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_dcnv2_paper_torch_model import TabularDCNv2PaperTorchModel

# 기존 방식으로 모델 등록
ag_model_registry.add(TabularDCNv2PaperTorchModel)

print("=== 논문과 동일한 DCNv2 테스트 ===")

# 샘플 데이터 생성
np.random.seed(42)
n_samples = 1000
n_features = 10

# 특성 생성
X = np.random.randn(n_samples, n_features)
# 타겟 생성 (이진 분류) - 특성 교차를 포함한 복잡한 패턴
y = (X[:, 0] + X[:, 1] * X[:, 2] + X[:, 3] * X[:, 4] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

# 데이터프레임 생성
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
df['Class'] = y

print(f"데이터셋 크기: {df.shape}")
print(f"클래스 분포: {df['Class'].value_counts()}")

# 논문과 동일한 DCNv2 모델 학습
print("\n=== 논문과 동일한 DCNv2 모델 학습 ===")
predictor = TabularPredictor(
    label="Class", 
    problem_type="binary", 
    eval_metric="f1", 
    path="models/dcnv2_paper_test"
).fit(
    df,
    hyperparameters={
        "DCNV2_PAPER": {
            "num_cross_layers": 2,
            "low_rank": 32,
            "cross_dropout": 0.1,
            "deep_layers": 3,
            "deep_hidden_size": 128,
            "deep_dropout": 0.1,
            "learning_rate": 0.001,
            "num_epochs": 20,
            "epochs_wo_improve": 5,
        }
    },
    time_limit=300,  # 5분 제한
    presets="best_quality"
)

# 성능 확인
print("\n=== 성능 확인 ===")
print(predictor.leaderboard())

print("\n=== 논문과 동일한 DCNv2 테스트 완료! ===") 