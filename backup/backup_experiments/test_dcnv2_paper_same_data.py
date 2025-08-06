# 기존 HPO와 동일한 데이터셋으로 논문 DCNv2 테스트
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_dcnv2_paper_torch_model import TabularDCNv2PaperTorchModel

# 기존 방식으로 모델 등록
ag_model_registry.add(TabularDCNv2PaperTorchModel)

print("=== 기존 HPO와 동일한 데이터셋으로 논문 DCNv2 테스트 ===")

# 기존 HPO에서 사용한 동일한 데이터셋 로드
df = pd.read_csv("datasets/creditcard.csv")
df["Class"] = df["Class"].astype("category")  # AutoGluon에서 분류로 인식하게

print(f"데이터셋 크기: {df.shape}")
print(f"클래스 분포: {df['Class'].value_counts()}")
print(f"클래스 비율: {df['Class'].value_counts(normalize=True)}")

# 논문과 동일한 DCNv2 모델 학습 (기존 HPO와 동일한 설정)
print("\n=== 논문과 동일한 DCNv2 모델 학습 (기존 HPO와 동일한 설정) ===")
predictor = TabularPredictor(
    label="Class", 
    problem_type="binary", 
    eval_metric="f1", 
    path="models/dcnv2_paper_same_data"
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
    time_limit=1800,  # 30분 (기존 HPO와 동일)
    presets="best_quality"
)

# 성능 확인
print("\n=== 성능 확인 ===")
print(predictor.leaderboard())

print("\n=== 기존 HPO 결과와 비교 ===")
print("기존 DCNV2 (20250730_083658 폴더):")
print("- 최고 성능: 82.35%")
print("- 평균 성능: 52.22%")
print("- 실험 횟수: 20번")

print("\n논문 DCNV2_PAPER (현재 테스트):")
leaderboard = predictor.leaderboard()
best_score = leaderboard['score_val'].max()
print(f"- 최고 성능: {best_score:.2%}")
print(f"- 모델 수: {len(leaderboard)}")

print("\n=== 논문 DCNv2 테스트 완료! ===") 