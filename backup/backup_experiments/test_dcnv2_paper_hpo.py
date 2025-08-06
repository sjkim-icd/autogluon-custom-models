# 논문 DCNv2 모델에 대한 HPO 테스트
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_dcnv2_paper_torch_model import TabularDCNv2PaperTorchModel
from autogluon.common import space

# 모델 등록
ag_model_registry.add(TabularDCNv2PaperTorchModel)

print("=== 논문 DCNv2 모델 HPO 테스트 (10개 실험, 1시간 제한) ===")

# 기존 HPO에서 사용한 동일한 데이터셋 로드
df = pd.read_csv("datasets/creditcard.csv")
df["Class"] = df["Class"].astype("category")  # AutoGluon에서 분류로 인식하게

print(f"데이터셋 크기: {df.shape}")
print(f"클래스 분포: {df['Class'].value_counts()}")
print(f"클래스 비율: {df['Class'].value_counts(normalize=True)}")

# 논문 DCNv2 모델에 대한 search space 정의
print("\n=== 논문 DCNv2 Search Space 설정 ===")
search_space = {
    "DCNV2_PAPER": {
        # Cross Network 파라미터
        "num_cross_layers": space.Int(1, 4),  # 1-4층
        "low_rank": space.Int(16, 64),  # 16-64
        "cross_dropout": space.Real(0.0, 0.3),  # 0-30%
        
        # Deep Network 파라미터
        "deep_layers": space.Int(2, 4),  # 2-4층
        "deep_hidden_size": space.Int(64, 256),  # 64-256
        "deep_dropout": space.Real(0.0, 0.3),  # 0-30%
        
        # 학습 파라미터
        "learning_rate": space.Real(0.0001, 0.01),  # 0.0001-0.01
        "num_epochs": space.Int(10, 30),  # 10-30 에포크
        "epochs_wo_improve": space.Int(3, 8),  # 3-8 에포크
    }
}

print("Search Space:")
for param, value in search_space["DCNV2_PAPER"].items():
    print(f"  {param}: {value}")

# 논문 DCNv2 모델 HPO 학습
print("\n=== 논문 DCNv2 모델 HPO 학습 ===")
predictor = TabularPredictor(
    label="Class", 
    problem_type="binary", 
    eval_metric="f1", 
    path="models/dcnv2_paper_hpo"
).fit(
    df,
    hyperparameters=search_space,
    hyperparameter_tune_kwargs={
        "num_trials": 10,  # 10개 실험
        "time_limit": 3600,  # 1시간 (60분)
        "scheduler": "local",  # scheduler 추가
        "searcher": "random",  # searcher 추가
    },
    time_limit=3600,  # 1시간 (60분)
    presets="best_quality",
    raise_on_no_models_fitted=False  # 모델 실패해도 계속 진행
)

# 성능 확인
print("\n=== 성능 확인 ===")
print(predictor.leaderboard())

print("\n=== 기존 HPO 결과와 비교 ===")
print("기존 DCNV2 (20250730_083658 폴더):")
print("- 최고 성능: 82.35%")
print("- 평균 성능: 52.22%")
print("- 실험 횟수: 20번")

print("\n논문 DCNV2_PAPER HPO (현재 테스트):")
leaderboard = predictor.leaderboard()
if len(leaderboard) > 0:
    best_score = leaderboard['score_val'].max()
    avg_score = leaderboard['score_val'].mean()
    print(f"- 최고 성능: {best_score:.2%}")
    print(f"- 평균 성능: {avg_score:.2%}")
    print(f"- 실험 횟수: {len(leaderboard)}")
else:
    print("- 모델 학습 실패")

print("\n=== 논문 DCNv2 HPO 테스트 완료! ===") 