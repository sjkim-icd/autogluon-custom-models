# FuxiCTR 스타일 DCNv2 구현 테스트 (데이터 스플릿 포함)
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_dcnv2_fuxictr_torch_model import TabularDCNv2FuxiCTRTorchModel
from autogluon.common import space
from sklearn.model_selection import train_test_split

# 모델 등록
ag_model_registry.add(TabularDCNv2FuxiCTRTorchModel)

print("=== FuxiCTR 스타일 DCNv2 구현 테스트 (데이터 스플릿 포함) ===")

# 기존 HPO에서 사용한 동일한 데이터셋 로드
df = pd.read_csv("datasets/creditcard.csv")
df["Class"] = df["Class"].astype("category")  # AutoGluon에서 분류로 인식하게

print(f"전체 데이터셋 크기: {df.shape}")
print(f"클래스 분포: {df['Class'].value_counts()}")
print(f"클래스 비율: {df['Class'].value_counts(normalize=True)}")

# 데이터 스플릿
print("\n=== 데이터 스플릿 ===")
# 먼저 train/test로 나누기
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['Class']
)

# train을 다시 train/validation으로 나누기
train_df, val_df = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df['Class']
)

print(f"Train 데이터 크기: {train_df.shape}")
print(f"Validation 데이터 크기: {val_df.shape}")
print(f"Test 데이터 크기: {test_df.shape}")

print(f"Train 클래스 분포: {train_df['Class'].value_counts()}")
print(f"Validation 클래스 분포: {val_df['Class'].value_counts()}")
print(f"Test 클래스 분포: {test_df['Class'].value_counts()}")

# FuxiCTR 스타일 DCNv2 모델에 대한 search space 정의
print("\n=== FuxiCTR 스타일 DCNv2 Search Space 설정 ===")
search_space = {
    "DCNV2_FUXICTR": {
        # Cross Network 파라미터
        "num_cross_layers": space.Int(1, 4),  # 1-4층
        "cross_dropout": space.Real(0.0, 0.3),  # 0-30%
        "low_rank": space.Int(16, 64),  # 16-64
        "use_low_rank_mixture": space.Categorical(False, True),  # MoE 사용 여부
        "num_experts": space.Int(2, 6),  # 2-6개 전문가
        "model_structure": space.Categorical("parallel", "stacked", "crossnet_only"),  # 구조 옵션
        
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
for param, value in search_space["DCNV2_FUXICTR"].items():
    print(f"  {param}: {value}")

# FuxiCTR 스타일 DCNv2 모델 HPO 학습
print("\n=== FuxiCTR 스타일 DCNv2 모델 HPO 학습 ===")
predictor = TabularPredictor(
    label="Class", 
    problem_type="binary", 
    eval_metric="f1", 
    path="models/dcnv2_fuxictr_split"
).fit(
    train_df,  # Train 데이터만 사용
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
print("Train/Validation 성능:")
print(predictor.leaderboard())

# Test 데이터로 최종 성능 평가
print("\n=== Test 데이터로 최종 성능 평가 ===")
test_predictions = predictor.predict(test_df)
test_proba = predictor.predict_proba(test_df)

from sklearn.metrics import f1_score, accuracy_score, classification_report
test_f1 = f1_score(test_df['Class'], test_predictions)
test_accuracy = accuracy_score(test_df['Class'], test_predictions)

print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nTest Classification Report:")
print(classification_report(test_df['Class'], test_predictions))

print("\n=== 기존 결과와 비교 ===")
print("기존 DCNV2 (20250730_083658 폴더):")
print("- 최고 성능: 82.35%")
print("- 평균 성능: 52.22%")
print("- 실험 횟수: 20번")

print("\n논문 DCNV2_PAPER HPO (이전 테스트):")
print("- 최고 성능: 81.43%")
print("- 평균 성능: 63.29%")
print("- 실험 횟수: 5")

print("\nFuxiCTR 스타일 DCNV2_FUXICTR HPO (현재 테스트):")
leaderboard = predictor.leaderboard()
if len(leaderboard) > 0:
    best_score = leaderboard['score_val'].max()
    avg_score = leaderboard['score_val'].mean()
    print(f"- Validation 최고 성능: {best_score:.2%}")
    print(f"- Validation 평균 성능: {avg_score:.2%}")
    print(f"- Test F1 Score: {test_f1:.2%}")
    print(f"- 실험 횟수: {len(leaderboard)}")
else:
    print("- 모델 학습 실패")

print("\n=== FuxiCTR 스타일 DCNv2 테스트 완료! ===") 