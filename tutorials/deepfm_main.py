import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from my_models.tabular_deepfm_torch_model import TabularDeepFMTorchModel
from sklearn.model_selection import train_test_split

# 모델 등록
ag_model_registry.add(TabularDeepFMTorchModel)

# 데이터 로드
df = pd.read_csv("C:/Users/woori/Downloads/archive/creditcard.csv")
df["Class"] = df["Class"].astype("category")  # AutoGluon에서 분류로 인식하게

# 학습/검증 분리
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)

print(f"학습 데이터 크기: {train_data.shape}")
print(f"테스트 데이터 크기: {test_data.shape}")
print(f"학습 데이터 클래스 분포:\n{train_data['Class'].value_counts()}")
print(f"학습 데이터 클래스 비율:\n{train_data['Class'].value_counts(normalize=True)}")

print("\n=== DeepFM 모델 학습 ===")

# DeepFM 모델만 학습
predictor = TabularPredictor(label="Class", problem_type="binary", eval_metric="f1").fit(
    train_data,
    hyperparameters={
        # DeepFM - 1개 설정
        "DEEPFM": {
            "fm_dropout": 0.1,
            "fm_embedding_dim": 10,
            "deep_output_size": 64,
            "deep_hidden_size": 64,
            "deep_dropout": 0.1,
            "deep_layers": 2,
        },
    },
    time_limit=300,  # 5분
    verbosity=3,  # 더 자세한 로그
)

print("\n=== 학습 완료! 결과 확인 ===")
print("리더보드:")
print(predictor.leaderboard())

print("\n=== 모델별 상세 정보 ===")
leaderboard = predictor.leaderboard()
for idx, row in leaderboard.iterrows():
    model_name = row['model']
    score = row['score_val']
    fit_time = row['fit_time_marginal']
    print(f"\n{model_name}:")
    print(f"  - 성능 점수: {score:.4f}")
    print(f"  - 학습 시간: {fit_time:.2f}초")

print("\n=== 최고 성능 모델 ===")
# get_model_best() 대신 leaderboard에서 최고 성능 모델 찾기
best_row = leaderboard.loc[leaderboard['score_val'].idxmax()]
best_model = best_row['model']
print(f"최고 성능 모델: {best_model}")
print(f"성능 점수: {best_row['score_val']:.4f}")

print("\n=== 예측 테스트 ===")
# 테스트 데이터로 예측
predictions = predictor.predict(test_data)
probabilities = predictor.predict_proba(test_data)

print(f"예측 결과 형태: {predictions.shape}")
print(f"확률 예측 형태: {probabilities.shape}")
print(f"예측값 샘플: {predictions.head()}")
print(f"확률값 샘플:\n{probabilities.head()}")

print("\n=== DeepFM 학습 완료! ===") 