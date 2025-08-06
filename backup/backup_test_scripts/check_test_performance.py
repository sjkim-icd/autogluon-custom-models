import pandas as pd
from autogluon.tabular import TabularPredictor

# 모델 로드
predictor = TabularPredictor.load('models/five_models_experiment')

# 테스트 데이터 로드
test_data = pd.read_csv('datasets/creditcard.csv')

# 테스트 데이터 기준 리더보드
print('=== 테스트 데이터 기준 성능 순위 ===')
leaderboard = predictor.leaderboard(data=test_data)

# 성능 순위로 정렬
sorted_leaderboard = leaderboard[['model', 'score_val']].sort_values('score_val', ascending=False)

print("\n모델별 테스트 성능 (F1 점수 기준):")
for idx, row in sorted_leaderboard.iterrows():
    if 'WeightedEnsemble' not in row['model']:  # 앙상블 제외
        print(f"{row['model']}: {row['score_val']:.4f}")

print("\n=== 전체 리더보드 ===")
print(leaderboard[['model', 'score_val']]) 