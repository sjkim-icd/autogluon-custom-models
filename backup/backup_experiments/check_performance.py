from autogluon.tabular import TabularPredictor
import pandas as pd

# 모델 로드
predictor = TabularPredictor.load('AutogluonModels/ag-20250730_083658')
df = predictor.leaderboard()

print("=== 20250730_083658 폴더 모델별 최고 성능 ===")
print()

# 모델 타입별로 그룹화하여 최고 성능 확인
for model_type in df['model'].str.split('\\').str[0].unique():
    if model_type != 'WeightedEnsemble_L2':
        model_df = df[df['model'].str.startswith(model_type)]
        max_score = model_df['score_val'].max()
        count = len(model_df)
        mean_score = model_df['score_val'].mean()
        
        print(f"{model_type}:")
        print(f"  최고 성능: {max_score:.6f}")
        print(f"  실험 횟수: {count}")
        print(f"  평균 성능: {mean_score:.6f}")
        print()

print("=== 전체 모델 성능 순위 (상위 10개) ===")
print(df.nlargest(10, 'score_val')[['model', 'score_val']]) 