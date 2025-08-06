# 기존 HPO 모델별 학습 시간 확인
from autogluon.tabular import TabularPredictor
import pandas as pd

print("=== 기존 HPO 모델별 학습 시간 확인 ===")

# 기존 HPO 결과 로드
predictor = TabularPredictor.load('AutogluonModels/ag-20250730_083658')
df = predictor.leaderboard()

print("모델별 학습 시간:")
print(df[['model', 'fit_time']].head(20))

print("\n=== DCNV2 모델들의 학습 시간 ===")
dcn_df = df[df['model'].str.startswith('DCNV2')]
if len(dcn_df) > 0:
    print("DCNV2 모델들:")
    print(dcn_df[['model', 'fit_time']])
    print(f"\nDCNV2 평균 학습 시간: {dcn_df['fit_time'].mean():.2f}초")
    print(f"DCNV2 최소 학습 시간: {dcn_df['fit_time'].min():.2f}초")
    print(f"DCNV2 최대 학습 시간: {dcn_df['fit_time'].max():.2f}초")
else:
    print("DCNV2 모델이 없습니다.")

print("\n=== 전체 모델 학습 시간 통계 ===")
print(f"전체 평균 학습 시간: {df['fit_time'].mean():.2f}초")
print(f"전체 최소 학습 시간: {df['fit_time'].min():.2f}초")
print(f"전체 최대 학습 시간: {df['fit_time'].max():.2f}초") 