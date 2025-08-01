from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np

# 모델 로드
predictor = TabularPredictor.load('AutogluonModels/ag-20250730_083658')
df = predictor.leaderboard()

print("=== DCN과 RandomForest 성능 분석 ===")
print()

# DCN 분석
dcn_df = df[df['model'].str.startswith('DCNV2')]
print("DCNV2 분석:")
print(f"총 실험 수: {len(dcn_df)}")
print(f"최고 성능: {dcn_df['score_val'].max():.6f}")
print(f"최저 성능: {dcn_df['score_val'].min():.6f}")
print(f"평균 성능: {dcn_df['score_val'].mean():.6f}")
print(f"성능 분산: {dcn_df['score_val'].var():.6f}")
print(f"0점 성능 개수: {(dcn_df['score_val'] == 0).sum()}")
print()

# RandomForest 분석
rf_df = df[df['model'].str.startswith('RandomForest')]
print("RandomForest 분석:")
print(f"총 실험 수: {len(rf_df)}")
print(f"최고 성능: {rf_df['score_val'].max():.6f}")
print(f"최저 성능: {rf_df['score_val'].min():.6f}")
print(f"평균 성능: {rf_df['score_val'].mean():.6f}")
print(f"성능 분산: {rf_df['score_val'].var():.6f}")
print(f"0점 성능 개수: {(rf_df['score_val'] == 0).sum()}")
print()

# 성능 분포 비교
print("=== 성능 분포 비교 ===")
print("DCNV2 성능 분포:")
print(dcn_df['score_val'].value_counts().sort_index())
print()
print("RandomForest 성능 분포:")
print(rf_df['score_val'].value_counts().sort_index())
print()

# 상위 성능 모델들과 비교
print("=== 상위 성능 모델들과 비교 ===")
custom_focal_df = df[df['model'].str.startswith('CUSTOM_FOCAL_DL')]
custom_nn_df = df[df['model'].str.startswith('CUSTOM_NN_TORCH')]

print("CUSTOM_FOCAL_DL:")
print(f"평균 성능: {custom_focal_df['score_val'].mean():.6f}")
print(f"성능 분산: {custom_focal_df['score_val'].var():.6f}")
print(f"0점 성능 개수: {(custom_focal_df['score_val'] == 0).sum()}")
print()

print("CUSTOM_NN_TORCH:")
print(f"평균 성능: {custom_nn_df['score_val'].mean():.6f}")
print(f"성능 분산: {custom_nn_df['score_val'].var():.6f}")
print(f"0점 성능 개수: {(custom_nn_df['score_val'] == 0).sum()}")
print()

# 데이터셋 특성 추론
print("=== 가능한 원인 분석 ===")
print("1. DCNV2가 낮은 성능을 보이는 이유:")
print("   - 딥러닝 모델이므로 하이퍼파라미터에 매우 민감")
print("   - 학습률, 배치 크기, 레이어 수 등이 잘못 설정되면 성능이 급격히 떨어짐")
print("   - 0점 성능이 많다는 것은 모델이 전혀 학습되지 않았음을 의미")
print()

print("2. RandomForest가 상대적으로 낮은 성능을 보이는 이유:")
print("   - 트리 기반 모델은 복잡한 패턴을 학습하는데 한계가 있음")
print("   - 데이터셋이 복잡한 비선형 관계를 가지고 있을 가능성")
print("   - 하지만 안정적인 성능을 보임 (분산이 낮음)")
print()

print("3. CUSTOM 모델들이 높은 성능을 보이는 이유:")
print("   - 데이터셋에 특화된 아키텍처 설계")
print("   - Focal Loss나 특별한 손실 함수 사용")
print("   - 더 많은 하이퍼파라미터 튜닝 가능") 