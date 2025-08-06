import pandas as pd
import numpy as np

# HPO 결과 데이터 (실행 결과에서 추출) - 정확한 길이
models = [
    'DCNV2\\c30a7_00010', 'DCNV2\\c30a7_00006', 'DCNV2\\c30a7_00011', 
    'CUSTOM_NN_TORCH\\dd5e4_00006', 'CUSTOM_FOCAL_DL\\7b840_00000', 
    'DCNV2_FUXICTR\\19640_00013', 'CUSTOM_NN_TORCH\\dd5e4_00003', 
    'CUSTOM_NN_TORCH\\dd5e4_00002', 'RandomForest\\T3', 'RandomForest\\T10',
    'RandomForest\\T8', 'RandomForest\\T2', 'RandomForest\\T4', 'RandomForest\\T9',
    'RandomForest\\T7', 'DCNV2_FUXICTR\\19640_00006', 'DCNV2\\c30a7_00012',
    'CUSTOM_NN_TORCH\\dd5e4_00000', 'DCNV2\\c30a7_00007', 'DCNV2_FUXICTR\\19640_00004',
    'CUSTOM_NN_TORCH\\dd5e4_00004', 'DCNV2_FUXICTR\\19640_00011', 'RandomForest\\T5',
    'RandomForest\\T1', 'RandomForest\\T6', 'DCNV2_FUXICTR\\19640_00012',
    'DCNV2_FUXICTR\\19640_00005', 'DCNV2\\c30a7_00000', 'DCNV2\\c30a7_00019',
    'DCNV2\\c30a7_00009', 'DCNV2\\c30a7_00017', 'DCNV2\\c30a7_00018',
    'DCNV2_FUXICTR\\19640_00001', 'DCNV2_FUXICTR\\19640_00000', 'DCNV2_FUXICTR\\19640_00008',
    'DCNV2_FUXICTR\\19640_00009', 'DCNV2_FUXICTR\\19640_00003', 'DCNV2\\c30a7_00016',
    'DCNV2\\c30a7_00014', 'DCNV2\\c30a7_00004', 'DCNV2\\c30a7_00003',
    'DCNV2\\c30a7_00015', 'DCNV2\\c30a7_00013', 'DCNV2\\c30a7_00008',
    'DCNV2\\c30a7_00001', 'DCNV2\\c30a7_00002', 'DCNV2\\c30a7_00005',
    'DCNV2_FUXICTR\\19640_00007', 'DCNV2_FUXICTR\\19640_00010'
]

score_val = [
    0.941176, 0.888889, 0.888889, 0.888889, 0.875000, 0.875000, 0.875000, 0.875000,
    0.859155, 0.853521, 0.851799, 0.849508, 0.849354, 0.846797, 0.840288, 0.823529,
    0.823529, 0.823529, 0.823529, 0.823529, 0.823529, 0.823529, 0.823529, 0.823529,
    0.795287, 0.791789, 0.777778, 0.750000, 0.705882, 0.705882, 0.705882, 0.631579,
    0.625000, 0.615385, 0.571429, 0.533333, 0.533333, 0.500000, 0.461538, 0.307692,
    0.307692, 0.166667, 0.133333, 0.129032, 0.064516, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000
]

score_test = [
    0.780749, 0.788177, 0.795918, 0.760181, 0.792899, 0.797927, 0.767442, 0.771084,
    0.860335, 0.868132, 0.876404, 0.866667, 0.879121, 0.870968, 0.865169, 0.740741,
    0.757282, 0.797927, 0.802030, 0.717391, 0.775120, 0.788177, 0.820225, 0.811429,
    0.820225, 0.777778, 0.686275, 0.659218, 0.650602, 0.696970, 0.691489, 0.656410,
    0.572973, 0.548387, 0.529412, 0.494382, 0.471698, 0.373626, 0.224138, 0.201342,
    0.201299, 0.178082, 0.114833, 0.075342, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000
]

# DataFrame 생성
df = pd.DataFrame({
    'model': models,
    'score_val': score_val,
    'score_test': score_test
})

print("=== HPO 결과 분석 ===")
print(f"총 모델 수: {len(df)}")
print()

# 1. 검증 데이터 (score_val) 기준 정렬
print("📊 검증 데이터 (score_val) 기준 순위:")
print("=" * 80)
val_sorted = df.sort_values('score_val', ascending=False).reset_index(drop=True)
for idx, row in val_sorted.head(15).iterrows():
    print(f"{idx+1:2d}. {row['model']:<35} | F1 = {row['score_val']:.4f}")
print()

# 2. 테스트 데이터 (score_test) 기준 정렬
print("📊 테스트 데이터 (score_test) 기준 순위:")
print("=" * 80)
test_sorted = df.sort_values('score_test', ascending=False).reset_index(drop=True)
for idx, row in test_sorted.head(15).iterrows():
    print(f"{idx+1:2d}. {row['model']:<35} | F1 = {row['score_test']:.4f}")
print()

# 3. 모델 타입별 분석
print("🔍 모델 타입별 최고 성능:")
print("=" * 80)

# 모델 타입 분류
def get_model_type(model_name):
    if 'DCNV2\\' in model_name and 'FUXICTR' not in model_name:
        return 'DCNV2'
    elif 'DCNV2_FUXICTR' in model_name:
        return 'DCNV2_FUXICTR'
    elif 'CUSTOM_FOCAL_DL' in model_name:
        return 'CUSTOM_FOCAL_DL'
    elif 'CUSTOM_NN_TORCH' in model_name:
        return 'CUSTOM_NN_TORCH'
    elif 'RandomForest' in model_name:
        return 'RandomForest'
    else:
        return 'Unknown'

df['model_type'] = df['model'].apply(get_model_type)

# 각 모델 타입별 최고 성능
for model_type in ['DCNV2', 'DCNV2_FUXICTR', 'CUSTOM_FOCAL_DL', 'CUSTOM_NN_TORCH', 'RandomForest']:
    type_df = df[df['model_type'] == model_type]
    if len(type_df) > 0:
        best_val = type_df.loc[type_df['score_val'].idxmax()]
        best_test = type_df.loc[type_df['score_test'].idxmax()]
        print(f"{model_type:15s} | 검증 최고: {best_val['model']:<25} | F1 = {best_val['score_val']:.4f}")
        print(f"{'':15s} | 테스트 최고: {best_test['model']:<25} | F1 = {best_test['score_test']:.4f}")
        print()

# 4. Focal Loss vs CrossEntropy 비교
print("⚔️ Focal Loss vs CrossEntropy 비교:")
print("=" * 80)
focal_models = df[df['model_type'] == 'CUSTOM_FOCAL_DL']
nn_models = df[df['model_type'] == 'CUSTOM_NN_TORCH']

if len(focal_models) > 0 and len(nn_models) > 0:
    focal_best_val = focal_models['score_val'].max()
    focal_best_test = focal_models['score_test'].max()
    nn_best_val = nn_models['score_val'].max()
    nn_best_test = nn_models['score_test'].max()
    
    print(f"Focal Loss 최고 성능:")
    print(f"  - 검증 데이터: {focal_best_val:.4f}")
    print(f"  - 테스트 데이터: {focal_best_test:.4f}")
    print()
    print(f"CrossEntropy 최고 성능:")
    print(f"  - 검증 데이터: {nn_best_val:.4f}")
    print(f"  - 테스트 데이터: {nn_best_test:.4f}")
    print()
    
    if focal_best_val > nn_best_val:
        print("✅ 검증 데이터에서 Focal Loss가 우수!")
    else:
        print("❌ 검증 데이터에서 CrossEntropy가 우수!")
        
    if focal_best_test > nn_best_test:
        print("✅ 테스트 데이터에서 Focal Loss가 우수!")
    else:
        print("❌ 테스트 데이터에서 CrossEntropy가 우수!")

# 5. 전체 통계
print()
print("📈 전체 통계:")
print("=" * 80)
print(f"검증 데이터 평균 F1: {df['score_val'].mean():.4f}")
print(f"테스트 데이터 평균 F1: {df['score_test'].mean():.4f}")
print(f"검증 데이터 최고 F1: {df['score_val'].max():.4f}")
print(f"테스트 데이터 최고 F1: {df['score_test'].max():.4f}")
print(f"검증 데이터 최저 F1: {df['score_val'].min():.4f}")
print(f"테스트 데이터 최저 F1: {df['score_test'].min():.4f}")

# 6. 성능 차이 분석
print()
print("🔍 검증 vs 테스트 성능 차이:")
print("=" * 80)
df['performance_diff'] = df['score_val'] - df['score_test']
overfitting_models = df[df['performance_diff'] > 0.1].sort_values('performance_diff', ascending=False)
print(f"과적합 의심 모델 (검증-테스트 > 0.1): {len(overfitting_models)}개")
for idx, row in overfitting_models.head(5).iterrows():
    print(f"  {row['model']:<35} | 차이 = {row['performance_diff']:.4f}")

print()
print("🎯 결론:")
print("=" * 80)
best_val_model = df.loc[df['score_val'].idxmax()]
best_test_model = df.loc[df['score_test'].idxmax()]

print(f"검증 데이터 최고 모델: {best_val_model['model']} (F1 = {best_val_model['score_val']:.4f})")
print(f"테스트 데이터 최고 모델: {best_test_model['model']} (F1 = {best_test_model['score_test']:.4f})")

if best_val_model['model'] == best_test_model['model']:
    print("✅ 검증과 테스트에서 동일한 모델이 최고 성능!")
else:
    print("⚠️ 검증과 테스트에서 다른 모델이 최고 성능") 