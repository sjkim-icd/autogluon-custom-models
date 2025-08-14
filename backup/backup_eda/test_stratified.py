import pandas as pd

# 원본 데이터 로드
df = pd.read_csv('datasets/creditcard.csv')
print('=== 원본 데이터 ===')
print('전체 크기:', df.shape)
print('Class 분포:')
print(df['Class'].value_counts())
print('비율:')
print(df['Class'].value_counts(normalize=True))

# 타겟 비율 유지 샘플링 테스트
print('\n=== 타겟 비율 유지 샘플링 테스트 ===')
target_counts = df['Class'].value_counts()
target_ratios = target_counts / len(df)
n_samples = 10000

# 각 클래스별 샘플 수 계산
sample_counts = {}
for class_val, ratio in target_ratios.items():
    sample_counts[class_val] = int(n_samples * ratio)

print('계산된 샘플 수:')
print(sample_counts)
print('총합:', sum(sample_counts.values()))

# 실제 샘플링 실행
sampled_dfs = []
for class_val, n_sample in sample_counts.items():
    if n_sample > 0:
        class_df = df[df['Class'] == class_val]
        if len(class_df) >= n_sample:
            sampled_class = class_df.sample(n=n_sample, random_state=42)
        else:
            sampled_class = class_df
        sampled_dfs.append(sampled_class)

result_df = pd.concat(sampled_dfs, ignore_index=True)

print('\n=== 샘플링 결과 ===')
print('샘플링 크기:', result_df.shape)
print('Class 분포:')
print(result_df['Class'].value_counts())
print('비율:')
print(result_df['Class'].value_counts(normalize=True))

# 단순 랜덤 샘플링과 비교
print('\n=== 단순 랜덤 샘플링 비교 ===')
simple_sample = df.sample(n=10000, random_state=42)
print('단순 랜덤 샘플링 결과:')
print(simple_sample['Class'].value_counts())
print('비율:')
print(simple_sample['Class'].value_counts(normalize=True)) 