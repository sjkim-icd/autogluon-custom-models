import pandas as pd

# 데이터 로드
data = pd.read_csv('datasets/creditcard.csv')

print('=== 데이터 분포 분석 ===')
print(f'전체 데이터 크기: {len(data)}')
print(f'검증 데이터 크기 (2.19%): {int(len(data) * 0.0219)}')
print(f'훈련 데이터 크기: {len(data) - int(len(data) * 0.0219)}')

print('\n=== Class 분포 ===')
print(data['Class'].value_counts())
print('\n=== Class 비율 ===')
print(data['Class'].value_counts(normalize=True))

# 검증 데이터 크기에서의 예상 Class 분포
val_size = int(len(data) * 0.0219)
print(f'\n=== 검증 데이터 예상 분포 (크기: {val_size}) ===')
print(f'Class 0 예상 개수: {int(val_size * 0.998)}')
print(f'Class 1 예상 개수: {int(val_size * 0.002)}') 