import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')

# 타이타닉 데이터 로드
print("🚢 타이타닉 데이터 로딩 중...")
titanic = fetch_openml(name='titanic', version=1, as_frame=True)
df = titanic.frame

print(f"📊 데이터 형태: {df.shape}")
print(f"📋 컬럼: {df.columns.tolist()}")

print("\n" + "="*60)
print("🔧 KLIB 상세 분석")
print("="*60)

# ============================================================================
# 1. 데이터 정보 요약
# ============================================================================
print("\n📋 1. 데이터 정보 요약:")
print("="*40)
print(df.info())

# ============================================================================
# 2. 결측치 분석
# ============================================================================
print("\n🔍 2. 결측치 분석:")
print("="*40)
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100

missing_df = pd.DataFrame({
    '결측치 개수': missing_data,
    '결측치 비율(%)': missing_percent
})

print("결측치가 있는 컬럼:")
print(missing_df[missing_df['결측치 개수'] > 0].sort_values('결측치 개수', ascending=False))

# ============================================================================
# 3. 수치형 변수 분석
# ============================================================================
print("\n📊 3. 수치형 변수 분석:")
print("="*40)
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"수치형 변수: {list(numeric_cols)}")

# 기술통계
print("\n📈 기술통계:")
print(df[numeric_cols].describe())

# ============================================================================
# 4. 상관관계 분석
# ============================================================================
print("\n📊 4. 상관관계 분석:")
print("="*40)
if len(numeric_cols) > 1:
    correlation_matrix = df[numeric_cols].corr()
    print("수치형 변수 간 상관관계:")
    print(correlation_matrix.round(3))
    
    # 높은 상관관계 찾기
    print("\n🔍 높은 상관관계 (절댓값 > 0.5):")
    high_corr = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                high_corr.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))
    
    for var1, var2, corr_val in high_corr:
        print(f"  {var1} ↔ {var2}: {corr_val:.3f}")

# ============================================================================
# 5. 범주형 변수 분석
# ============================================================================
print("\n📊 5. 범주형 변수 분석:")
print("="*40)
categorical_cols = df.select_dtypes(include=['category', 'object']).columns
print(f"범주형 변수: {list(categorical_cols)}")

for col in categorical_cols:
    if col != 'name':  # 이름은 너무 많으므로 제외
        print(f"\n📋 {col}:")
        value_counts = df[col].value_counts()
        print(f"  고유값 개수: {len(value_counts)}")
        print(f"  상위 5개 값:")
        for val, count in value_counts.head().items():
            print(f"    {val}: {count}개 ({count/len(df)*100:.1f}%)")

# ============================================================================
# 6. 생존률 분석
# ============================================================================
print("\n📊 6. 생존률 분석:")
print("="*40)

# survived 컬럼을 숫자로 변환
survived_numeric = pd.to_numeric(df['survived'], errors='coerce')

print(f"전체 생존률: {survived_numeric.mean():.2%}")

# 성별 생존률
if 'sex' in df.columns:
    survival_by_sex = df.groupby('sex')['survived'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
    print(f"\n👥 성별 생존률:")
    for sex, rate in survival_by_sex.items():
        print(f"  {sex}: {rate:.2%}")

# 클래스별 생존률
if 'pclass' in df.columns:
    survival_by_class = df.groupby('pclass')['survived'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
    print(f"\n🎫 클래스별 생존률:")
    for pclass, rate in survival_by_class.items():
        print(f"  {pclass}등급: {rate:.2%}")

# 나이별 생존률
if 'age' in df.columns:
    df['age_group'] = pd.cut(pd.to_numeric(df['age'], errors='coerce'), 
                             bins=[0, 18, 30, 50, 100], 
                             labels=['Child', 'Young', 'Adult', 'Senior'])
    age_survival = df.groupby('age_group')['survived'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
    print(f"\n👶 나이대별 생존률:")
    for age_group, rate in age_survival.items():
        print(f"  {age_group}: {rate:.2%}")

print("\n✅ Klib 상세 분석이 완료되었습니다!") 