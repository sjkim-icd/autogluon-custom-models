import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
print("🔍 4개 EDA 패키지로 타이타닉 데이터 분석 시작!")
print("="*60)

# ============================================================================
# 1. YDATA-PROFILING 적용
# ============================================================================
print("\n" + "="*60)
print("1️⃣ YDATA-PROFILING 적용")
print("="*60)

import ydata_profiling as yp

print("📈 ydata-profiling으로 상세 분석 리포트 생성 중...")
profile = yp.ProfileReport(df, title="Titanic Dataset Analysis")
profile.to_file("titanic_ydata_profiling.html")
print("✅ HTML 리포트가 'titanic_ydata_profiling.html'로 저장되었습니다!")

# ============================================================================
# 2. AUTOVIZ 적용
# ============================================================================
print("\n" + "="*60)
print("2️⃣ AUTOVIZ 적용")
print("="*60)

from autoviz.AutoViz_Class import AutoViz_Class

print("🎨 AutoViz로 자동 시각화 생성 중...")
AV = AutoViz_Class()

# AutoViz 실행
df_viz = AV.AutoViz(
    filename="",  # 파일명이 없으면 데이터프레임 사용
    dfte=df,     # 데이터프레임
    depVar='survived',  # 타겟 변수
    max_rows_analyzed=1000,  # 분석할 최대 행 수
    max_cols_analyzed=20,    # 분석할 최대 컬럼 수
    verbose=1,               # 상세 출력
    save_plot_dir='autoviz_plots'  # 플롯 저장 디렉토리
)

print("✅ AutoViz 시각화가 'autoviz_plots' 폴더에 저장되었습니다!")

# ============================================================================
# 3. KLIB 적용
# ============================================================================
print("\n" + "="*60)
print("3️⃣ KLIB 적용")
print("="*60)

import klib

print("🔧 Klib로 데이터 클리닝 및 분석 중...")

# 데이터 정보 요약
print("📋 데이터 정보:")
klib.describe(df)

# 결측치 분석
print("\n🔍 결측치 분석:")
klib.missing_values(df)

# 상관관계 분석
print("\n📊 상관관계 분석:")
# 수치형 컬럼만 선택
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    klib.corr_mat(df[numeric_cols])

# 분포 분석
print("\n📈 분포 분석:")
for col in numeric_cols:
    if col != 'survived':  # survived는 제외
        klib.dist_plot(df[col])

print("✅ Klib 분석이 완료되었습니다!")

# ============================================================================
# 4. D-TALE 적용
# ============================================================================
print("\n" + "="*60)
print("4️⃣ D-TALE 적용")
print("="*60)

import dtale

print("🌐 D-Tale 대화형 인터페이스 시작 중...")

# D-Tale 인스턴스 생성
d = dtale.show(df, name="Titanic Dataset")

print("✅ D-Tale이 시작되었습니다!")
print(f"🌐 브라우저에서 다음 URL로 접속하세요: {d._url}")
print("💡 브라우저가 자동으로 열리지 않으면 위 URL을 복사해서 접속하세요!")

# ============================================================================
# 5. 주요 발견사항 요약
# ============================================================================
print("\n" + "="*60)
print("📊 주요 발견사항 요약")
print("="*60)

# 기본 통계 정보 출력
print("📈 기본 통계:")
print(f"- 총 승객 수: {len(df)}")

# survived 컬럼을 숫자로 변환
survived_numeric = pd.to_numeric(df['survived'], errors='coerce')
print(f"- 생존률: {survived_numeric.mean():.2%}")
print(f"- 결측치가 있는 컬럼: {df.columns[df.isnull().any()].tolist()}")

# 성별 생존률
if 'sex' in df.columns:
    survival_by_sex = df.groupby('sex')['survived'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
    print(f"\n👥 성별 생존률:")
    print(survival_by_sex)

# 클래스별 생존률
if 'pclass' in df.columns:
    survival_by_class = df.groupby('pclass')['survived'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
    print(f"\n🎫 클래스별 생존률:")
    print(survival_by_class)

# 나이별 생존률 (나이가 있는 경우)
if 'age' in df.columns:
    df['age_group'] = pd.cut(pd.to_numeric(df['age'], errors='coerce'), 
                             bins=[0, 18, 30, 50, 100], 
                             labels=['Child', 'Young', 'Adult', 'Senior'])
    age_survival = df.groupby('age_group')['survived'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
    print(f"\n👶 나이대별 생존률:")
    print(age_survival)

print("\n" + "="*60)
print("📁 생성된 파일들:")
print("- titanic_ydata_profiling.html: 상세 분석 리포트")
print("- autoviz_plots/: 자동 생성된 시각화 파일들")
print("- D-Tale 웹 인터페이스: 브라우저에서 확인 가능")
print("="*60)

print("\n🎉 모든 EDA 패키지 분석이 완료되었습니다!")
print("💡 각 패키지의 특징:")
print("   • ydata-profiling: 포괄적인 데이터 품질 분석")
print("   • AutoViz: 자동 시각화 및 패턴 발견")
print("   • Klib: 데이터 클리닝 및 간단한 분석")
print("   • D-Tale: 대화형 데이터 탐색 인터페이스") 