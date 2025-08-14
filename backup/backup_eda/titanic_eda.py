import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# 타이타닉 데이터 로드
print("타이타닉 데이터 로딩 중...")
titanic = fetch_openml(name='titanic', version=1, as_frame=True)
df = titanic.frame

print(f"데이터 형태: {df.shape}")
print(f"컬럼: {df.columns.tolist()}")
print("\n" + "="*50)
print("1. YDATA-PROFILING 적용")
print("="*50)

# 1. YDATA-PROFILING 적용
import ydata_profiling as yp

print("ydata-profiling으로 상세 분석 리포트 생성 중...")
profile = yp.ProfileReport(df, title="Titanic Dataset Analysis")
profile.to_file("titanic_ydata_profiling.html")
print("✅ HTML 리포트가 'titanic_ydata_profiling.html'로 저장되었습니다!")

print("\n" + "="*50)
print("2. AUTOVIZ 적용")
print("="*50)

# 2. AUTOVIZ 적용
from autoviz.AutoViz_Class import AutoViz_Class

print("AutoViz로 자동 시각화 생성 중...")
AV = AutoViz_Class()

# AutoViz 실행 (depVar 변수는 'survived'로 설정)
df_viz = AV.AutoViz(
    filename="",  # 파일명이 없으면 데이터프레임 사용
    dfte=df,     # 데이터프레임
    depVar='survived',  # 타겟 변수 (depVar 사용)
    max_rows_analyzed=1000,  # 분석할 최대 행 수
    max_cols_analyzed=20,    # 분석할 최대 컬럼 수
    verbose=1,               # 상세 출력
    save_plot_dir='autoviz_plots'  # 플롯 저장 디렉토리
)

print("✅ AutoViz 시각화가 'autoviz_plots' 폴더에 저장되었습니다!")

print("\n" + "="*50)
print("3. 주요 발견사항 요약")
print("="*50)

# 기본 통계 정보 출력
print("📊 기본 통계:")
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

print("\n" + "="*50)
print("📁 생성된 파일들:")
print("- titanic_ydata_profiling.html: 상세 분석 리포트")
print("- autoviz_plots/: 자동 생성된 시각화 파일들")
print("="*50) 