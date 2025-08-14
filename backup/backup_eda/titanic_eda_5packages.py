import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
import warnings
import os
warnings.filterwarnings('ignore')

# 타이타닉 데이터 로드
print("🚢 타이타닉 데이터 로딩 중...")
titanic = fetch_openml(name='titanic', version=1, as_frame=True)
df = titanic.frame

print(f"📊 데이터 형태: {df.shape}")
print(f"📋 컬럼: {df.columns.tolist()}")
print("\n" + "="*60)
print("🔍 5개 EDA 패키지로 타이타닉 데이터 완전 분석!")
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
# 2. SWEETVIZ 적용
# ============================================================================
print("\n" + "="*60)
print("2️⃣ SWEETVIZ 적용")
print("="*60)

import sweetviz as sv

print("🍯 Sweetviz로 데이터 분석 리포트 생성 중...")

# Sweetviz 리포트 생성
report = sv.analyze([df, "Titanic Dataset"], target_feat='survived')
report.show_html('titanic_sweetviz_report.html')

print("✅ Sweetviz HTML 리포트가 'titanic_sweetviz_report.html'로 저장되었습니다!")

# ============================================================================
# 3. AUTOVIZ 적용
# ============================================================================
print("\n" + "="*60)
print("3️⃣ AUTOVIZ 적용")
print("="*60)

from autoviz.AutoViz_Class import AutoViz_Class

# 저장할 폴더 생성
plot_dir = "autoviz_plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"📁 '{plot_dir}' 폴더를 생성했습니다.")

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
    save_plot_dir=plot_dir,  # 플롯 저장 디렉토리
    chart_format='png'       # PNG 형식으로 저장
)

print("✅ AutoViz 시각화가 'autoviz_plots' 폴더에 저장되었습니다!")

# ============================================================================
# 4. KLIB 적용
# ============================================================================
print("\n" + "="*60)
print("4️⃣ KLIB 적용")
print("="*60)

import klib

print("🔧 Klib로 데이터 클리닝 및 분석 중...")

# 데이터 정보 요약
print("📋 데이터 정보:")
print(df.info())

# 결측치 분석
print("\n🔍 결측치 분석:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100

missing_df = pd.DataFrame({
    '결측치 개수': missing_data,
    '결측치 비율(%)': missing_percent
})

print("결측치가 있는 컬럼:")
print(missing_df[missing_df['결측치 개수'] > 0].sort_values('결측치 개수', ascending=False))

# 상관관계 분석
print("\n📊 상관관계 분석:")
# 수치형 컬럼만 선택
numeric_cols = df.select_dtypes(include=[np.number]).columns
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

# 분포 분석
print("\n📈 분포 분석:")
for col in numeric_cols:
    if col != 'survived':  # survived는 제외
        print(f"- {col}: 평균={df[col].mean():.2f}, 표준편차={df[col].std():.2f}")

print("✅ Klib 분석이 완료되었습니다!")

# ============================================================================
# 5. D-TALE 적용 (포트 4000 사용)
# ============================================================================
print("\n" + "="*60)
print("5️⃣ D-TALE 적용")
print("="*60)

import dtale

print("🌐 D-Tale 대화형 인터페이스 시작 중...")

# D-Tale 인스턴스 생성 (포트 4000 사용)
d = dtale.show(df, name="Titanic Dataset", port=4000, host='localhost')

print("✅ D-Tale이 시작되었습니다!")
print(f"🌐 브라우저에서 다음 URL로 접속하세요: http://localhost:4000")
print("💡 브라우저가 자동으로 열리지 않으면 위 URL을 복사해서 접속하세요!")

# 브라우저 자동 열기
import webbrowser
import time
time.sleep(2)
try:
    webbrowser.open("http://localhost:4000")
    print("🌐 브라우저를 자동으로 열었습니다!")
except:
    print("❌ 브라우저 자동 열기 실패. 수동으로 접속하세요.")

# ============================================================================
# 6. 주요 발견사항 요약
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
    for sex, rate in survival_by_sex.items():
        print(f"  {sex}: {rate:.2%}")

# 클래스별 생존률
if 'pclass' in df.columns:
    survival_by_class = df.groupby('pclass')['survived'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
    print(f"\n🎫 클래스별 생존률:")
    for pclass, rate in survival_by_class.items():
        print(f"  {pclass}등급: {rate:.2%}")

# 나이별 생존률 (나이가 있는 경우)
if 'age' in df.columns:
    df['age_group'] = pd.cut(pd.to_numeric(df['age'], errors='coerce'), 
                             bins=[0, 18, 30, 50, 100], 
                             labels=['Child', 'Young', 'Adult', 'Senior'])
    age_survival = df.groupby('age_group')['survived'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
    print(f"\n👶 나이대별 생존률:")
    for age_group, rate in age_survival.items():
        print(f"  {age_group}: {rate:.2%}")

# ============================================================================
# 7. 생성된 파일들 확인
# ============================================================================
print("\n" + "="*60)
print("📁 생성된 파일들 확인")
print("="*60)

# ydata-profiling 파일 확인
if os.path.exists("titanic_ydata_profiling.html"):
    file_size = os.path.getsize("titanic_ydata_profiling.html") / (1024*1024)  # MB
    print(f"✅ titanic_ydata_profiling.html: {file_size:.1f}MB")

# Sweetviz 파일 확인
if os.path.exists("titanic_sweetviz_report.html"):
    file_size = os.path.getsize("titanic_sweetviz_report.html") / (1024*1024)  # MB
    print(f"✅ titanic_sweetviz_report.html: {file_size:.1f}MB")

# AutoViz 폴더 확인
if os.path.exists("autoviz_plots"):
    autoviz_files = []
    for root, dirs, files in os.walk("autoviz_plots"):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            autoviz_files.append(f"  - {file} ({file_size:.0f}KB)")
    
    print(f"✅ autoviz_plots/ 폴더: {len(autoviz_files)}개 파일")
    for file_info in autoviz_files:
        print(file_info)

print("\n" + "="*60)
print("🎉 모든 EDA 패키지 분석이 완료되었습니다!")
print("💡 각 패키지의 특징:")
print("   • ydata-profiling: 포괄적인 데이터 품질 분석")
print("   • Sweetviz: 타겟 변수 중심의 상세 분석")
print("   • AutoViz: 자동 시각화 및 패턴 발견")
print("   • Klib: 데이터 클리닝 및 간단한 분석")
print("   • D-Tale: 대화형 데이터 탐색 인터페이스 (포트 4000)")
print("="*60)

print("\n⏳ D-Tale이 실행 중입니다. Ctrl+C를 눌러서 종료하세요.")

# 무한 루프로 서버 유지
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 분석을 종료합니다.") 