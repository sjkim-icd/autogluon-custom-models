import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# 타이타닉 데이터 로드
print("🚢 타이타닉 데이터 로딩 중...")
titanic = fetch_openml(name='titanic', version=1, as_frame=True)
df = titanic.frame

print(f"📊 데이터 형태: {df.shape}")
print(f"📋 컬럼: {df.columns.tolist()}")

# ============================================================================
# 폴더 구조 생성
# ============================================================================
print("\n" + "="*60)
print("📁 EDA 결과 폴더 구조 생성")
print("="*60)

# 메인 EDA 폴더 생성
eda_folder = "EDA"
if not os.path.exists(eda_folder):
    os.makedirs(eda_folder)
    print(f"📁 '{eda_folder}' 폴더를 생성했습니다.")

# 타이타닉 데이터 폴더 생성
titanic_folder = os.path.join(eda_folder, "titanic")
if not os.path.exists(titanic_folder):
    os.makedirs(titanic_folder)
    print(f"📁 '{titanic_folder}' 폴더를 생성했습니다.")

# 각 패키지별 하위 폴더 생성
packages = {
    "ydata_profiling": os.path.join(titanic_folder, "ydata_profiling"),
    "sweetviz": os.path.join(titanic_folder, "sweetviz"),
    "autoviz": os.path.join(titanic_folder, "autoviz"),
    "klib": os.path.join(titanic_folder, "klib"),
    "dtale": os.path.join(titanic_folder, "dtale")
}

for package_name, package_path in packages.items():
    if not os.path.exists(package_path):
        os.makedirs(package_path)
        print(f"📁 '{package_path}' 폴더를 생성했습니다.")

print("✅ 폴더 구조 생성 완료!")

# ============================================================================
# 1. YDATA-PROFILING 적용
# ============================================================================
print("\n" + "="*60)
print("1️⃣ YDATA-PROFILING 적용")
print("="*60)

import ydata_profiling as yp

print("📈 ydata-profiling으로 상세 분석 리포트 생성 중...")
profile = yp.ProfileReport(df, title="Titanic Dataset Analysis")
profile_path = os.path.join(packages["ydata_profiling"], "titanic_ydata_profiling.html")
profile.to_file(profile_path)
print(f"✅ HTML 리포트가 '{profile_path}'로 저장되었습니다!")

# ============================================================================
# 2. SWEETVIZ 적용
# ============================================================================
print("\n" + "="*60)
print("2️⃣ SWEETVIZ 적용")
print("="*60)

import sweetviz as sv

print("🍯 Sweetviz로 데이터 분석 리포트 생성 중...")

# survived 컬럼을 숫자로 변환
df_for_sweetviz = df.copy()
df_for_sweetviz['survived'] = pd.to_numeric(df_for_sweetviz['survived'], errors='coerce')

# Sweetviz 리포트 생성
report = sv.analyze([df_for_sweetviz, "Titanic Dataset"], target_feat='survived')
sweetviz_path = os.path.join(packages["sweetviz"], "titanic_sweetviz_report.html")
report.show_html(sweetviz_path)

print(f"✅ Sweetviz HTML 리포트가 '{sweetviz_path}'로 저장되었습니다!")

# ============================================================================
# 3. AUTOVIZ 적용
# ============================================================================
print("\n" + "="*60)
print("3️⃣ AUTOVIZ 적용")
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
    save_plot_dir=packages["autoviz"],  # 플롯 저장 디렉토리
    chart_format='png'       # PNG 형식으로 저장
)

print(f"✅ AutoViz 시각화가 '{packages['autoviz']}' 폴더에 저장되었습니다!")

# ============================================================================
# 4. KLIB 적용 (엑셀로 저장)
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
distribution_data = []
for col in numeric_cols:
    if col != 'survived':  # survived는 제외
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"- {col}: 평균={mean_val:.2f}, 표준편차={std_val:.2f}")
        distribution_data.append({
            '변수명': col,
            '평균': mean_val,
            '표준편차': std_val,
            '최소값': df[col].min(),
            '최대값': df[col].max(),
            '중앙값': df[col].median()
        })

# 생존률 분석
print("\n📊 생존률 분석:")
survived_numeric = pd.to_numeric(df['survived'], errors='coerce')
overall_survival = survived_numeric.mean()

survival_data = []
survival_data.append({
    '분류': '전체',
    '생존률': overall_survival,
    '생존자 수': survived_numeric.sum(),
    '전체 수': len(df)
})

# 성별 생존률
if 'sex' in df.columns:
    survival_by_sex = df.groupby('sex')['survived'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
    print(f"\n👥 성별 생존률:")
    for sex, rate in survival_by_sex.items():
        print(f"  {sex}: {rate:.2%}")
        # 생존자 수 계산을 위해 먼저 숫자로 변환
        sex_df = df[df['sex'] == sex]
        survived_numeric_sex = pd.to_numeric(sex_df['survived'], errors='coerce')
        survival_data.append({
            '분류': f'성별_{sex}',
            '생존률': rate,
            '생존자 수': survived_numeric_sex.sum(),
            '전체 수': len(sex_df)
        })

# 클래스별 생존률
if 'pclass' in df.columns:
    survival_by_class = df.groupby('pclass')['survived'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
    print(f"\n🎫 클래스별 생존률:")
    for pclass, rate in survival_by_class.items():
        print(f"  {pclass}등급: {rate:.2%}")
        # 생존자 수 계산을 위해 먼저 숫자로 변환
        class_df = df[df['pclass'] == pclass]
        survived_numeric_class = pd.to_numeric(class_df['survived'], errors='coerce')
        survival_data.append({
            '분류': f'클래스_{pclass}등급',
            '생존률': rate,
            '생존자 수': survived_numeric_class.sum(),
            '전체 수': len(class_df)
        })

# 나이별 생존률
if 'age' in df.columns:
    df['age_group'] = pd.cut(pd.to_numeric(df['age'], errors='coerce'), 
                             bins=[0, 18, 30, 50, 100], 
                             labels=['Child', 'Young', 'Adult', 'Senior'])
    age_survival = df.groupby('age_group')['survived'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
    print(f"\n👶 나이대별 생존률:")
    for age_group, rate in age_survival.items():
        print(f"  {age_group}: {rate:.2%}")
        # 생존자 수 계산을 위해 먼저 숫자로 변환
        age_df = df[df['age_group'] == age_group]
        survived_numeric_age = pd.to_numeric(age_df['survived'], errors='coerce')
        survival_data.append({
            '분류': f'나이대_{age_group}',
            '생존률': rate,
            '생존자 수': survived_numeric_age.sum(),
            '전체 수': len(age_df)
        })

# 엑셀 파일로 저장
print("\n💾 Klib 분석 결과를 엑셀로 저장 중...")

with pd.ExcelWriter(os.path.join(packages["klib"], "titanic_klib_analysis.xlsx")) as writer:
    # 결측치 분석
    missing_df.to_excel(writer, sheet_name='결측치_분석', index=True)
    
    # 상관관계 분석
    correlation_matrix.to_excel(writer, sheet_name='상관관계_분석', index=True)
    
    # 분포 분석
    distribution_df = pd.DataFrame(distribution_data)
    distribution_df.to_excel(writer, sheet_name='분포_분석', index=False)
    
    # 생존률 분석
    survival_df = pd.DataFrame(survival_data)
    survival_df.to_excel(writer, sheet_name='생존률_분석', index=False)
    
    # 원본 데이터 (샘플)
    df.head(100).to_excel(writer, sheet_name='데이터_샘플', index=False)

print(f"✅ Klib 분석 결과가 '{os.path.join(packages['klib'], 'titanic_klib_analysis.xlsx')}'로 저장되었습니다!")

# ============================================================================
# 5. D-TALE 적용
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
# 6. 생성된 파일들 확인
# ============================================================================
print("\n" + "="*60)
print("📁 생성된 파일들 확인")
print("="*60)

# 각 패키지별 파일 확인
for package_name, package_path in packages.items():
    if os.path.exists(package_path):
        files = []
        for root, dirs, filenames in os.walk(package_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                file_size = os.path.getsize(file_path) / 1024  # KB
                files.append(f"  - {filename} ({file_size:.0f}KB)")
        
        print(f"✅ {package_name}/ 폴더: {len(files)}개 파일")
        for file_info in files:
            print(file_info)

print("\n" + "="*60)
print("🎉 모든 EDA 패키지 분석이 완료되었습니다!")
print("💡 각 패키지의 특징:")
print("   • ydata-profiling: 포괄적인 데이터 품질 분석")
print("   • Sweetviz: 타겟 변수 중심의 상세 분석")
print("   • AutoViz: 자동 시각화 및 패턴 발견")
print("   • Klib: 데이터 클리닝 및 엑셀 분석 결과")
print("   • D-Tale: 대화형 데이터 탐색 인터페이스 (포트 4000)")
print("="*60)

print("\n⏳ D-Tale이 실행 중입니다. Ctrl+C를 눌러서 종료하세요.")

# 무한 루프로 서버 유지
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 분석을 종료합니다.") 