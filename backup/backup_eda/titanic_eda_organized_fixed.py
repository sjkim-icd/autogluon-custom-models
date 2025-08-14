import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
import warnings
import os
from datetime import datetime
import glob
import base64 # Added for HTML report
warnings.filterwarnings('ignore')

# 한글 폰트 설정
import matplotlib.font_manager as fm

# 윈도우 한글 폰트 설정
try:
    # 윈도우 기본 한글 폰트들
    font_list = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Dotum', 'Batang']
    font_found = False
    
    for font_name in font_list:
        try:
            plt.rcParams['font.family'] = font_name
            # 테스트용 텍스트로 폰트 확인
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '한글', fontsize=12)
            plt.close(fig)
            font_found = True
            print(f"✅ 한글 폰트 설정 완료: {font_name}")
            break
        except:
            continue
    
    if not font_found:
        # 폰트를 찾지 못한 경우 기본 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("⚠️ 한글 폰트를 찾지 못했습니다. 기본 폰트를 사용합니다.")
        
except Exception as e:
    print(f"⚠️ 폰트 설정 중 오류: {e}")

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

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
import matplotlib.pyplot as plt
import seaborn as sns

print("🎨 AutoViz로 자동 시각화 생성 중...")

# AutoViz 실행 전 한글 폰트 재설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

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

# 개별 변수별 시각화 추가
print("📊 개별 변수별 시각화 생성 중...")

# 수치형 변수별 분포
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'survived':
        try:
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'{col} 분포')
            plt.xlabel(col)
            plt.ylabel('빈도')
            
            plt.subplot(1, 2, 2)
            survived_numeric = pd.to_numeric(df['survived'], errors='coerce')
            plt.scatter(df[col], survived_numeric, alpha=0.5)
            plt.title(f'{col} vs 생존')
            plt.xlabel(col)
            plt.ylabel('생존 (0/1)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(packages["autoviz"], f'autoviz_{col}_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ {col} 개별 분석 저장")
        except Exception as e:
            print(f"❌ {col} 분석 오류: {e}")

# 범주형 변수별 분포
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    if col not in ['name', 'ticket', 'cabin', 'boat', 'home.dest']:
        try:
            plt.figure(figsize=(12, 5))
            
            # 값 분포
            plt.subplot(1, 2, 1)
            value_counts = df[col].value_counts().head(10)
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.title(f'{col} 상위 10개 값')
            plt.xlabel('값')
            plt.ylabel('빈도')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
            
            # 생존률 비교
            plt.subplot(1, 2, 2)
            survival_by_cat = df.groupby(col)['survived'].apply(
                lambda x: pd.to_numeric(x, errors='coerce').mean()
            ).sort_values(ascending=False).head(10)
            plt.bar(range(len(survival_by_cat)), survival_by_cat.values)
            plt.title(f'{col}별 생존률')
            plt.xlabel('값')
            plt.ylabel('생존률')
            plt.xticks(range(len(survival_by_cat)), survival_by_cat.index, rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(packages["autoviz"], f'autoviz_{col}_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ {col} 개별 분석 저장")
        except Exception as e:
            print(f"❌ {col} 분석 오류: {e}")

# 상관관계 히트맵 (수치형 변수들)
if len(numeric_cols) > 1:
    try:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('수치형 변수 상관관계')
        plt.tight_layout()
        plt.savefig(os.path.join(packages["autoviz"], 'autoviz_correlation_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 상관관계 히트맵 저장")
    except Exception as e:
        print(f"❌ 상관관계 히트맵 오류: {e}")

# 생존률 분석 시각화
try:
    plt.figure(figsize=(15, 10))
    
    # 성별 생존률
    plt.subplot(2, 3, 1)
    survival_by_sex = df.groupby('sex')['survived'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').mean()
    )
    colors = ['pink', 'lightblue']
    plt.bar(survival_by_sex.index, survival_by_sex.values, color=colors)
    plt.title('성별 생존률')
    plt.ylabel('생존률')
    plt.ylim(0, 1)
    
    # 클래스별 생존률
    plt.subplot(2, 3, 2)
    survival_by_class = df.groupby('pclass')['survived'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').mean()
    )
    plt.bar(survival_by_class.index, survival_by_class.values, color='lightgreen')
    plt.title('클래스별 생존률')
    plt.xlabel('클래스')
    plt.ylabel('생존률')
    plt.ylim(0, 1)
    
    # 나이대별 생존률
    plt.subplot(2, 3, 3)
    df['age_group'] = pd.cut(df['age'], bins=[0, 12, 25, 65, 100], 
                             labels=['Child', 'Young', 'Adult', 'Senior'])
    survival_by_age = df.groupby('age_group')['survived'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').mean()
    )
    plt.bar(survival_by_age.index, survival_by_age.values, color='orange')
    plt.title('나이대별 생존률')
    plt.ylabel('생존률')
    plt.ylim(0, 1)
    
    # 가족 규모별 생존률
    plt.subplot(2, 3, 4)
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    survival_by_family = df.groupby('family_size')['survived'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').mean()
    ).head(8)  # 상위 8개만 표시
    plt.bar(survival_by_family.index, survival_by_family.values, color='purple')
    plt.title('가족 규모별 생존률')
    plt.xlabel('가족 규모')
    plt.ylabel('생존률')
    plt.ylim(0, 1)
    
    # 요금 구간별 생존률
    plt.subplot(2, 3, 5)
    df['fare_group'] = pd.cut(df['fare'], bins=5, labels=['낮음', '낮은중간', '중간', '높은중간', '높음'])
    survival_by_fare = df.groupby('fare_group')['survived'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').mean()
    )
    plt.bar(survival_by_fare.index, survival_by_fare.values, color='gold')
    plt.title('요금 구간별 생존률')
    plt.ylabel('생존률')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # 승선항별 생존률
    plt.subplot(2, 3, 6)
    survival_by_embarked = df.groupby('embarked')['survived'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').mean()
    )
    plt.bar(survival_by_embarked.index, survival_by_embarked.values, color='lightcoral')
    plt.title('승선항별 생존률')
    plt.ylabel('생존률')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(packages["autoviz"], 'autoviz_survival_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 생존률 종합 분석 저장")
except Exception as e:
    print(f"❌ 생존률 분석 오류: {e}")

# ============================================================================
# 고급 AutoViz 시각화 추가
# ============================================================================
print("🎨 고급 AutoViz 시각화 생성 중...")

# 1. 박스플롯 (Box Plot) - 이상치 탐지
try:
    plt.figure(figsize=(15, 10))
    
    # 수치형 변수들의 박스플롯
    numeric_cols_for_box = [col for col in numeric_cols if col not in ['survived', 'body']]
    
    for i, col in enumerate(numeric_cols_for_box, 1):
        plt.subplot(2, 3, i)
        plt.boxplot(df[col].dropna())
        plt.title(f'{col} 박스플롯 (이상치 탐지)')
        plt.ylabel(col)
    
    plt.tight_layout()
    plt.savefig(os.path.join(packages["autoviz"], 'autoviz_boxplots_outliers.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 박스플롯 (이상치 탐지) 저장")
except Exception as e:
    print(f"❌ 박스플롯 오류: {e}")

# 2. 바이올린 플롯 (Violin Plot) - 분포 비교
try:
    plt.figure(figsize=(15, 10))
    
    # 성별에 따른 나이 분포
    plt.subplot(2, 3, 1)
    import seaborn as sns
    sns.violinplot(data=df, x='sex', y='age', hue='survived')
    plt.title('성별에 따른 나이 분포 (생존 여부)')
    
    # 클래스별 요금 분포
    plt.subplot(2, 3, 2)
    sns.violinplot(data=df, x='pclass', y='fare', hue='survived')
    plt.title('클래스별 요금 분포 (생존 여부)')
    
    # 승선항별 나이 분포
    plt.subplot(2, 3, 3)
    sns.violinplot(data=df, x='embarked', y='age', hue='survived')
    plt.title('승선항별 나이 분포 (생존 여부)')
    
    # 가족 규모별 나이 분포
    plt.subplot(2, 3, 4)
    sns.violinplot(data=df, x='family_size', y='age', hue='survived')
    plt.title('가족 규모별 나이 분포 (생존 여부)')
    
    # 성별 요금 분포
    plt.subplot(2, 3, 5)
    sns.violinplot(data=df, x='sex', y='fare', hue='survived')
    plt.title('성별 요금 분포 (생존 여부)')
    
    # 클래스별 나이 분포
    plt.subplot(2, 3, 6)
    sns.violinplot(data=df, x='pclass', y='age', hue='survived')
    plt.title('클래스별 나이 분포 (생존 여부)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(packages["autoviz"], 'autoviz_violin_plots.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 바이올린 플롯 저장")
except Exception as e:
    print(f"❌ 바이올린 플롯 오류: {e}")

# 3. 페어플롯 (Pair Plot) - 변수 간 관계
try:
    # 주요 수치형 변수들만 선택
    pair_cols = ['age', 'fare', 'pclass', 'sibsp', 'parch']
    pair_df = df[pair_cols + ['survived']].copy()
    pair_df['survived'] = pd.to_numeric(pair_df['survived'], errors='coerce')
    
    plt.figure(figsize=(20, 16))
    sns.pairplot(pair_df, hue='survived', diag_kind='hist', 
                 plot_kws={'alpha': 0.6}, diag_kws={'bins': 20})
    plt.savefig(os.path.join(packages["autoviz"], 'autoviz_pair_plot.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 페어플롯 저장")
except Exception as e:
    print(f"❌ 페어플롯 오류: {e}")

# 4. 히트맵 확장 (더 상세한 상관관계)
try:
    plt.figure(figsize=(12, 10))
    
    # 수치형 변수들 + 생존률
    corr_cols = numeric_cols.tolist() + ['survived_numeric']
    df_corr = df[numeric_cols].copy()
    df_corr['survived_numeric'] = pd.to_numeric(df['survived'], errors='coerce')
    
    correlation_matrix = df_corr.corr()
    
    # 마스크 생성 (상삼각형만 표시)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
               square=True, linewidths=0.5, fmt='.2f')
    plt.title('상세 상관관계 히트맵')
    plt.tight_layout()
    plt.savefig(os.path.join(packages["autoviz"], 'autoviz_detailed_correlation.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 상세 상관관계 히트맵 저장")
except Exception as e:
    print(f"❌ 상세 상관관계 히트맵 오류: {e}")

# 5. 스트리프 플롯 (Strip Plot) - 분포와 산점도 결합
try:
    plt.figure(figsize=(15, 10))
    
    # 성별에 따른 나이 분포
    plt.subplot(2, 3, 1)
    sns.stripplot(data=df, x='sex', y='age', hue='survived', jitter=0.3, alpha=0.6)
    plt.title('성별 나이 분포 (생존 여부)')
    
    # 클래스별 요금 분포
    plt.subplot(2, 3, 2)
    sns.stripplot(data=df, x='pclass', y='fare', hue='survived', jitter=0.3, alpha=0.6)
    plt.title('클래스별 요금 분포 (생존 여부)')
    
    # 승선항별 나이 분포
    plt.subplot(2, 3, 3)
    sns.stripplot(data=df, x='embarked', y='age', hue='survived', jitter=0.3, alpha=0.6)
    plt.title('승선항별 나이 분포 (생존 여부)')
    
    # 가족 규모별 나이 분포
    plt.subplot(2, 3, 4)
    sns.stripplot(data=df, x='family_size', y='age', hue='survived', jitter=0.3, alpha=0.6)
    plt.title('가족 규모별 나이 분포 (생존 여부)')
    
    # 성별 요금 분포
    plt.subplot(2, 3, 5)
    sns.stripplot(data=df, x='sex', y='fare', hue='survived', jitter=0.3, alpha=0.6)
    plt.title('성별 요금 분포 (생존 여부)')
    
    # 클래스별 나이 분포
    plt.subplot(2, 3, 6)
    sns.stripplot(data=df, x='pclass', y='age', hue='survived', jitter=0.3, alpha=0.6)
    plt.title('클래스별 나이 분포 (생존 여부)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(packages["autoviz"], 'autoviz_strip_plots.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 스트리프 플롯 저장")
except Exception as e:
    print(f"❌ 스트리프 플롯 오류: {e}")

# 6. 스왈름 플롯 (Swarm Plot) - 정확한 분포
try:
    plt.figure(figsize=(15, 10))
    
    # 성별에 따른 나이 분포
    plt.subplot(2, 3, 1)
    sns.swarmplot(data=df, x='sex', y='age', hue='survived', size=3)
    plt.title('성별 나이 분포 (생존 여부)')
    
    # 클래스별 요금 분포
    plt.subplot(2, 3, 2)
    sns.swarmplot(data=df, x='pclass', y='fare', hue='survived', size=3)
    plt.title('클래스별 요금 분포 (생존 여부)')
    
    # 승선항별 나이 분포
    plt.subplot(2, 3, 3)
    sns.swarmplot(data=df, x='embarked', y='age', hue='survived', size=3)
    plt.title('승선항별 나이 분포 (생존 여부)')
    
    # 가족 규모별 나이 분포
    plt.subplot(2, 3, 4)
    sns.swarmplot(data=df, x='family_size', y='age', hue='survived', size=3)
    plt.title('가족 규모별 나이 분포 (생존 여부)')
    
    # 성별 요금 분포
    plt.subplot(2, 3, 5)
    sns.swarmplot(data=df, x='sex', y='fare', hue='survived', size=3)
    plt.title('성별 요금 분포 (생존 여부)')
    
    # 클래스별 나이 분포
    plt.subplot(2, 3, 6)
    sns.swarmplot(data=df, x='pclass', y='age', hue='survived', size=3)
    plt.title('클래스별 나이 분포 (생존 여부)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(packages["autoviz"], 'autoviz_swarm_plots.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 스왈름 플롯 저장")
except Exception as e:
    print(f"❌ 스왈름 플롯 오류: {e}")

# 7. 조인트 플롯 (Joint Plot) - 2변수 관계 + 분포
try:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 나이 vs 요금
    sns.jointplot(data=df, x='age', y='fare', hue='survived', 
                  kind='scatter', alpha=0.6, ax=axes[0,0])
    axes[0,0].set_title('나이 vs 요금 (생존 여부)')
    
    # 나이 vs 클래스
    sns.jointplot(data=df, x='age', y='pclass', hue='survived', 
                  kind='scatter', alpha=0.6, ax=axes[0,1])
    axes[0,1].set_title('나이 vs 클래스 (생존 여부)')
    
    # 요금 vs 클래스
    sns.jointplot(data=df, x='fare', y='pclass', hue='survived', 
                  kind='scatter', alpha=0.6, ax=axes[0,2])
    axes[0,2].set_title('요금 vs 클래스 (생존 여부)')
    
    # 가족 규모 vs 나이
    sns.jointplot(data=df, x='family_size', y='age', hue='survived', 
                  kind='scatter', alpha=0.6, ax=axes[1,0])
    axes[1,0].set_title('가족 규모 vs 나이 (생존 여부)')
    
    # 가족 규모 vs 요금
    sns.jointplot(data=df, x='family_size', y='fare', hue='survived', 
                  kind='scatter', alpha=0.6, ax=axes[1,1])
    axes[1,1].set_title('가족 규모 vs 요금 (생존 여부)')
    
    # sibsp vs parch
    sns.jointplot(data=df, x='sibsp', y='parch', hue='survived', 
                  kind='scatter', alpha=0.6, ax=axes[1,2])
    axes[1,2].set_title('형제자매 vs 부모자식 (생존 여부)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(packages["autoviz"], 'autoviz_joint_plots.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 조인트 플롯 저장")
except Exception as e:
    print(f"❌ 조인트 플롯 오류: {e}")

print("🎨 고급 AutoViz 시각화 완료!")

# AutoViz 결과 보장: 폴더가 비어있으면 대체 출력 경로에서 이동
try:
    import shutil
    autoviz_dir = packages["autoviz"]
    has_files = any(os.path.isfile(os.path.join(autoviz_dir, f)) for f in os.listdir(autoviz_dir))
    if not has_files:
        alt_dirs = [
            os.path.join(os.getcwd(), "AutoViz_Plots"),
            os.path.join(os.getcwd(), "autoviz_plots"),
        ]
        picked = None
        for d in alt_dirs:
            if os.path.isdir(d):
                picked = d
                break
        if not picked:
            # 프로젝트 전체에서 AutoViz_Plots 검색 (최초 1곳)
            for root, dirs, files in os.walk(os.getcwd()):
                if os.path.basename(root).lower() in ("autoviz_plots",):
                    picked = root
                    break
        if picked:
            for root, dirs, files in os.walk(picked):
                rel = os.path.relpath(root, picked)
                dest = os.path.join(autoviz_dir, rel) if rel != "." else autoviz_dir
                os.makedirs(dest, exist_ok=True)
                for fname in files:
                    src = os.path.join(root, fname)
                    shutil.copy2(src, os.path.join(dest, fname))
            print(f"📦 AutoViz 대체 경로에서 결과를 복사했습니다: {picked} -> {autoviz_dir}")
except Exception as e:
    print(f"❌ AutoViz 결과 이동 중 오류: {e}")

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

# Klib 결측치 시각화 저장
try:
    plt.figure()
    klib.missingval_plot(df)
    mv_plot_path = os.path.join(packages["klib"], "missing_values_plot.png")
    plt.savefig(mv_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 결측치 시각화 저장: {mv_plot_path}")
except Exception as e:
    print(f"❌ missingval_plot 오류: {e}")

# 상관관계 분석
print("\n📊 상관관계 분석:")
# 수치형 컬럼만 선택
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    correlation_matrix = df[numeric_cols].corr()
    print("수치형 변수 간 상관관계:")
    print(correlation_matrix.round(3))
    
    # klib corr_plot 저장
    try:
        plt.figure()
        klib.corr_plot(df[numeric_cols])
        corr_plot_path = os.path.join(packages["klib"], "correlation_plot.png")
        plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 상관관계 히트맵 저장: {corr_plot_path}")
    except Exception as e:
        print(f"❌ corr_plot 오류: {e}")

    # klib corr_interactive_plot (저장 불가 시 실행만)
    try:
        klib.corr_interactive_plot(df[numeric_cols])
        print("✅ 인터랙티브 상관관계 플롯 생성")
    except Exception as e:
        print(f"❌ corr_interactive_plot 오류: {e}")

# 분포 분석 (klib.dist_plot는 DataFrame 입력, figsize 인자 없음)
try:
    plt.figure()
    if len(numeric_cols) > 0:
        klib.dist_plot(df[numeric_cols])
        dist_plot_path = os.path.join(packages["klib"], "dist_plot.png")
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 분포 시각화 저장: {dist_plot_path}")
except Exception as e:
    print(f"❌ dist_plot 오류: {e}")

# 범주형 분석 (klib.cat_plot는 DataFrame 입력, 컬럼명 인자 없음)
try:
    plt.figure()
    klib.cat_plot(df)
    cat_plot_path = os.path.join(packages["klib"], "cat_plot.png")
    plt.savefig(cat_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 범주형 시각화 저장: {cat_plot_path}")
except Exception as e:
    print(f"❌ cat_plot 오류: {e}")

# 생존률 분석 (수치 변환 포함)
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
        age_df = df[df['age_group'] == age_group]
        survived_numeric_age = pd.to_numeric(age_df['survived'], errors='coerce')
        survival_data.append({
            '분류': f'나이대_{age_group}',
            '생존률': rate,
            '생존자 수': survived_numeric_age.sum(),
            '전체 수': len(age_df)
        })

# 엑셀 파일로 저장 (확장)
print("\n💾 Klib 분석 결과를 엑셀로 저장 중...")

excel_output_path = os.path.join(packages["klib"], "titanic_klib_analysis.xlsx")
# 기존 파일이 열려있어 PermissionError가 나면 새 파일명으로 저장
try:
    if os.path.exists(excel_output_path):
        os.remove(excel_output_path)
except PermissionError:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_output_path = os.path.join(packages["klib"], f"titanic_klib_analysis_{timestamp}.xlsx")
    print(f"⚠️ 파일이 열려 있어 새 파일로 저장합니다: {excel_output_path}")

try:
    with pd.ExcelWriter(excel_output_path) as writer:
        # 결측치 분석
        missing_df.to_excel(writer, sheet_name='결측치_분석', index=True)
        
        # 상관관계 분석 (없으면 빈 DF 저장)
        try:
            correlation_matrix
        except NameError:
            correlation_matrix = pd.DataFrame()
        correlation_matrix.to_excel(writer, sheet_name='상관관계_분석', index=True)
        
        # 분포 분석(요약)
        distribution_data = []
        for col in numeric_cols:
            if col != 'survived':
                distribution_data.append({
                    '변수명': col,
                    '평균': df[col].mean(),
                    '표준편차': df[col].std(),
                    '최소값': df[col].min(),
                    '최대값': df[col].max(),
                    '중앙값': df[col].median()
                })
        pd.DataFrame(distribution_data).to_excel(writer, sheet_name='분포_분석', index=False)
        
        # 생존률 분석
        pd.DataFrame(survival_data).to_excel(writer, sheet_name='생존률_분석', index=False)

        # 결측치 처리 제안 (klib.mv_col_handling 요약 + 샘플)
        try:
            mv_result = klib.mv_col_handling(df.copy())
            summary_rows = []
            sample_dict = {}
            for col, series in mv_result.items():
                orig_missing = int(df[col].isna().sum()) if col in df.columns else None
                s = pd.Series(series)
                after_missing = int(s.isna().sum())
                summary_rows.append({
                    '컬럼명': col,
                    '원본_결측치수': orig_missing,
                    '처리후_결측치수': after_missing,
                    'dtype': str(s.dtype),
                    '고유값수(처리후)': int(s.nunique(dropna=False))
                })
                sample_dict[col] = list(s.head(10))
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='결측치_처리제안_요약', index=False)
            # 샘플 값 테이블 (열: 컬럼명, 행: 상위10개)
            max_len = max((len(v) for v in sample_dict.values()), default=0)
            for k in list(sample_dict.keys()):
                if len(sample_dict[k]) < max_len:
                    sample_dict[k] += [None] * (max_len - len(sample_dict[k]))
            if sample_dict:
                pd.DataFrame(sample_dict).to_excel(writer, sheet_name='결측치_처리_샘플', index=False)
        except Exception as e:
            pd.DataFrame({'오류': [str(e)]}).to_excel(writer, sheet_name='결측치_처리제안_오류', index=False)

        # 데이터 정보 요약(타입/결측/고유값)
        info_df = pd.DataFrame({
            '컬럼명': df.columns,
            '데이터타입': df.dtypes.astype(str),
            '결측치수': df.isnull().sum(),
            '결측치비율(%)': (df.isnull().sum() / len(df)) * 100,
            '고유값수': [df[c].nunique() for c in df.columns]
        })
        info_df.to_excel(writer, sheet_name='데이터_정보', index=False)

        # 수치형/범주형 통계 (기술 통계)
        if len(numeric_cols) > 0:
            df[numeric_cols].describe().to_excel(writer, sheet_name='수치형_통계')
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        cat_stats_rows = []
        for c in categorical_cols:
            vc = df[c].value_counts(dropna=False)
            top_val = vc.index[0] if len(vc) else None
            top_cnt = int(vc.iloc[0]) if len(vc) else 0
            cat_stats_rows.append({'컬럼명': c, '고유값수': int(df[c].nunique(dropna=False)), '최빈값': top_val, '최빈값빈도': top_cnt})
        pd.DataFrame(cat_stats_rows).to_excel(writer, sheet_name='범주형_통계', index=False)

        # 원본 데이터 (샘플)
        df.head(100).to_excel(writer, sheet_name='데이터_샘플', index=False)
except PermissionError:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_output_path = os.path.join(packages["klib"], f"titanic_klib_analysis_{timestamp}.xlsx")
    with pd.ExcelWriter(excel_output_path) as writer:
        missing_df.to_excel(writer, sheet_name='결측치_분석', index=True)
        try:
            correlation_matrix
        except NameError:
            correlation_matrix = pd.DataFrame()
        correlation_matrix.to_excel(writer, sheet_name='상관관계_분석', index=True)
        pd.DataFrame(distribution_data).to_excel(writer, sheet_name='분포_분석', index=False)
        pd.DataFrame(survival_data).to_excel(writer, sheet_name='생존률_분석', index=False)
        df.head(100).to_excel(writer, sheet_name='데이터_샘플', index=False)
    print(f"⚠️ 엑셀이 열려 있어 새 파일로 저장했습니다: {excel_output_path}")

print(f"✅ Klib 분석 결과가 '{excel_output_path}'로 저장되었습니다!")

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

# 선택적 ngrok 공개 URL (환경변수 USE_NGROK=1 설정 시)
try:
    use_ngrok = str(os.getenv('USE_NGROK', '0')).lower() in ['1', 'true', 'yes']
    if use_ngrok:
        try:
            from pyngrok import ngrok, conf
            authtoken = os.getenv('NGROK_AUTHTOKEN', '')
            if authtoken:
                # 최신 pyngrok에서는 conf로 기본 토큰 설정 또는 set_auth_token 사용
                try:
                    ngrok.set_auth_token(authtoken)
                except Exception:
                    conf.get_default().auth_token = authtoken
            public_url = ngrok.connect(4000, bind_tls=True)
            print(f"🔗 ngrok 공개 URL: {public_url}")
        except Exception as e:
            print(f"❌ ngrok 설정 실패: {e}")
except Exception:
    pass

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

# ============================================================================
# 7. AUTOVIZ 이미지 HTML 보고서 생성
# ============================================================================
print("\n" + "="*60)
print("🖼️ AUTOVIZ 이미지 HTML 보고서 생성")
print("="*60)

def create_autoviz_html_report():
    """AutoViz 폴더의 모든 이미지를 읽어서 HTML 보고서 생성"""
    autoviz_folder = packages["autoviz"]
    
    # PNG 이미지 파일들 찾기
    image_files = glob.glob(os.path.join(autoviz_folder, "*.png"))
    
    if not image_files:
        print("❌ AutoViz 폴더에서 이미지를 찾을 수 없습니다.")
        return
    
    print(f"📸 {len(image_files)}개의 이미지를 발견했습니다.")
    
    # HTML 시작
    html_content = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoViz 타이타닉 데이터 분석 보고서</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .image-section {
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: #fafafa;
        }
        .image-title {
            font-size: 1.5em;
            color: #34495e;
            margin-bottom: 15px;
            font-weight: bold;
            text-align: center;
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .file-info {
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 10px;
        }
        .summary {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .summary h2 {
            color: #2c3e50;
            margin-top: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚢 AutoViz 타이타닉 데이터 분석 보고서</h1>
        
        <div class="summary">
            <h2>📊 분석 개요</h2>
            <p>이 보고서는 AutoViz를 사용하여 타이타닉 데이터셋을 분석한 결과입니다. 
            각 이미지는 데이터의 다양한 측면을 시각화하여 보여줍니다.</p>
            <p><strong>생성 시간:</strong> """ + datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S") + """</p>
            <p><strong>총 이미지 수:</strong> """ + str(len(image_files)) + """개</p>
        </div>
    """
    
    # 각 이미지에 대해 HTML 섹션 생성
    for i, image_path in enumerate(sorted(image_files), 1):
        filename = os.path.basename(image_path)
        file_size = os.path.getsize(image_path) / 1024  # KB
        
        # 파일명에서 제목 추출 (확장자 제거, 언더스코어를 공백으로 변경)
        title = filename.replace('.png', '').replace('_', ' ').title()
        
        # 이미지를 base64로 인코딩
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                html_content += f"""
        <div class="image-section">
            <div class="image-title">{title}</div>
            <div class="image-container">
                <img src="data:image/png;base64,{img_base64}" alt="{title}">
            </div>
            <div class="file-info">
                파일명: {filename} | 크기: {file_size:.1f}KB
            </div>
        </div>
                """
        except Exception as e:
            print(f"❌ 이미지 로드 실패 ({filename}): {e}")
            html_content += f"""
        <div class="image-section">
            <div class="image-title">{title}</div>
            <div class="image-container">
                <p style="color: red;">이미지를 로드할 수 없습니다: {filename}</p>
            </div>
        </div>
            """
    
    # HTML 끝
    html_content += """
    </div>
</body>
</html>
    """
    
    # HTML 파일 저장
    html_output_path = os.path.join(autoviz_folder, "autoviz_analysis_report.html")
    try:
        with open(html_output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ AutoViz HTML 보고서가 '{html_output_path}'로 저장되었습니다!")
        print(f"📊 총 {len(image_files)}개의 이미지가 포함되었습니다.")
        
        # 브라우저에서 열기
        try:
            webbrowser.open(f"file://{os.path.abspath(html_output_path)}")
            print("🌐 브라우저에서 HTML 보고서를 열었습니다!")
        except:
            print("💡 브라우저에서 수동으로 HTML 파일을 열어보세요.")
            
    except Exception as e:
        print(f"❌ HTML 파일 저장 실패: {e}")

# HTML 보고서 생성 실행
create_autoviz_html_report()

# 무한 루프로 서버 유지
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 분석을 종료합니다.") 