import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
import warnings
import os
import klib
warnings.filterwarnings('ignore')

# 타이타닉 데이터 로드
print("🚢 타이타닉 데이터 로딩 중...")
titanic = fetch_openml(name='titanic', version=1, as_frame=True)
df = titanic.frame

print(f"📊 데이터 형태: {df.shape}")
print(f"📋 컬럼: {df.columns.tolist()}")

# ============================================================================
# 폴더 생성
# ============================================================================
print("\n" + "="*60)
print("📁 Klib 고급 분석 결과 폴더 생성")
print("="*60)

# Klib 분석 폴더 생성
klib_folder = "EDA/titanic/klib_advanced"
if not os.path.exists(klib_folder):
    os.makedirs(klib_folder)
    print(f"📁 '{klib_folder}' 폴더를 생성했습니다.")

print("✅ 폴더 생성 완료!")

# ============================================================================
# 1. 데이터 클리닝 (Klib의 핵심 기능)
# ============================================================================
print("\n" + "="*60)
print("1️⃣ KLIB 데이터 클리닝")
print("="*60)

print("🧹 Klib로 데이터 클리닝 중...")

# 컬럼명 정리
print("📝 컬럼명 정리:")
df_cleaned = klib.clean_column_names(df)
print(f"  - 정리된 컬럼명: {df_cleaned.columns.tolist()}")

# 데이터 타입 변환
print("\n🔄 데이터 타입 변환:")
df_cleaned = klib.convert_datatypes(df_cleaned)
print("  - 데이터 타입이 자동으로 최적화되었습니다.")

# 중복 데이터 처리
print("\n🔍 중복 데이터 분석:")
try:
    duplicate_subsets = klib.pool_duplicate_subsets(df_cleaned)
    if isinstance(duplicate_subsets, pd.DataFrame) and not duplicate_subsets.empty:
        print(f"  - 발견된 중복 서브셋: {len(duplicate_subsets)}개")
        for i, subset in enumerate(duplicate_subsets.head(3).itertuples()):
            print(f"    * {subset}")
    else:
        print("  - 중복 서브셋이 발견되지 않았습니다.")
except Exception as e:
    print(f"  - 중복 데이터 분석 중 오류: {e}")

# ============================================================================
# 2. 결측치 분석 (Klib의 고급 기능)
# ============================================================================
print("\n" + "="*60)
print("2️⃣ KLIB 결측치 분석")
print("="*60)

print("🔍 결측치 상세 분석:")

# 결측치 시각화
missing_plot_path = os.path.join(klib_folder, "missing_values_plot.png")
try:
    klib.missingval_plot(df_cleaned, figsize=(12, 8))
    plt.savefig(missing_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 결측치 시각화가 '{missing_plot_path}'로 저장되었습니다!")
except Exception as e:
    print(f"❌ 결측치 시각화 오류: {e}")

# 결측치 처리
print("\n🛠️ 결측치 처리:")
try:
    missing_handling = klib.mv_col_handling(df_cleaned)
    print("  - 결측치 처리 방법 제안:")
    for col, method in missing_handling.items():
        print(f"    * {col}: {method}")
except Exception as e:
    print(f"❌ 결측치 처리 분석 오류: {e}")

# ============================================================================
# 3. 상관관계 분석 (Klib의 고급 기능)
# ============================================================================
print("\n" + "="*60)
print("3️⃣ KLIB 상관관계 분석")
print("="*60)

print("📊 상관관계 분석:")

# 수치형 컬럼만 선택
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    # 상관관계 매트릭스
    try:
        corr_matrix = klib.corr_mat(df_cleaned[numeric_cols])
        print("  - 상관관계 매트릭스 생성 완료")
    except Exception as e:
        print(f"❌ 상관관계 매트릭스 오류: {e}")
    
    # 상관관계 시각화
    corr_plot_path = os.path.join(klib_folder, "correlation_plot.png")
    try:
        klib.corr_plot(df_cleaned[numeric_cols], figsize=(10, 8))
        plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 상관관계 시각화가 '{corr_plot_path}'로 저장되었습니다!")
    except Exception as e:
        print(f"❌ 상관관계 시각화 오류: {e}")
    
    # 인터랙티브 상관관계 플롯
    try:
        klib.corr_interactive_plot(df_cleaned[numeric_cols])
        print("✅ 인터랙티브 상관관계 플롯이 생성되었습니다!")
    except Exception as e:
        print(f"❌ 인터랙티브 상관관계 플롯 오류: {e}")

# ============================================================================
# 4. 분포 분석 (Klib의 고급 기능)
# ============================================================================
print("\n" + "="*60)
print("4️⃣ KLIB 분포 분석")
print("="*60)

print("📈 분포 분석:")

# 각 수치형 컬럼의 분포 시각화
for col in numeric_cols:
    if col != 'survived':  # survived는 제외
        dist_plot_path = os.path.join(klib_folder, f"distribution_{col}.png")
        try:
            klib.dist_plot(df_cleaned[col], figsize=(10, 6))
            plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ {col} 분포 시각화가 '{dist_plot_path}'로 저장되었습니다!")
        except Exception as e:
            print(f"❌ {col} 분포 시각화 오류: {e}")

# ============================================================================
# 5. 범주형 데이터 분석 (Klib의 고급 기능)
# ============================================================================
print("\n" + "="*60)
print("5️⃣ KLIB 범주형 데이터 분석")
print("="*60)

print("📊 범주형 데이터 분석:")

# 범주형 컬럼 선택
categorical_cols = df_cleaned.select_dtypes(include=['category', 'object']).columns

for col in categorical_cols:
    if col not in ['name', 'ticket', 'cabin', 'boat', 'home.dest']:
        cat_plot_path = os.path.join(klib_folder, f"categorical_{col}.png")
        try:
            klib.cat_plot(df_cleaned, col, figsize=(10, 6))
            plt.savefig(cat_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ {col} 범주형 시각화가 '{cat_plot_path}'로 저장되었습니다!")
        except Exception as e:
            print(f"❌ {col} 범주형 시각화 오류: {e}")

# ============================================================================
# 6. 데이터 품질 분석 (Klib의 고급 기능)
# ============================================================================
print("\n" + "="*60)
print("6️⃣ KLIB 데이터 품질 분석")
print("="*60)

print("🔍 데이터 품질 분석:")

# 데이터 품질 점수
try:
    quality_score = klib.data_cleaning(df_cleaned)
    print(f"  - 데이터 품질 점수: {quality_score:.2f}")
except Exception as e:
    print(f"❌ 데이터 품질 분석 오류: {e}")

# ============================================================================
# 7. 통계적 요약 (Klib의 고급 기능)
# ============================================================================
print("\n" + "="*60)
print("7️⃣ KLIB 통계적 요약")
print("="*60)

print("📋 통계적 요약:")

# 상세한 기술 통계
try:
    detailed_stats = klib.describe(df_cleaned)
    print("  - 상세 기술 통계 생성 완료")
except Exception as e:
    print(f"❌ 상세 기술 통계 오류: {e}")

# ============================================================================
# 8. 엑셀 파일로 저장
# ============================================================================
print("\n" + "="*60)
print("8️⃣ 분석 결과 엑셀 저장")
print("="*60)

print("💾 분석 결과를 엑셀로 저장 중...")

# 엑셀 파일 생성
excel_path = os.path.join(klib_folder, "titanic_klib_advanced_analysis.xlsx")

with pd.ExcelWriter(excel_path) as writer:
    # 원본 데이터
    df_cleaned.to_excel(writer, sheet_name='원본_데이터', index=False)
    
    # 데이터 정보
    info_df = pd.DataFrame({
        '컬럼명': df_cleaned.columns,
        '데이터타입': df_cleaned.dtypes.astype(str),
        '결측치수': df_cleaned.isnull().sum(),
        '결측치비율(%)': (df_cleaned.isnull().sum() / len(df_cleaned)) * 100,
        '고유값수': [df_cleaned[col].nunique() for col in df_cleaned.columns]
    })
    info_df.to_excel(writer, sheet_name='데이터_정보', index=False)
    
    # 수치형 컬럼 통계
    if len(numeric_cols) > 0:
        numeric_stats = df_cleaned[numeric_cols].describe()
        numeric_stats.to_excel(writer, sheet_name='수치형_통계')
    
    # 범주형 컬럼 통계
    categorical_stats = []
    for col in categorical_cols:
        if col not in ['name', 'ticket', 'cabin', 'boat', 'home.dest']:
            value_counts = df_cleaned[col].value_counts()
            categorical_stats.append({
                '컬럼명': col,
                '고유값수': len(value_counts),
                '최빈값': value_counts.index[0] if len(value_counts) > 0 else None,
                '최빈값빈도': value_counts.iloc[0] if len(value_counts) > 0 else 0
            })
    
    if categorical_stats:
        cat_stats_df = pd.DataFrame(categorical_stats)
        cat_stats_df.to_excel(writer, sheet_name='범주형_통계', index=False)
    
    # 생존률 분석
    survived_numeric = pd.to_numeric(df_cleaned['survived'], errors='coerce')
    survival_analysis = []
    
    # 전체 생존률
    survival_analysis.append({
        '분류': '전체',
        '생존률': survived_numeric.mean(),
        '생존자수': survived_numeric.sum(),
        '전체수': len(df_cleaned)
    })
    
    # 성별 생존률
    if 'sex' in df_cleaned.columns:
        for sex in df_cleaned['sex'].unique():
            sex_df = df_cleaned[df_cleaned['sex'] == sex]
            sex_survived = pd.to_numeric(sex_df['survived'], errors='coerce')
            survival_analysis.append({
                '분류': f'성별_{sex}',
                '생존률': sex_survived.mean(),
                '생존자수': sex_survived.sum(),
                '전체수': len(sex_df)
            })
    
    # 클래스별 생존률
    if 'pclass' in df_cleaned.columns:
        for pclass in df_cleaned['pclass'].unique():
            class_df = df_cleaned[df_cleaned['pclass'] == pclass]
            class_survived = pd.to_numeric(class_df['survived'], errors='coerce')
            survival_analysis.append({
                '분류': f'클래스_{pclass}등급',
                '생존률': class_survived.mean(),
                '생존자수': class_survived.sum(),
                '전체수': len(class_df)
            })
    
    survival_df = pd.DataFrame(survival_analysis)
    survival_df.to_excel(writer, sheet_name='생존률_분석', index=False)

print(f"✅ 고급 분석 결과가 '{excel_path}'로 저장되었습니다!")

# ============================================================================
# 9. 생성된 파일들 확인
# ============================================================================
print("\n" + "="*60)
print("📁 생성된 파일들 확인")
print("="*60)

if os.path.exists(klib_folder):
    files = []
    for root, dirs, filenames in os.walk(klib_folder):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            file_size = os.path.getsize(file_path) / 1024  # KB
            files.append(f"  - {filename} ({file_size:.0f}KB)")
    
    print(f"✅ klib_advanced/ 폴더: {len(files)}개 파일")
    for file_info in files:
        print(file_info)

print("\n" + "="*60)
print("🎉 Klib 고급 분석이 완료되었습니다!")
print("💡 Klib의 주요 기능:")
print("   • 데이터 클리닝: 컬럼명 정리, 데이터타입 변환, 중복 처리")
print("   • 결측치 분석: 시각화 및 처리 방법 제안")
print("   • 상관관계 분석: 매트릭스, 시각화, 인터랙티브 플롯")
print("   • 분포 분석: 각 변수의 분포 시각화")
print("   • 범주형 분석: 범주형 변수의 분포 시각화")
print("   • 데이터 품질: 품질 점수 및 개선 제안")
print("   • 통계적 요약: 상세한 기술 통계")
print("="*60) 