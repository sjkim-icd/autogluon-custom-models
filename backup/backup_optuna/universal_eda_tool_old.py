import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
import glob
import base64
import argparse
import sys
from pathlib import Path
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

def load_data(data_path, file_type=None, max_rows=None, target_var=None):
    """
    데이터를 로드하고 필요시 타겟 비율을 유지하며 샘플링
    """
    print(f"📁 데이터 로딩 중: {data_path}")
    
    # 파일 타입 자동 감지
    if file_type is None or file_type == 'auto':
        if data_path.endswith('.csv'):
            file_type = 'csv'
        elif data_path.endswith(('.xlsx', '.xls')):
            file_type = 'excel'
        elif data_path.endswith('.parquet'):
            file_type = 'parquet'
        else:
            file_type = 'csv'  # 기본값
    
    # 데이터 로드
    try:
        if file_type == 'csv':
            df = pd.read_csv(data_path)
        elif file_type == 'excel':
            df = pd.read_excel(data_path)
        elif file_type == 'parquet':
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"지원하지 않는 파일 타입: {file_type}")
        
        print(f"✅ 데이터 로드 완료: {df.shape}")
        print(f"📋 컬럼: {list(df.columns)}")
        
        # 타겟 비율 유지 샘플링
        if max_rows and len(df) > max_rows:
            # 원본 타겟 비율 계산
            if target_var and target_var in df.columns:
                original_counts = df[target_var].value_counts()
                original_ratios = original_counts / len(df)
                print(f"📊 원본 데이터 타겟 비율:")
                for class_val, ratio in original_ratios.items():
                    print(f"   Class {class_val}: {original_counts[class_val]:,}개 ({ratio:.3%})")
            
            df = stratified_sample(df, max_rows, target_var)
            
            # 샘플링 후 타겟 비율 계산
            if target_var and target_var in df.columns:
                sampled_counts = df[target_var].value_counts()
                sampled_ratios = sampled_counts / len(df)
                print(f"📊 샘플링 후 타겟 비율:")
                for class_val, ratio in sampled_ratios.items():
                    print(f"   Class {class_val}: {sampled_counts[class_val]:,}개 ({ratio:.3%})")
            
            print(f"📊 타겟 비율 유지 샘플링 완료: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"❌ 데이터 로드 오류: {e}")
        return None

def stratified_sample(df, n_samples, target_var=None):
    """
    타겟 변수 비율을 유지하며 샘플링
    """
    if target_var is None or target_var not in df.columns:
        # 타겟 변수가 없으면 단순 랜덤 샘플링
        return df.sample(n=min(n_samples, len(df)), random_state=42)
    
    # 타겟 변수 비율 계산
    target_counts = df[target_var].value_counts()
    target_ratios = target_counts / len(df)
    
    # 각 클래스별 샘플 수 계산 (반올림 사용)
    sample_counts = {}
    for class_val, ratio in target_ratios.items():
        sample_counts[class_val] = round(n_samples * ratio)
    
    # 비율 조정 (총합이 n_samples가 되도록)
    total_sampled = sum(sample_counts.values())
    if total_sampled != n_samples:
        # 가장 큰 클래스에서 조정
        largest_class = max(sample_counts, key=sample_counts.get)
        sample_counts[largest_class] += (n_samples - total_sampled)
    
    # 각 클래스별로 샘플링
    sampled_dfs = []
    for class_val, n_sample in sample_counts.items():
        if n_sample > 0:
            class_df = df[df[target_var] == class_val]
            if len(class_df) >= n_sample:
                sampled_class = class_df.sample(n=n_sample, random_state=42)
            else:
                # 해당 클래스의 샘플이 부족하면 전체 사용
                sampled_class = class_df
            sampled_dfs.append(sampled_class)
    
    # 결과 합치기
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # 최종 샘플 수 조정
    if len(result_df) > n_samples:
        result_df = result_df.sample(n=n_samples, random_state=42)
    
    return result_df

def create_folder_structure(dataset_name, output_dir="EDA"):
    """
    EDA 결과를 저장할 폴더 구조를 생성합니다.
    
    Args:
        dataset_name (str): 데이터셋 이름
        output_dir (str): 출력 디렉토리
    
    Returns:
        dict: 패키지별 폴더 경로
    """
    print(f"\n{'='*60}")
    print(f"📁 EDA 결과 폴더 구조 생성")
    print(f"{'='*60}")
    
    # 메인 EDA 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 '{output_dir}' 폴더를 생성했습니다.")
    
    # 데이터셋 폴더 생성
    dataset_folder = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        print(f"📁 '{dataset_folder}' 폴더를 생성했습니다.")
    
    # 각 패키지별 하위 폴더 생성
    packages = {
        "ydata_profiling": os.path.join(dataset_folder, "ydata_profiling"),
        "sweetviz": os.path.join(dataset_folder, "sweetviz"),
        "autoviz": os.path.join(dataset_folder, "autoviz"),
        "klib": os.path.join(dataset_folder, "klib"),
        "dtale": os.path.join(dataset_folder, "dtale")
    }
    
    for package_name, package_path in packages.items():
        if not os.path.exists(package_path):
            os.makedirs(package_path)
            print(f"📁 '{package_path}' 폴더를 생성했습니다.")
    
    print("✅ 폴더 구조 생성 완료!")
    return packages

def run_ydata_profiling(df, packages, dataset_name):
    """YData-Profiling 분석을 실행합니다."""
    print(f"\n{'='*60}")
    print("1️⃣ YDATA-PROFILING 적용")
    print(f"{'='*60}")
    
    try:
        import ydata_profiling as yp
        
        print("📈 ydata-profiling으로 상세 분석 리포트 생성 중...")
        profile = yp.ProfileReport(df, title=f"{dataset_name} Dataset Analysis")
        profile_path = os.path.join(packages["ydata_profiling"], f"{dataset_name}_ydata_profiling.html")
        profile.to_file(profile_path)
        print(f"✅ HTML 리포트가 '{profile_path}'로 저장되었습니다!")
        
    except ImportError:
        print("❌ ydata-profiling이 설치되지 않았습니다. pip install ydata-profiling")
    except Exception as e:
        print(f"❌ ydata-profiling 오류: {e}")

def run_sweetviz(df, packages, dataset_name, target_var):
    """Sweetviz 분석을 실행합니다."""
    print(f"\n{'='*60}")
    print("2️⃣ SWEETVIZ 적용")
    print(f"{'='*60}")
    
    try:
        import sweetviz as sv
        
        print("🍯 Sweetviz로 데이터 분석 리포트 생성 중...")
        
        # 타겟 변수를 숫자로 변환
        df_for_sweetviz = df.copy()
        
        # 타겟 변수가 있고 유효한 경우에만 처리
        if target_var and target_var in df_for_sweetviz.columns:
            try:
                df_for_sweetviz[target_var] = pd.to_numeric(df_for_sweetviz[target_var], errors='coerce')
                # Sweetviz 리포트 생성 (타겟 변수 포함)
                report = sv.analyze([df_for_sweetviz, f"{dataset_name} Dataset"], target_feat=target_var)
                print(f"✅ 타겟 변수 '{target_var}'를 포함한 Sweetviz 분석 수행")
            except Exception as e:
                print(f"⚠️ 타겟 변수 '{target_var}' 처리 중 오류: {e}")
                # 타겟 변수 처리 실패 시 전체 분석으로 대체
                report = sv.analyze([df_for_sweetviz, f"{dataset_name} Dataset"])
                print(f"✅ 전체 데이터 분석으로 대체")
        else:
            # 타겟 변수가 없거나 유효하지 않은 경우 전체 분석
            if target_var:
                print(f"⚠️ 타겟 변수 '{target_var}'를 찾을 수 없습니다. 전체 데이터 분석을 수행합니다.")
            else:
                print("ℹ️ 타겟 변수가 지정되지 않았습니다. 전체 데이터 분석을 수행합니다.")
            report = sv.analyze([df_for_sweetviz, f"{dataset_name} Dataset"])
            
        sweetviz_path = os.path.join(packages["sweetviz"], f"{dataset_name}_sweetviz_report.html")
        report.show_html(sweetviz_path)
        print(f"✅ Sweetviz HTML 리포트가 '{sweetviz_path}'로 저장되었습니다!")
        
    except ImportError:
        print("❌ sweetviz가 설치되지 않았습니다. pip install sweetviz")
    except Exception as e:
        print(f"❌ Sweetviz 오류: {e}")

def run_autoviz(df, packages, dataset_name, target_var, max_rows=1000, max_cols=20):
    """AutoViz 분석을 실행합니다."""
    print(f"\n{'='*60}")
    print("3️⃣ AUTOVIZ 적용")
    print(f"{'='*60}")
    
    try:
        from autoviz.AutoViz_Class import AutoViz_Class
        
        print("🎨 AutoViz로 자동 시각화 생성 중...")
        
        # AutoViz 실행 전 한글 폰트 재설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        AV = AutoViz_Class()
        
        # 타겟 변수 유효성 검사 및 분기처리
        dep_var = ""
        if target_var and target_var in df.columns:
            try:
                # 타겟 변수가 숫자형인지 확인
                target_numeric = pd.to_numeric(df[target_var], errors='coerce')
                if not target_numeric.isna().all():  # 모든 값이 NaN이 아닌 경우
                    dep_var = target_var
                    print(f"✅ 타겟 변수 '{target_var}'를 사용한 AutoViz 분석 수행")
                else:
                    print(f"⚠️ 타겟 변수 '{target_var}'가 모두 NaN입니다. 전체 데이터 분석을 수행합니다.")
            except Exception as e:
                print(f"⚠️ 타겟 변수 '{target_var}' 처리 중 오류: {e}")
        else:
            if target_var:
                print(f"⚠️ 타겟 변수 '{target_var}'를 찾을 수 없습니다. 전체 데이터 분석을 수행합니다.")
            else:
                print("ℹ️ 타겟 변수가 지정되지 않았습니다. 전체 데이터 분석을 수행합니다.")
        
        # AutoViz 실행 (dep_var가 빈 문자열이면 전체 분석)
        df_viz = AV.AutoViz(
            filename="",  # 파일명이 없으면 데이터프레임 사용
            dfte=df,     # 데이터프레임
            depVar=dep_var,  # 타겟 변수 (빈 문자열이면 전체 분석)
            max_rows_analyzed=max_rows,  # 분석할 최대 행 수
            max_cols_analyzed=max_cols,    # 분석할 최대 컬럼 수
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
            if col != target_var:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.subplot(1, 2, 1)
                    plt.hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.title(f'{col} 분포')
                    plt.xlabel(col)
                    plt.ylabel('빈도')
                    
                    if target_var and target_var in df.columns:
                        plt.subplot(1, 2, 2)
                        target_numeric = pd.to_numeric(df[target_var], errors='coerce')
                        plt.scatter(df[col], target_numeric, alpha=0.5)
                        plt.title(f'{col} vs {target_var}')
                        plt.xlabel(col)
                        plt.ylabel(target_var)
                    
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
            if col not in ['name', 'ticket', 'cabin', 'boat', 'home.dest']:  # 텍스트 컬럼 제외
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
                    
                    # 타겟 변수와의 관계
                    if target_var and target_var in df.columns:
                        plt.subplot(1, 2, 2)
                        target_by_cat = df.groupby(col)[target_var].apply(
                            lambda x: pd.to_numeric(x, errors='coerce').mean()
                        ).sort_values(ascending=False).head(10)
                        plt.bar(range(len(target_by_cat)), target_by_cat.values)
                        plt.title(f'{col}별 {target_var}')
                        plt.xlabel('값')
                        plt.ylabel(target_var)
                        plt.xticks(range(len(target_by_cat)), target_by_cat.index, rotation=45)
                    
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
                sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5)
                plt.title('수치형 변수 상관관계')
                plt.tight_layout()
                plt.savefig(os.path.join(packages["autoviz"], 'autoviz_correlation_heatmap.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ 상관관계 히트맵 저장")
            except Exception as e:
                print(f"❌ 상관관계 히트맵 오류: {e}")
        
        # 고급 시각화 추가
        print("🎨 고급 AutoViz 시각화 생성 중...")
        
        # 박스플롯 (이상치 탐지)
        try:
            plt.figure(figsize=(15, 10))
            numeric_cols_for_box = [col for col in numeric_cols if col != target_var]
            
            # 더 많은 변수 포함 (신용카드 데이터의 경우 V1~V28까지 있으므로)
            max_box_vars = min(12, len(numeric_cols_for_box))  # 최대 12개까지
            for i, col in enumerate(numeric_cols_for_box[:max_box_vars], 1):
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
        
        # 바이올린 플롯
        try:
            plt.figure(figsize=(15, 10))
            
            # 더 많은 변수 포함
            cat_vars = [col for col in df.columns if df[col].dtype in ['object', 'category']][:6]
            num_vars = [col for col in numeric_cols if col != target_var][:6]
            
            for i, (cat_var, num_var) in enumerate(zip(cat_vars, num_vars), 1):
                plt.subplot(2, 3, i)
                if target_var and target_var in df.columns:
                    sns.violinplot(data=df, x=cat_var, y=num_var, hue=target_var)
                else:
                    sns.violinplot(data=df, x=cat_var, y=num_var)
                plt.title(f'{cat_var}별 {num_var} 분포')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(packages["autoviz"], 'autoviz_violin_plots.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 바이올린 플롯 저장")
        except Exception as e:
            print(f"❌ 바이올린 플롯 오류: {e}")
        
        # 데이터 타입에 따른 적절한 시각화 선택
        try:
            data_size = len(df)
            print(f"📊 데이터 타입에 따른 시각화 선택 (데이터 크기: {data_size:,}개)")
            
            # 데이터 타입 확인
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            numeric_cols_plot = [col for col in numeric_cols if col != target_var][:6]
            
            print(f"📋 범주형 변수: {len(categorical_cols)}개, 수치형 변수: {len(numeric_cols_plot)}개")
            
            if len(categorical_cols) > 0:
                # 범주형 변수가 있는 경우: Swarm Plot 적합
                print("🔄 범주형 변수 기반 Swarm Plots 생성 중...")
                plt.figure(figsize=(20, 12))
                
                # 범주형 변수와 수치형 변수 조합
                cat_vars = list(categorical_cols)[:3]  # 상위 3개 범주형 변수
                num_vars = numeric_cols_plot[:3]  # 상위 3개 수치형 변수
                
                combinations = []
                for i, (cat_var, num_var) in enumerate(zip(cat_vars, num_vars)):
                    combinations.append((cat_var, num_var, f'{cat_var}별 {num_var} 분포'))
                
                # 6개까지 채우기
                while len(combinations) < 6 and len(cat_vars) > 1 and len(num_vars) > 1:
                    for i in range(len(cat_vars) - 1):
                        for j in range(len(num_vars) - 1):
                            if len(combinations) >= 6:
                                break
                            cat_var = cat_vars[i + 1]
                            num_var = num_vars[j + 1]
                            combinations.append((cat_var, num_var, f'{cat_var}별 {num_var} 분포'))
                
                for i, (cat_var, num_var, title) in enumerate(combinations, 1):
                    plt.subplot(2, 3, i)
                    if target_var and target_var in df.columns:
                        sns.swarmplot(data=df, x=cat_var, y=num_var, hue=target_var, 
                                     palette=['blue', 'red'], alpha=0.7)
                    else:
                        sns.swarmplot(data=df, x=cat_var, y=num_var, alpha=0.7)
                    plt.title(f'Autoviz Swarm Plot: {title}')
                    plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig(os.path.join(packages["autoviz"], 'autoviz_swarm_plots.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ 범주형 기반 Swarm Plots 저장")
                
            else:
                # 모든 변수가 수치형인 경우: 산점도 사용
                print("📊 수치형 변수 기반 산점도 생성 중...")
                plt.figure(figsize=(20, 12))
                
                # 수치형 변수 조합
                combinations = []
                for i, var1 in enumerate(numeric_cols_plot):
                    for j, var2 in enumerate(numeric_cols_plot[i+1:], i+1):
                        if len(combinations) >= 6:
                            break
                        combinations.append((var1, var2, f'{var1} vs {var2}'))
                
                # 데이터 크기에 따른 샘플링
                if data_size > 5000:
                    sample_data = df.sample(n=5000, random_state=42)
                    print(f"📊 데이터 샘플링: {data_size:,}개 → 5,000개")
                else:
                    sample_data = df
                
                for i, (var1, var2, title) in enumerate(combinations, 1):
                    plt.subplot(2, 3, i)
                    if target_var and target_var in df.columns:
                        plt.scatter(sample_data[var1], sample_data[var2], 
                                   c=sample_data[target_var], cmap='viridis', alpha=0.6, s=10)
                    else:
                        plt.scatter(sample_data[var1], sample_data[var2], alpha=0.6, s=10)
                    plt.title(f'Autoviz Scatter Plot: {title}')
                    plt.xlabel(var1)
                    plt.ylabel(var2)
                
                plt.tight_layout()
                plt.savefig(os.path.join(packages["autoviz"], 'autoviz_scatter_plots.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ 수치형 기반 산점도 저장")
                
        except Exception as e:
            print(f"❌ 데이터 타입 기반 시각화 오류: {e}")
        
        # 고급 Violin Plots (타겟변수 중심)
        if target_var and target_var in df.columns:
            try:
                print("🎻 고급 Violin Plots 생성 중...")
                plt.figure(figsize=(18, 6))
                
                # 주요 범주형 변수들과 수치형 변수 조합 (데이터에 맞게 동적 선택)
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                
                # 사용 가능한 범주형 변수들 (타겟변수 제외)
                available_cat_cols = [col for col in categorical_cols if col != target_var and df[col].nunique() <= 10]
                # 사용 가능한 수치형 변수들 (타겟변수 제외)
                available_num_cols = [col for col in numeric_cols if col != target_var]
                
                # 조합 생성 (최대 3개)
                advanced_combinations = []
                for i, cat_var in enumerate(available_cat_cols[:3]):
                    if i < len(available_num_cols):
                        num_var = available_num_cols[i]
                        title = f'{cat_var}별 {num_var} 분포'
                        advanced_combinations.append((cat_var, num_var, title))
                
                if len(advanced_combinations) == 0:
                    # 범주형 변수가 없으면 수치형 변수들만 사용
                    for i in range(min(3, len(available_num_cols))):
                        num_var1 = available_num_cols[i]
                        num_var2 = available_num_cols[(i+1) % len(available_num_cols)]
                        title = f'{num_var1} vs {num_var2}'
                        advanced_combinations.append((num_var1, num_var2, title))
                
                for i, (var1, var2, title) in enumerate(advanced_combinations, 1):
                    plt.subplot(1, 3, i)
                    try:
                        if var1 in categorical_cols:
                            # 범주형 vs 수치형
                            sns.violinplot(data=df, x=var1, y=var2, hue=target_var, 
                                          palette=['blue', 'red'], split=True)
                        else:
                            # 수치형 vs 수치형 (구간으로 나누어)
                            df_temp = df.copy()
                            df_temp[f'{var1}_group'] = pd.qcut(df_temp[var1], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                            sns.violinplot(data=df_temp, x=f'{var1}_group', y=var2, hue=target_var, 
                                          palette=['blue', 'red'], split=True)
                        plt.title(f'Autoviz Violin Plot: {title}')
                        plt.xticks(rotation=45)
                    except Exception as e:
                        plt.text(0.5, 0.5, f'{var1} vs {var2}\n오류', ha='center', va='center')
                        plt.title(f'Autoviz Violin Plot: {title}')
                
                plt.tight_layout()
                plt.savefig(os.path.join(packages["autoviz"], 'autoviz_advanced_violin_plots.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ 고급 Violin Plots 저장")
            except Exception as e:
                print(f"❌ 고급 Violin Plots 오류: {e}")
        
        # Pair Plot (상관관계 시각화)
        try:
            print("🔗 Pair Plot 생성 중...")
            # 주요 수치형 변수들 선택 (더 많은 변수 포함)
            # 신용카드 데이터의 경우 V1~V28까지 있으므로 더 많은 변수 포함
            if len(numeric_cols) > 10:
                # 변수가 많으면 상위 10개 선택 (Time, Amount, V1~V8)
                numeric_for_pair = [col for col in numeric_cols if col != target_var][:10]
            else:
                # 변수가 적으면 최대 8개까지 선택
                numeric_for_pair = [col for col in numeric_cols if col != target_var][:8]
            
            if target_var and target_var in df.columns:
                # 타겟변수가 있을 때
                pair_data = df[numeric_for_pair + [target_var]].dropna()
                sns.pairplot(pair_data, hue=target_var, palette=['blue', 'red'])
            else:
                # 타겟변수가 없을 때
                pair_data = df[numeric_for_pair].dropna()
                sns.pairplot(pair_data)
            
            plt.savefig(os.path.join(packages["autoviz"], 'autoviz_pair_plot.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Pair Plot 저장")
        except Exception as e:
            print(f"❌ Pair Plot 오류: {e}")
        
        # 타겟변수 종합 분석 (타겟변수가 있을 때만)
        if target_var and target_var in df.columns:
            try:
                print(f"📊 {target_var} 변수 종합 분석 생성 중...")
                plt.figure(figsize=(15, 10))
                
                # 범주형 변수별 타겟 분석 (상위 6개)
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                cat_cols_for_analysis = [col for col in categorical_cols if col != target_var and df[col].nunique() <= 10][:6]
                
                # 범주형 변수가 부족하면 수치형 변수를 구간으로 나누어 사용
                if len(cat_cols_for_analysis) < 6:
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    num_cols_for_analysis = [col for col in numeric_cols if col != target_var][:6-len(cat_cols_for_analysis)]
                    cat_cols_for_analysis.extend(num_cols_for_analysis)
                
                # 모든 변수를 구간으로 나누어 분석 (수치형 변수는 구간화, 범주형 변수는 그대로)
                all_cols_for_analysis = []
                
                # 범주형 변수는 그대로 사용
                for cat_col in cat_cols_for_analysis:
                    all_cols_for_analysis.append(('categorical', cat_col))
                
                # 수치형 변수는 구간으로 나누어 사용
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                num_cols_for_analysis = [col for col in numeric_cols if col != target_var][:6-len(cat_cols_for_analysis)]
                
                for num_col in num_cols_for_analysis:
                    all_cols_for_analysis.append(('numeric', num_col))
                
                # 6개까지 제한
                all_cols_for_analysis = all_cols_for_analysis[:6]
                
                for i, (col_type, col_name) in enumerate(all_cols_for_analysis):
                    plt.subplot(2, 3, i+1)
                    try:
                        if col_type == 'categorical':
                            # 범주형 변수는 그대로 사용
                            target_by_col = df.groupby(col_name)[target_var].apply(
                                lambda x: pd.to_numeric(x, errors='coerce').mean()
                            )
                            title = f'{col_name}별 {target_var} 비율'
                        else:
                            # 수치형 변수는 구간으로 나누어 분석
                            df_temp = df.copy()
                            df_temp[f'{col_name}_group'] = pd.qcut(df_temp[col_name], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                            target_by_col = df_temp.groupby(f'{col_name}_group')[target_var].apply(
                                lambda x: pd.to_numeric(x, errors='coerce').mean()
                            )
                            title = f'{col_name} 구간별 {target_var} 비율'
                        
                        if len(target_by_col) > 0:  # 데이터가 있는 경우만
                            colors = plt.cm.Set3(np.linspace(0, 1, len(target_by_col)))
                            plt.bar(target_by_col.index, target_by_col.values, color=colors)
                            plt.title(title)
                            plt.ylabel(f'{target_var} 비율')
                            plt.ylim(0, 1)
                            plt.xticks(rotation=45)
                        else:
                            plt.text(0.5, 0.5, f'{col_name} 데이터 없음', ha='center', va='center')
                            plt.title(title)
                    except Exception as e:
                        plt.text(0.5, 0.5, f'{col_name} 오류', ha='center', va='center')
                        plt.title(title)
                
                plt.tight_layout()
                plt.savefig(os.path.join(packages["autoviz"], f'autoviz_{target_var}_analysis.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ {target_var} 변수 종합 분석 저장")
            except Exception as e:
                print(f"❌ {target_var} 변수 분석 오류: {e}")
        
        # 상세 상관관계 히트맵
        try:
            print("📈 상세 상관관계 히트맵 생성 중...")
            plt.figure(figsize=(12, 10))
            
            # 수치형 변수들 + 타겟변수
            if target_var and target_var in df.columns:
                corr_cols = [col for col in numeric_cols if col != target_var] + [target_var]
                df_corr = df[corr_cols].copy()
                df_corr[target_var] = pd.to_numeric(df_corr[target_var], errors='coerce')
            else:
                df_corr = df[numeric_cols].copy()
            
            correlation_matrix = df_corr.corr()
            
            # 마스크 생성 (상삼각형만 표시)
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            # 숫자 제거하여 시각적 가독성 향상
            sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5)
            plt.title('상세 상관관계 히트맵')
            plt.tight_layout()
            plt.savefig(os.path.join(packages["autoviz"], 'autoviz_detailed_correlation.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 상세 상관관계 히트맵 저장")
        except Exception as e:
            print(f"❌ 상세 상관관계 히트맵 오류: {e}")
        
        # 스트리프 플롯 (Strip Plot)
        try:
            print("📊 스트리프 플롯 생성 중...")
            plt.figure(figsize=(15, 10))
            
            # 가족 규모 계산
            df_for_strip = df.copy()
            df_for_strip['family_size'] = df_for_strip['sibsp'] + df_for_strip['parch'] + 1
            
            # 주요 변수 조합들
            strip_combinations = [
                ('sex', 'age', '성별 나이 분포'),
                ('pclass', 'fare', '클래스별 요금 분포'),
                ('embarked', 'age', '승선항별 나이 분포'),
                ('family_size', 'age', '가족 규모별 나이 분포'),
                ('sex', 'fare', '성별 요금 분포'),
                ('pclass', 'age', '클래스별 나이 분포')
            ]
            
            for i, (cat_var, num_var, title) in enumerate(strip_combinations, 1):
                plt.subplot(2, 3, i)
                if target_var and target_var in df.columns:
                    sns.stripplot(data=df_for_strip, x=cat_var, y=num_var, hue=target_var, 
                                 jitter=0.3, alpha=0.6)
                else:
                    sns.stripplot(data=df_for_strip, x=cat_var, y=num_var, jitter=0.3, alpha=0.6)
                plt.title(f'Strip Plot: {title}')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(packages["autoviz"], 'autoviz_strip_plots.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 스트리프 플롯 저장")
        except Exception as e:
            print(f"❌ 스트리프 플롯 오류: {e}")
        
        # 조인트 플롯 (Joint Plot)
        try:
            print("🔗 조인트 플롯 생성 중...")
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # 주요 변수 조합들
            joint_combinations = [
                ('age', 'fare', '나이 vs 요금'),
                ('age', 'pclass', '나이 vs 클래스'),
                ('fare', 'pclass', '요금 vs 클래스'),
                ('family_size', 'age', '가족 규모 vs 나이'),
                ('family_size', 'fare', '가족 규모 vs 요금'),
                ('sibsp', 'parch', '형제자매 vs 부모자식')
            ]
            
            for i, (var1, var2, title) in enumerate(joint_combinations):
                row, col = i // 3, i % 3
                
                if target_var and target_var in df.columns:
                    sns.jointplot(data=df, x=var1, y=var2, hue=target_var, 
                                  kind='scatter', alpha=0.6, ax=axes[row, col])
                else:
                    sns.jointplot(data=df, x=var1, y=var2, 
                                  kind='scatter', alpha=0.6, ax=axes[row, col])
                axes[row, col].set_title(f'Joint Plot: {title}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(packages["autoviz"], 'autoviz_joint_plots.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 조인트 플롯 저장")
        except Exception as e:
            print(f"❌ 조인트 플롯 오류: {e}")
        
        print("🎨 고급 AutoViz 시각화 완료!")
        
    except ImportError:
        print("❌ autoviz가 설치되지 않았습니다. pip install autoviz")
    except Exception as e:
        print(f"❌ AutoViz 오류: {e}")
        print("🔧 AutoViz 대신 기본 matplotlib/seaborn 시각화를 생성합니다...")
        
        # AutoViz 실패 시 기본 시각화 생성
        try:
            print("📊 기본 시각화 생성 중...")
            
            # 기본 히스토그램들
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:5]:  # 최대 5개
                try:
                    plt.figure(figsize=(8, 6))
                    plt.hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.title(f'{col} 분포')
                    plt.xlabel(col)
                    plt.ylabel('빈도')
                    plt.tight_layout()
                    plt.savefig(os.path.join(packages["autoviz"], f'basic_{col}_histogram.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✅ 기본 {col} 히스토그램 저장")
                except Exception as e:
                    print(f"❌ {col} 기본 시각화 오류: {e}")
            
            # 기본 상관관계 히트맵
            if len(numeric_cols) > 1:
                try:
                    plt.figure(figsize=(10, 8))
                    correlation_matrix = df[numeric_cols].corr()
                    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                               square=True, linewidths=0.5)
                    plt.title('수치형 변수 상관관계')
                    plt.tight_layout()
                    plt.savefig(os.path.join(packages["autoviz"], 'basic_correlation_heatmap.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print("✅ 기본 상관관계 히트맵 저장")
                except Exception as e:
                    print(f"❌ 기본 상관관계 히트맵 오류: {e}")
            
            print("✅ 기본 시각화 생성 완료!")
            
        except Exception as e:
            print(f"❌ 기본 시각화도 실패: {e}")

def run_klib(df, packages, dataset_name):
    """Klib 분석을 실행합니다."""
    print(f"\n{'='*60}")
    print("4️⃣ KLIB 적용")
    print(f"{'='*60}")
    
    try:
        import klib
        
        print("🔧 Klib로 데이터 클리닝 및 분석 중...")
        
        # 데이터 정보
        print("📋 데이터 정보:")
        print(df.info())
        
        # 결측치 분석
        print("🔍 결측치 분석:")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            '결측치 개수': missing_data,
            '결측치 비율(%)': missing_percent
        }).sort_values('결측치 비율(%)', ascending=False)
        
        print("결측치가 있는 컬럼:")
        print(missing_df[missing_df['결측치 개수'] > 0])
        
        # 결측치 시각화
        try:
            klib.missingval_plot(df)
            plt.savefig(os.path.join(packages["klib"], 'missing_values_plot.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 결측치 시각화 저장: {packages['klib']}/missing_values_plot.png")
        except Exception as e:
            print(f"❌ 결측치 시각화 오류: {e}")
        
        # 상관관계 분석
        print("📊 상관관계 분석:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            print("수치형 변수 간 상관관계:")
            print(correlation_matrix.round(3))
            
            # 상관관계 히트맵
            try:
                klib.corr_plot(df)
                plt.savefig(os.path.join(packages["klib"], 'correlation_plot.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ 상관관계 히트맵 저장: {packages['klib']}/correlation_plot.png")
            except Exception as e:
                print(f"❌ 상관관계 히트맵 오류: {e}")
        
        # 분포 시각화
        try:
            klib.dist_plot(df)
            plt.savefig(os.path.join(packages["klib"], 'dist_plot.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 분포 시각화 저장: {packages['klib']}/dist_plot.png")
        except Exception as e:
            print(f"❌ 분포 시각화 오류: {e}")
        
        # 범주형 시각화
        try:
            klib.cat_plot(df)
            plt.savefig(os.path.join(packages["klib"], 'cat_plot.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 범주형 시각화 저장: {packages['klib']}/cat_plot.png")
        except Exception as e:
            print(f"❌ 범주형 시각화 오류: {e}")
        
        # 엑셀 파일로 분석 결과 저장
        excel_output_path = os.path.join(packages["klib"], f"{dataset_name}_klib_analysis.xlsx")
        
        try:
            # 기존 파일 삭제 시도
            if os.path.exists(excel_output_path):
                try:
                    os.remove(excel_output_path)
                except PermissionError:
                    # 파일이 열려있으면 타임스탬프 추가
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    excel_output_path = os.path.join(packages["klib"], f"{dataset_name}_klib_analysis_{timestamp}.xlsx")
            
            with pd.ExcelWriter(excel_output_path) as writer:
                # 데이터 정보
                info_data = []
                for col in df.columns:
                    info_data.append({
                        '컬럼명': col,
                        '데이터타입': str(df[col].dtype),
                        '결측치 개수': df[col].isnull().sum(),
                        '결측치 비율(%)': (df[col].isnull().sum() / len(df)) * 100,
                        '고유값 개수': df[col].nunique()
                    })
                pd.DataFrame(info_data).to_excel(writer, sheet_name='데이터_정보', index=False)
                
                # 결측치 분석
                missing_df.to_excel(writer, sheet_name='결측치_분석', index=True)
                
                # 상관관계 분석
                if len(numeric_cols) > 1:
                    correlation_matrix.to_excel(writer, sheet_name='상관관계_분석', index=True)
                
                # 수치형 변수 통계
                if len(numeric_cols) > 0:
                    df[numeric_cols].describe().to_excel(writer, sheet_name='수치형_통계', index=True)
                
                # 범주형 변수 통계
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    cat_stats = []
                    for col in categorical_cols:
                        value_counts = df[col].value_counts()
                        cat_stats.append({
                            '컬럼명': col,
                            '상위값1': value_counts.index[0] if len(value_counts) > 0 else None,
                            '상위값1_빈도': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                            '상위값2': value_counts.index[1] if len(value_counts) > 1 else None,
                            '상위값2_빈도': value_counts.iloc[1] if len(value_counts) > 1 else 0,
                            '고유값_개수': len(value_counts)
                        })
                    pd.DataFrame(cat_stats).to_excel(writer, sheet_name='범주형_통계', index=False)
                
                # 데이터 샘플
                df.head(100).to_excel(writer, sheet_name='데이터_샘플', index=False)
            
            print(f"✅ Klib 분석 결과가 '{excel_output_path}'로 저장되었습니다!")
            
        except Exception as e:
            print(f"❌ 엑셀 저장 오류: {e}")
        
    except ImportError:
        print("❌ klib가 설치되지 않았습니다. pip install klib")
    except Exception as e:
        print(f"❌ Klib 오류: {e}")

def run_dtale(df, packages, dataset_name, port=4000, use_ngrok=False):
    """D-Tale 인터랙티브 인터페이스를 실행합니다."""
    print(f"\n{'='*60}")
    print("5️⃣ D-TALE 적용")
    print(f"{'='*60}")
    
    try:
        import dtale
        
        print("🌐 D-Tale 대화형 인터페이스 시작 중...")
        
        # D-Tale 인스턴스 생성
        d = dtale.show(df, name=f"{dataset_name} Dataset", port=port, host='localhost')
        
        print("✅ D-Tale이 시작되었습니다!")
        print(f"🌐 브라우저에서 다음 URL로 접속하세요: http://localhost:{port}")
        
        # 선택적 ngrok 공개 URL
        if use_ngrok:
            try:
                from pyngrok import ngrok, conf
                authtoken = os.getenv('NGROK_AUTHTOKEN', '')
                if authtoken:
                    try:
                        ngrok.set_auth_token(authtoken)
                    except Exception:
                        conf.get_default().auth_token = authtoken
                public_url = ngrok.connect(port, bind_tls=True)
                print(f"🔗 ngrok 공개 URL: {public_url}")
            except Exception as e:
                print(f"❌ ngrok 설정 실패: {e}")
        
        print("💡 브라우저가 자동으로 열리지 않으면 위 URL을 복사해서 접속하세요!")
        
        # 브라우저 자동 열기
        import webbrowser
        import time
        time.sleep(2)
        try:
            webbrowser.open(f"http://localhost:{port}")
            print("🌐 브라우저를 자동으로 열었습니다!")
        except:
            print("❌ 브라우저 자동 열기 실패. 수동으로 접속하세요.")
        
        return d
        
    except ImportError:
        print("❌ dtale이 설치되지 않았습니다. pip install dtale")
    except Exception as e:
        print(f"❌ D-Tale 오류: {e}")

def create_html_report(packages, dataset_name):
    """AutoViz 이미지들을 HTML 보고서로 생성합니다."""
    print(f"\n{'='*60}")
    print("🖼️ AUTOVIZ 이미지 HTML 보고서 생성")
    print(f"{'='*60}")
    
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
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_name} AutoViz 데이터 분석 보고서</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        .image-section {{
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: #fafafa;
        }}
        .image-title {{
            font-size: 1.5em;
            color: #34495e;
            margin-bottom: 15px;
            font-weight: bold;
            text-align: center;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .file-info {{
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .summary h2 {{
            color: #2c3e50;
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {dataset_name} AutoViz 데이터 분석 보고서</h1>
        
        <div class="summary">
            <h2>📊 분석 개요</h2>
            <p>이 보고서는 AutoViz를 사용하여 {dataset_name} 데이터셋을 분석한 결과입니다. 
            각 이미지는 데이터의 다양한 측면을 시각화하여 보여줍니다.</p>
            <p><strong>생성 시간:</strong> {datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")}</p>
            <p><strong>총 이미지 수:</strong> {len(image_files)}개</p>
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
        html_output_path = os.path.join(packages["autoviz"], f"{dataset_name}_autoviz_analysis_report.html")
        try:
            with open(html_output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ AutoViz HTML 보고서가 '{html_output_path}'로 저장되었습니다!")
            print(f"📊 총 {len(image_files)}개의 이미지가 포함되었습니다.")
            
            # 브라우저에서 열기
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(html_output_path)}")
                print("🌐 브라우저에서 HTML 보고서를 열었습니다!")
            except:
                print("💡 브라우저에서 수동으로 HTML 파일을 열어보세요.")
                
        except Exception as e:
            print(f"❌ HTML 파일 저장 실패: {e}")
    
    # HTML 보고서 생성 실행
    create_autoviz_html_report()

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='범용 EDA 도구')
    parser.add_argument('--data_path', required=True, help='데이터 파일 경로')
    parser.add_argument('--dataset_name', required=True, help='데이터셋 이름')
    parser.add_argument('--target_var', default=None, help='타겟 변수명 (선택사항)')
    parser.add_argument('--file_type', default='auto', choices=['csv', 'excel', 'parquet', 'auto'], 
                       help='파일 타입 (기본값: auto)')
    parser.add_argument('--output_dir', default='EDA', help='출력 디렉토리 (기본값: EDA)')
    parser.add_argument('--max_rows', type=int, default=1000, help='AutoViz 분석할 최대 행 수 (기본값: 1000)')
    parser.add_argument('--max_cols', type=int, default=20, help='AutoViz 분석할 최대 컬럼 수 (기본값: 20)')
    parser.add_argument('--dtale_port', type=int, default=4000, help='D-Tale 포트 (기본값: 4000)')
    parser.add_argument('--use_ngrok', action='store_true', help='ngrok 사용 여부')
    
    # 패키지 선택 옵션 추가
    parser.add_argument('--packages', nargs='+', 
                       choices=['ydata_profiling', 'sweetviz', 'autoviz', 'klib', 'dtale'],
                       default=['ydata_profiling', 'sweetviz', 'autoviz', 'klib', 'dtale'],
                       help='실행할 EDA 패키지들 (예: --packages ydata_profiling klib)')
    
    args = parser.parse_args()
    
    print("🚀 범용 EDA 도구 시작!")
    print(f"📁 데이터 경로: {args.data_path}")
    print(f"📊 데이터셋 이름: {args.dataset_name}")
    print(f"🎯 타겟 변수: {args.target_var if args.target_var else '없음'}")
    print(f"📦 실행할 패키지: {', '.join(args.packages)}")
    
    df = load_data(args.data_path, args.file_type, args.max_rows, args.target_var)
    if df is None:
        sys.exit(1) # 데이터 로드 실패 시 종료

    packages = create_folder_structure(args.dataset_name, args.output_dir)
    
    # 선택된 패키지만 실행
    if 'ydata_profiling' in args.packages:
        run_ydata_profiling(df, packages, args.dataset_name)
    else:
        print("⏭️ ydata-profiling 건너뛰기")
    
    if 'sweetviz' in args.packages:
        run_sweetviz(df, packages, args.dataset_name, args.target_var)
    else:
        print("⏭️ Sweetviz 건너뛰기")
    
    if 'autoviz' in args.packages:
        run_autoviz(df, packages, args.dataset_name, args.target_var, args.max_rows, args.max_cols)
    else:
        print("⏭️ AutoViz 건너뛰기")
    
    if 'klib' in args.packages:
        run_klib(df, packages, args.dataset_name)
    else:
        print("⏭️ Klib 건너뛰기")
    
    # HTML 리포트는 AutoViz가 실행된 경우에만 생성
    if 'autoviz' in args.packages:
        create_html_report(packages, args.dataset_name)
    
    if 'dtale' in args.packages:
        dtale_instance = run_dtale(df, packages, args.dataset_name, args.dtale_port, args.use_ngrok)
        if dtale_instance:
            print(f"\n⏳ D-Tale이 실행 중입니다. Ctrl+C를 눌러서 종료하세요.")
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 분석을 종료합니다.")
    else:
        print("⏭️ D-Tale 건너뛰기")
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("📁 생성된 파일들 확인")
    print(f"{'='*60}")
    
    for package_name in args.packages:
        if package_name in packages:
            package_dir = packages[package_name]
            if os.path.exists(package_dir):
                files = os.listdir(package_dir)
                if files:
                    print(f"✅ {package_name}/ 폴더: {len(files)}개 파일")
                    for file in files[:3]:  # 최대 3개만 표시
                        file_path = os.path.join(package_dir, file)
                        file_size = os.path.getsize(file_path) / 1024
                        print(f"  - {file} ({file_size:.0f}KB)")
                    if len(files) > 3:
                        print(f"  ... 외 {len(files)-3}개 파일")
                else:
                    print(f"⚠️ {package_name}/ 폴더: 파일 없음")
            else:
                print(f"❌ {package_name}/ 폴더: 존재하지 않음")
    
    print(f"\n{'='*60}")
    print("🎉 선택된 EDA 패키지 분석이 완료되었습니다!")
    print(f"💡 실행된 패키지: {', '.join(args.packages)}")
    print("💡 각 패키지의 특징:")
    if 'ydata_profiling' in args.packages:
        print("   • ydata-profiling: 포괄적인 데이터 품질 분석")
    if 'sweetviz' in args.packages:
        print("   • Sweetviz: 타겟 변수 중심의 상세 분석")
    if 'autoviz' in args.packages:
        print("   • AutoViz: 자동 시각화 및 패턴 발견")
    if 'klib' in args.packages:
        print("   • Klib: 데이터 클리닝 및 엑셀 분석 결과")
    if 'dtale' in args.packages:
        print("   • D-Tale: 대화형 데이터 탐색 인터페이스")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 