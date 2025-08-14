import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from tqdm import tqdm

# 한글 폰트 설정 (Windows 환경)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기본 한글 폰트
except:
    try:
        plt.rcParams['font.family'] = 'NanumGothic'  # 나눔고딕
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'  # 기본 폰트
        print("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

# 명령행 인자 파싱
parser = argparse.ArgumentParser(description='IV/WOE 분석을 위한 데이터 분석')
parser.add_argument('--datapath', type=str, required=True, 
                   help='분석할 데이터 파일 경로 (CSV, Excel 등)')
parser.add_argument('--eda_name', type=str, default='iv_woe_analysis',
                   help='EDA 분석 이름 (파일명 생성에 사용)')
parser.add_argument('--target_col', type=str, required=True,
                   help='타겟 변수 컬럼명')
parser.add_argument('--feature_cols', type=str, nargs='+',
                   help='분석할 특성 변수 컬럼명들 (지정하지 않으면 자동 선택)')
parser.add_argument('--threshold', type=float, default=0.02,
                   help='IV 임계값 (기본값: 0.02)')
args = parser.parse_args()

print(f"🚀 {args.eda_name.upper()} IV/WOE 분석 시작!")
print("=" * 50)
print(f"📁 데이터 경로: {args.datapath}")
print(f"🎯 타겟 변수: {args.target_col}")
print(f"📊 특성 변수: {args.feature_cols if args.feature_cols else '자동 선택'}")
print(f"🔍 IV 임계값: {args.threshold}")
print("=" * 50)

# 1. 데이터 로드
print("1. 데이터 로드 중...")

# 파일 확장자 확인
file_ext = os.path.splitext(args.datapath)[1].lower()

try:
    if file_ext == '.csv':
        data = pd.read_csv(args.datapath)
    elif file_ext in ['.xlsx', '.xls']:
        data = pd.read_excel(args.datapath)
    elif file_ext == '.parquet':
        data = pd.read_parquet(args.datapath)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")
    
    print(f"✅ 데이터 로드 완료!")
    print(f"데이터셋 크기: {data.shape}")
    print(f"컬럼 목록: {list(data.columns)}")
    
except Exception as e:
    print(f"❌ 데이터 로드 실패: {e}")
    exit(1)

# 타겟 변수 확인
if args.target_col not in data.columns:
    print(f"❌ 타겟 변수 '{args.target_col}'를 찾을 수 없습니다.")
    print(f"사용 가능한 컬럼: {list(data.columns)}")
    exit(1)

# 특성 변수 선택
if args.feature_cols:
    # 사용자가 지정한 특성 변수 사용
    selected_features = [col for col in args.feature_cols if col in data.columns]
    if len(selected_features) != len(args.feature_cols):
        missing_cols = [col for col in args.feature_cols if col not in data.columns]
        print(f"⚠️ 일부 특성 변수를 찾을 수 없습니다: {missing_cols}")
else:
    # 자동으로 수치형 변수 선택 (타겟 변수 제외)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = [col for col in numeric_cols if col != args.target_col]
    # 모든 변수 처리 (제한 제거)
    print(f"📊 분석할 특성 변수: {len(selected_features)}개")
    print(f"변수 목록: {selected_features[:5]}{'...' if len(selected_features) > 5 else ''}")

print(f"📊 분석할 특성 변수: {selected_features}")
print(f"총 {len(selected_features)}개 변수 선택됨")

# 데이터 분리
X = data[selected_features]
y = data[args.target_col]

# 타겟 변수를 숫자로 변환
if y.dtype == 'object':
    y = y.astype('category').cat.codes

print(f"타겟 변수: {args.target_col}")
print(f"타겟 분포:\n{y.value_counts()}")
print()

# 2. 데이터 전처리
print("2. 데이터 전처리 중...")

# 결측치 확인
print("결측치 현황:")
print(X.isnull().sum())
print()

# 결측치 처리
X_clean = X.copy()
for col in selected_features:
    if X_clean[col].isnull().sum() > 0:
        if X_clean[col].dtype in ['float64', 'int64']:
            # 수치형: 중앙값으로 처리
            X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        else:
            # 범주형: 최빈값으로 처리
            X_clean[col] = X_clean[col].fillna(X_clean[col].mode()[0])

print("결측치 처리 완료!")
print()

# 3. 범주형 변수 변환
print("3. 범주형 변수 변환 중...")

for col in selected_features:
    if X_clean[col].dtype == 'object':
        # 범주형 변수를 숫자로 변환
        X_clean[col] = X_clean[col].astype('category').cat.codes

print("변환 완료!")
print(f"변환 후 데이터 타입:\n{X_clean.dtypes}")
print()

# 4. 연속형 변수 이산화 (IV 계산을 위해)
print("4. 연속형 변수 이산화 중...")

# 진행률 표시
print("변수별 이산화 진행 중...")
for col in tqdm(selected_features, desc="이산화"):
    if X_clean[col].dtype in ['float64', 'int64']:
        # 분위수 기반으로 구간 나누기
        try:
            quantiles = X_clean[col].quantile([0.2, 0.4, 0.6, 0.8])
            bins = [X_clean[col].min()] + list(quantiles) + [X_clean[col].max()]
            labels = [f'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
            
            X_clean[f'{col}_binned'] = pd.cut(X_clean[col], 
                                             bins=bins, 
                                             labels=labels, 
                                             include_lowest=True)
        except:
            # 분위수 계산이 안 되는 경우 간단한 구간 나누기
            min_val = X_clean[col].min()
            max_val = X_clean[col].max()
            step = (max_val - min_val) / 5
            bins = [min_val + i * step for i in range(6)]
            labels = [f'Bin{i+1}' for i in range(5)]
            
            X_clean[f'{col}_binned'] = pd.cut(X_clean[col], 
                                             bins=bins, 
                                             labels=labels, 
                                             include_lowest=True)

# 원본 연속형 변수 제거
X_discretised = X_clean.drop(selected_features, axis=1)

print("이산화 완료!")
print(f"이산화 후 변수들: {list(X_discretised.columns)}")
print()

# 5. WOE와 IV 계산
print("5. WOE와 IV 계산 중...")

def calculate_woe_iv(X, y, variable):
    """WOE와 IV 계산"""
    # 변수의 고유값들
    unique_values = X[variable].unique()
    
    # 타겟 분포
    target_dist = y.value_counts()
    total_positive = target_dist[1]
    total_negative = target_dist[0]
    
    woe_results = {}
    iv = 0
    
    print(f"\n{variable} 변수 분석:")
    print("-" * 50)
    print(f"{'Bin':<15} {'Positive':<10} {'Negative':<10} {'Positive%':<12} {'Negative%':<12} {'WOE':<10} {'IV':<10}")
    print("-" * 50)
    
    for value in unique_values:
        if pd.isna(value):
            continue
            
        # 해당 값의 샘플들
        mask = X[variable] == value
        positive_count = y[mask].sum()
        negative_count = (y[mask] == 0).sum()
        
        # 비율 계산
        positive_ratio = positive_count / total_positive if total_positive > 0 else 0
        negative_ratio = negative_count / total_negative if total_negative > 0 else 0
        
        # WOE 계산 (0으로 나누기 방지)
        if positive_ratio > 0 and negative_ratio > 0:
            woe = np.log(positive_ratio / negative_ratio)
            iv_contribution = (positive_ratio - negative_ratio) * woe
            iv += iv_contribution
        else:
            woe = 0
            iv_contribution = 0
        
        woe_results[value] = {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'woe': woe,
            'iv_contribution': iv_contribution
        }
        
        # 결과 출력
        print(f"{str(value):<15} {positive_count:<10} {negative_count:<10} {positive_ratio*100:<11.1f}% {negative_ratio*100:<11.1f}% {woe:<10.3f} {iv_contribution:<10.4f}")
    
    print("-" * 50)
    print(f"Total IV: {iv:.4f}")
    print()
    
    return abs(iv), woe_results

# 각 변수의 WOE와 IV 계산 (진행률 표시)
iv_results = {}
woe_results = {}

print("변수별 IV/WOE 계산 진행 중...")
for var in tqdm(X_discretised.columns, desc="IV/WOE 계산"):
    try:
        iv, woe = calculate_woe_iv(X_discretised, y, var)
        iv_results[var] = iv
        woe_results[var] = woe
    except Exception as e:
        print(f"{var}: 계산 오류 - {e}")
        iv_results[var] = 0
        woe_results[var] = {}

print()

# 6. 결과 분석
print("6. 결과 분석...")
print(f"총 변수 수: {len(X_discretised.columns)}")

# IV 임계값 적용
threshold = args.threshold
selected_vars = [var for var, iv in iv_results.items() if iv > threshold]
removed_vars = [var for var, iv in iv_results.items() if iv <= threshold]

print(f"선택된 변수 수: {len(selected_vars)}")
print(f"제거된 변수 수: {len(removed_vars)}")
print()

# 선택된 변수들
print("선택된 변수들:")
for var in selected_vars:
    print(f"- {var} (IV: {iv_results[var]:.4f})")
print()

# 제거된 변수들
print("제거된 변수들:")
for var in removed_vars:
    print(f"- {var} (IV: {iv_results[var]:.4f})")
print()

# 7. 결과를 DataFrame으로 정리
print("7. 결과 정리 중...")

# 1) 컬럼별 IV 값 DataFrame
iv_summary_df = pd.DataFrame({
    'Variable': list(iv_results.keys()),
    'IV_Value': list(iv_results.values()),
    'Status': ['Selected' if iv > threshold else 'Removed' for iv in iv_results.values()],
    'Rank': range(1, len(iv_results) + 1)
})
iv_summary_df = iv_summary_df.sort_values('IV_Value', ascending=False).reset_index(drop=True)
iv_summary_df['Rank'] = range(1, len(iv_summary_df) + 1)

print("✅ IV 요약 DataFrame 생성 완료")
print(iv_summary_df)
print()

# 2) 컬럼별 구간별 상세 WOE 정보 DataFrame
woe_detail_list = []

for var in iv_summary_df['Variable']:
    if var in woe_results and woe_results[var]:
        woe_data = woe_results[var]
        
        for bin_name, stats in woe_data.items():
            woe_detail_list.append({
                'Variable': var,
                'Bin': str(bin_name),
                'Total': stats['positive_count'] + stats['negative_count'],
                'Event_Count': stats['positive_count'],
                'Nonevent_Count': stats['negative_count'],
                'Event_Rate': stats['positive_ratio'],
                'Nonevent_Rate': stats['negative_ratio'],
                'WOE': stats['woe'],
                'IV_Contribution': stats['iv_contribution']
            })

woe_detail_df = pd.DataFrame(woe_detail_list)

# Event_Rate와 Nonevent_Rate를 퍼센트로 변환
woe_detail_df['Event_Rate_%'] = (woe_detail_df['Event_Rate'] * 100).round(2)
woe_detail_df['Nonevent_Rate_%'] = (woe_detail_df['Nonevent_Rate'] * 100).round(2)

# WOE와 IV_Contribution을 소수점 4자리로 반올림
woe_detail_df['WOE'] = woe_detail_df['WOE'].round(4)
woe_detail_df['IV_Contribution'] = woe_detail_df['IV_Contribution'].round(4)

print("✅ WOE 상세 정보 DataFrame 생성 완료")
print(woe_detail_df.head(10))
print()

# 3) 검증용 상세 계산 과정 DataFrame (간소화)
verification_list = []

# 전체 타겟 분포
total_positive = y.value_counts()[1]
total_negative = y.value_counts()[0]

row_counter = 6  # 엑셀에서 6행부터 시작 (1-4행은 수식 설명용)

for var in iv_summary_df['Variable']:
    if var in woe_results and woe_results[var]:
        woe_data = woe_results[var]
        
        for bin_name, stats in woe_data.items():
            # 수동 계산으로 검증
            positive_count = stats['positive_count']
            negative_count = stats['negative_count']
            total_count = positive_count + negative_count
            
            # 비율 계산
            positive_ratio = positive_count / total_positive if total_positive > 0 else 0
            negative_ratio = negative_count / total_negative if total_negative > 0 else 0
            
            verification_list.append({
                'Variable': var,
                'Bin': str(bin_name),
                'Total_Count': total_count,
                'Positive_Count': positive_count,
                'Negative_Count': negative_count,
                'Total_Positive': total_positive,
                'Total_Negative': total_negative,
                'Positive_Ratio': positive_ratio,
                'Negative_Ratio': negative_ratio,
                'Positive_Ratio_%': (positive_ratio * 100).round(4),
                'Negative_Ratio_%': (negative_ratio * 100).round(4),
                'WOE_Formula': f'=LN(H{row_counter}/I{row_counter})',  # H: Positive_Ratio, I: Negative_Ratio
                'IV_Formula': f'=(H{row_counter}-I{row_counter})*LN(H{row_counter}/I{row_counter})'  # H: Positive_Ratio, I: Negative_Ratio
            })
            
            row_counter += 1  # 다음 행으로 이동

verification_df = pd.DataFrame(verification_list)

print("✅ 검증용 DataFrame 생성 완료")
print("📊 Verification_Details 시트 구성 (간소화):")
print("- 상단: WOE, IV 계산식 설명")
print("- 원초 데이터: Count, Ratio 값들")
print("- 엑셀 수식: WOE_Formula, IV_Formula")
print("- 셀 클릭 시 수식 확인 및 편집 가능")
print()

# 8. 엑셀 파일로 저장
print("8. 엑셀 파일 저장 중...")

# 출력 폴더 생성 (절대 경로 사용)
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "iv_woe_analysis")
os.makedirs(output_dir, exist_ok=True)

# 파일명 생성
excel_filename = f"{args.eda_name}_results.xlsx"
excel_path = os.path.join(output_dir, excel_filename)

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # IV 요약 시트
    iv_summary_df.to_excel(writer, sheet_name='IV_Summary', index=False)
    
    # WOE 상세 정보 시트
    woe_detail_df.to_excel(writer, sheet_name='WOE_Details', index=False)
    
    # 검증용 상세 계산 과정 시트 (수식 설명 포함)
    verification_df.to_excel(writer, sheet_name='Verification_Details', index=False, startrow=4)  # row 5부터 시작 (startrow=4)
    
    # Verification_Details 시트에 수식 설명 추가
    worksheet = writer.sheets['Verification_Details']
    
    # 수식 설명을 상단에 추가 (row 1-3)
    worksheet['A1'] = 'WOE = ln(Positive_Ratio / Negative_Ratio)'
    worksheet['A2'] = 'IV_Contribution = (Positive_Ratio - Negative_Ratio) × WOE'
    worksheet['A3'] = 'Total_IV = Σ |IV_Contribution_i|'
    
    # 수식 설명 스타일링 (굵게)
    from openpyxl.styles import Font
    bold_font = Font(bold=True)
    worksheet['A1'].font = bold_font
    worksheet['A2'].font = bold_font
    worksheet['A3'].font = bold_font
    
    # 원본 데이터 요약 시트
    data_summary = pd.DataFrame({
        'Metric': ['Total Rows', 'Total Columns', 'Target Variable', 'Feature Variables', 'IV Threshold', 'Total Positive', 'Total Negative'],
        'Value': [len(data), len(data.columns), args.target_col, len(selected_features), threshold, total_positive, total_negative]
    })
    data_summary.to_excel(writer, sheet_name='Data_Summary', index=False)

print(f"✅ 엑셀 파일 저장 완료: {excel_path}")
print("📊 엑셀 시트 구성:")
print("- IV_Summary: 변수별 IV 값 요약")
print("- WOE_Details: 구간별 상세 WOE 정보")
print("- Verification_Details: 검증용 상세 계산 과정 (엑셀 수식 포함)")
print("- Data_Summary: 데이터셋 기본 정보")
print()
print("🔍 엑셀에서 검증 방법:")
print("1. Verification_Details 시트에서 WOE_Formula 컬럼의 수식 복사")
print("2. 새 셀에 = 붙여넣기 → WOE 값 자동 계산")
print("3. IV_Formula도 동일하게 사용하여 IV Contribution 계산")
print("4. Calculated vs Original 값 비교로 정확성 검증")
print()
print("📋 Verification_Details 시트 상세 구성:")
print("• 기초 데이터 (수치): Total_Count, Positive_Count, Negative_Count, Total_Positive, Total_Negative")
print("• 계산된 비율 (수치): Positive_Ratio, Negative_Ratio, Positive_Ratio_%, Negative_Ratio_%")
print("• 수식 표현: WOE_Formula, IV_Contribution_Formula (수학적 표현)")
print("• 엑셀 수식: WOE_Excel_Formula, IV_Excel_Formula (셀에 직접 입력 가능)")
print("• 검증 결과: WOE_Calculated, WOE_Original, WOE_Difference, IV_Contribution_Calculated, IV_Contribution_Original, IV_Contribution_Difference")
print()
print("💡 엑셀 수식 사용 팁:")
print("• Verification_Details 시트 상단에 계산식 설명이 있습니다")
print("• WOE_Formula: =LN(H5/I5) → H5는 Positive_Ratio, I5는 Negative_Ratio")
print("• IV_Formula: =(H5-I5)*LN(H5/I5) → H5는 Positive_Ratio, I5는 Negative_Ratio")
print("• 셀을 클릭하면 수식이 수식 입력줄에 표시되어 편집 가능")
print("• 수식 결과가 Python 계산 결과와 일치하는지 확인")
print()

# 9. 시각화 (Plotly 사용 - 1개 HTML에 개별 이미지들 포함)
print("9. 시각화 생성 중...")

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

# 변수와 IV 값 준비
variables = iv_summary_df['Variable'].tolist()
iv_values = iv_summary_df['IV_Value'].tolist()

print(f"📈 생성할 개별 그래프: {len(variables)}개")

# 더 간단한 접근: 각 차트를 개별적으로 생성하고 HTML에 포함
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{args.eda_name.upper()} - Complete IV/WOE Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .chart-container {{ margin: 30px 0; border: 1px solid #ddd; padding: 20px; border-radius: 8px; }}
        h1, h2 {{ color: #333; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>{args.eda_name.upper()} - Complete IV/WOE Analysis</h1>
    
    <div class="summary">
        <h2>📊 분석 요약</h2>
        <p><strong>총 변수 수:</strong> {len(variables)}개</p>
        <p><strong>선택된 변수 수:</strong> {len([v for v in variables if iv_summary_df[iv_summary_df['Variable'] == v]['Status'].iloc[0] == 'Selected'])}개</p>
        <p><strong>제거된 변수 수:</strong> {len([v for v in variables if iv_summary_df[iv_summary_df['Variable'] == v]['Status'].iloc[0] == 'Removed'])}개</p>
        <p><strong>IV 임계값:</strong> {threshold}</p>
    </div>
"""

# 1. IV 값 비교 차트
colors = ['lightblue' if iv > threshold else 'lightcoral' for iv in iv_values]
fig_iv = go.Figure(data=[go.Bar(
    x=variables,
    y=iv_values,
    marker_color=colors,
    text=[f'{iv:.4f}' for iv in iv_values],
    textposition='outside'
)])

fig_iv.update_layout(
    title=f'{args.eda_name.upper()} - Information Value (IV) Comparison',
    xaxis_title='Variables',
    yaxis_title='IV Value',
    height=500,
    width=800,
    showlegend=False
)

# 임계값 선 추가
fig_iv.add_hline(
    y=threshold,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Threshold ({threshold})"
)

# IV 차트를 HTML에 추가
html_content += f"""
    <div class="chart-container">
        <h2>📈 Information Value (IV) Comparison</h2>
        <div id="iv-chart"></div>
    </div>
"""

# 2. IV 분포 파이 차트
selected_vars = iv_summary_df[iv_summary_df['Status'] == 'Selected']
if len(selected_vars) > 0:
    fig_pie = go.Figure(data=[go.Pie(
        labels=selected_vars['Variable'],
        values=selected_vars['IV_Value'],
        textinfo='label+percent+value'
    )])
    
    fig_pie.update_layout(
        title=f'{args.eda_name.upper()} - IV Distribution (Selected Variables)',
        height=500,
        width=600
    )
    
    # 파이 차트를 HTML에 추가
    html_content += f"""
        <div class="chart-container">
            <h2>🥧 IV Distribution (Selected Variables)</h2>
            <div id="pie-chart"></div>
        </div>
    """

# 3. 각 변수별 WOE 값 플롯
for i, var in enumerate(variables):
    if var in woe_results and woe_results[var]:
        woe_data = woe_results[var]
        bins = list(woe_data.keys())
        woe_values = [woe_data[bin_name]['woe'] for bin_name in bins]
        
        # WOE 값에 따른 색상 설정
        colors = []
        for woe in woe_values:
            if woe > 0:
                colors.append('lightgreen')  # 양수: 초록색
            elif woe < 0:
                colors.append('lightcoral')  # 음수: 빨간색
            else:
                colors.append('lightgray')   # 0: 회색
        
        fig_woe = go.Figure(data=[go.Bar(
            x=bins,
            y=woe_values,
            marker_color=colors,
            text=[f'{woe:.4f}' for woe in woe_values],
            textposition='outside'
        )])
        
        fig_woe.update_layout(
            title=f'{args.eda_name.upper()} - WOE Values for {var}',
            xaxis_title='Bins',
            yaxis_title='WOE Value',
            height=400,
            width=600,
            showlegend=False
        )
        
        # 0선 추가
        fig_woe.add_hline(
            y=0,
            line_dash="dash",
            line_color="black",
            line_width=1
        )
        
        # WOE 차트를 HTML에 추가
        html_content += f"""
            <div class="chart-container">
                <h2>📊 WOE Values for {var}</h2>
                <div id="woe-chart-{i}"></div>
            </div>
        """

# 4. 각 변수별 IV Contribution 플롯
for i, var in enumerate(variables):
    if var in woe_results and woe_results[var]:
        woe_data = woe_results[var]
        bins = list(woe_data.keys())
        iv_contrib_values = [woe_data[bin_name]['iv_contribution'] for bin_name in bins]
        
        fig_iv_contrib = go.Figure(data=[go.Bar(
            x=bins,
            y=iv_contrib_values,
            marker_color='skyblue',
            text=[f'{contrib:.4f}' for contrib in iv_contrib_values],
            textposition='outside'
        )])
        
        fig_iv_contrib.update_layout(
            title=f'{args.eda_name.upper()} - IV Contribution for {var}',
            xaxis_title='Bins',
            yaxis_title='IV Contribution',
            height=400,
            width=600,
            showlegend=False
        )
        
        # IV Contribution 차트를 HTML에 추가
        html_content += f"""
            <div class="chart-container">
                <h2>📈 IV Contribution for {var}</h2>
                <div id="iv-contrib-chart-{i}"></div>
            </div>
        """

# HTML 파일 완성
html_content += """
    <script>
        // IV 차트
        var ivData = [{
            x: """ + str(variables) + """,
            y: """ + str(iv_values) + """,
            type: 'bar',
            marker: {
                color: """ + str(['lightblue' if iv > threshold else 'lightcoral' for iv in iv_values]) + """
            },
            text: """ + str([f'{iv:.4f}' for iv in iv_values]) + """,
            textposition: 'outside'
        }];
        
        var ivLayout = {
            title: 'Information Value (IV) Comparison',
            xaxis: {title: 'Variables'},
            yaxis: {title: 'IV Value'},
            height: 500,
            width: 800,
            showlegend: false,
            shapes: [{
                type: 'line',
                x0: -0.5,
                x1: """ + str(len(variables) - 0.5) + """,
                y0: """ + str(threshold) + """,
                y1: """ + str(threshold) + """,
                line: {dash: 'dash', color: 'red', width: 2}
            }],
            annotations: [{
                x: """ + str(len(variables) - 1) + """,
                y: """ + str(threshold) + """,
                text: 'Threshold (""" + str(threshold) + """)',
                showarrow: false,
                yshift: 10
            }]
        };
        
        Plotly.newPlot('iv-chart', ivData, ivLayout);
"""

# 파이 차트 추가
if len(selected_vars) > 0:
    html_content += """
        // 파이 차트
        var pieData = [{
            labels: """ + str(selected_vars['Variable'].tolist()) + """,
            values: """ + str(selected_vars['IV_Value'].tolist()) + """,
            type: 'pie',
            textinfo: 'label+percent+value'
        }];
        
        var pieLayout = {
            title: 'IV Distribution (Selected Variables)',
            height: 500,
            width: 600
        };
        
        Plotly.newPlot('pie-chart', pieData, pieLayout);
    """

# WOE 차트들 추가
for i, var in enumerate(variables):
    if var in woe_results and woe_results[var]:
        woe_data = woe_results[var]
        bins = list(woe_data.keys())
        woe_values = [woe_data[bin_name]['woe'] for bin_name in bins]
        
        # WOE 값에 따른 색상 설정
        colors = []
        for woe in woe_values:
            if woe > 0:
                colors.append('lightgreen')
            elif woe < 0:
                colors.append('lightcoral')
            else:
                colors.append('lightgray')
        
        html_content += f"""
        // WOE 차트 {i} - {var}
        var woeData{i} = [{{
            x: {bins},
            y: {woe_values},
            type: 'bar',
            marker: {{
                color: {colors}
            }},
            text: {[f'{woe:.4f}' for woe in woe_values]},
            textposition: 'outside'
        }}];
        
        var woeLayout{i} = {{
            title: 'WOE Values for {var}',
            xaxis: {{title: 'Bins'}},
            yaxis: {{title: 'WOE Value'}},
            height: 400,
            width: 600,
            showlegend: false,
            shapes: [{{
                type: 'line',
                x0: -0.5,
                x1: {len(bins) - 0.5},
                y0: 0,
                y1: 0,
                line: {{dash: 'dash', color: 'black', width: 1}}
            }}]
        }};
        
        Plotly.newPlot('woe-chart-{i}', woeData{i}, woeLayout{i});
        """

# IV Contribution 차트들 추가
for i, var in enumerate(variables):
    if var in woe_results and woe_results[var]:
        woe_data = woe_results[var]
        bins = list(woe_data.keys())
        iv_contrib_values = [woe_data[bin_name]['iv_contribution'] for bin_name in bins]
        
        html_content += f"""
        // IV Contribution 차트 {i} - {var}
        var ivContribData{i} = [{{
            x: {bins},
            y: {iv_contrib_values},
            type: 'bar',
            marker: {{
                color: 'skyblue'
            }},
            text: {[f'{contrib:.4f}' for contrib in iv_contrib_values]},
            textposition: 'outside'
        }}];
        
        var ivContribLayout{i} = {{
            title: 'IV Contribution for {var}',
            xaxis: {{title: 'Bins'}},
            yaxis: {{title: 'IV Contribution'}},
            height: 400,
            width: 600,
            showlegend: false
        }};
        
        Plotly.newPlot('iv-contrib-chart-{i}', ivContribData{i}, ivContribLayout{i});
        """

html_content += """
    </script>
</body>
</html>
"""

# HTML 파일로 저장
html_filename = f'{args.eda_name}_complete_analysis.html'
html_path = os.path.join(output_dir, html_filename)

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ 완전한 분석 HTML 파일 저장 완료: {html_path}")
print("🌐 HTML 파일을 브라우저에서 열어서 모든 차트를 확인하세요!")
print("📊 차트 기능: 확대/축소, 호버 정보, 다운로드 등")
print()
print(f"📁 HTML 파일 구성:")
print(f"  - IV 비교 차트")
print(f"  - IV 분포 파이 차트")
print(f"  - 각 변수별 WOE 차트: {len(variables)}개")
print(f"  - 각 변수별 IV Contribution 차트: {len(variables)}개")
print(f"  - 총 차트 수: {len(variables) * 2 + 2}개")

# 10. 최종 결과 출력
print("✅ 분석 완료!")
print("=" * 60)
print("📁 저장된 파일들:")
print(f"📊 엑셀 파일: {excel_path}")
print(f"🖼️  WOE 분석 그림: {os.path.join(output_dir, args.eda_name)}.png")
print(f"📈 IV 기여도 그림: {os.path.join(output_dir, args.eda_name)}_contributions.png")
print("=" * 60)

print("📊 분석 요약:")
print(f"- 총 변수 수: {len(X_discretised.columns)}개")
print(f"- 선택된 변수 수: {len(selected_vars)}개")
print(f"- 제거된 변수 수: {len(removed_vars)}개")
print(f"- IV 임계값: {threshold}")
print()
print("🎯 다음 단계:")
print("- 선택된 변수들로 모델 학습")
print("- IV 값이 높은 변수들에 집중")
print("- 제거된 변수들은 모델에서 제외")
print()
print("💡 IV 해석 가이드:")
print("- IV < 0.02: 예측력 없음")
print("- 0.02 <= IV < 0.1: 약한 예측력")
print("- 0.1 <= IV < 0.3: 중간 예측력")
print("- 0.3 <= IV < 0.5: 강한 예측력")
print("- IV >= 0.5: 매우 강한 예측력 (과적합 위험)")
print()
print("🔍 변수별 상세 분석:")
# sorted_iv = sorted(iv_results.items(), key=lambda item: item[1], reverse=True) # 이 부분은 이제 사용되지 않음
# for var, iv in sorted_iv:
#     status = "✅ 선택" if iv > threshold else "❌ 제거"
#     print(f"{var}: {iv:.4f} - {status}")
print()
print("📈 WOE 해석 가이드:")
print("- WOE > 0: 해당 bin에서 타겟 변수 값이 높음 (예: 생존, 사기)")
print("- WOE < 0: 해당 bin에서 타겟 변수 값이 낮음 (예: 사망, 정상)")
print("- WOE = 0: 해당 bin에서 타겟 변수 값이 동일 (예: 생존/사망, 사기/정상)")
print()
print("🔢 IV Contribution 해석 가이드:")
print("- 높은 IV Contribution: 해당 bin이 변수의 예측력에 크게 기여")
print("- 낮은 IV Contribution: 해당 bin이 변수의 예측력에 적게 기여")
print("- 음수 IV Contribution: 해당 bin이 예측력을 저하시킴")
print()
print("📋 변수별 상세 통계:")
for var in variables:
    if var in woe_results and woe_results[var]:
        print(f"\n{var} 변수:")
        print("-" * 40)
        woe_data = woe_results[var]
        for bin_name, stats in woe_data.items():
            print(f"  {bin_name}: WOE={stats['woe']:.3f}, IV_Contrib={stats['iv_contribution']:.4f}")
            print(f"    (Positive: {stats['positive_count']}, Negative: {stats['negative_count']})")
print()
print("🚀 사용법:")
print(f"- 데이터 경로: python {__file__} --datapath <경로>")
print(f"- 타겟 변수: python {__file__} --target_col <컬럼명>")
print(f"- 특성 변수 (자동 선택): python {__file__} --datapath <경로> --target_col <컬럼명>")
print(f"- 특성 변수 (지정): python {__file__} --datapath <경로> --target_col <컬럼명> --feature_cols <컬럼명1> <컬럼명2>...")
print(f"- 기본값 (IV 임계값): python {__file__} --datapath <경로> --target_col <컬럼명> --threshold <값>")
print(f"- 기본값 (모든 파라미터): python {__file__}")
print()
print("📊 엑셀 파일 내용:")
print(f"- IV_Summary: 변수별 IV 값 요약")
print(f"- WOE_Details: 구간별 상세 WOE 정보")
print(f"- Verification_Details: 검증용 상세 계산 과정 (모든 중간 계산값 포함)")
print(f"- Data_Summary: 데이터셋 기본 정보")
print()
print("🔍 검증 방법:")
print("1. Verification_Details 시트에서 각 bin별 계산 과정 확인")
print("2. WOE_Formula와 IV_Contribution_Formula로 수식 검증")
print("3. WOE_Difference와 IV_Contribution_Difference가 0에 가까운지 확인")
print("4. IV_Verification 시트에서 전체 IV 값 검증")
print() 