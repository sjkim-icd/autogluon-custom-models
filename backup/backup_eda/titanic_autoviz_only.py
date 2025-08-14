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

# ============================================================================
# AUTOVIZ 적용 (폴더 저장 버전)
# ============================================================================
print("\n" + "="*60)
print("🎨 AUTOVIZ 적용 - 폴더 저장 버전")
print("="*60)

from autoviz.AutoViz_Class import AutoViz_Class
import os

# 저장할 폴더 생성
plot_dir = "autoviz_plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"📁 '{plot_dir}' 폴더를 생성했습니다.")

print("🎨 AutoViz로 자동 시각화 생성 중...")
AV = AutoViz_Class()

# AutoViz 실행 - 명시적으로 폴더 저장
df_viz = AV.AutoViz(
    filename="",  # 파일명이 없으면 데이터프레임 사용
    dfte=df,     # 데이터프레임
    depVar='survived',  # 타겟 변수
    max_rows_analyzed=1000,  # 분석할 최대 행 수
    max_cols_analyzed=20,    # 분석할 최대 컬럼 수
    verbose=2,               # 더 상세한 출력
    save_plot_dir=plot_dir,  # 플롯 저장 디렉토리
    chart_format='png'       # PNG 형식으로 저장
)

print(f"✅ AutoViz 시각화가 '{plot_dir}' 폴더에 저장되었습니다!")

# 폴더 내용 확인
print(f"\n📁 '{plot_dir}' 폴더 내용:")
if os.path.exists(plot_dir):
    files = os.listdir(plot_dir)
    if files:
        for file in files:
            print(f"  - {file}")
    else:
        print("  (폴더가 비어있습니다)")
else:
    print("  (폴더가 존재하지 않습니다)")

print("\n🎉 AutoViz 분석이 완료되었습니다!") 