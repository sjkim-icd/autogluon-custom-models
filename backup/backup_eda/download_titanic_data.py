import pandas as pd
from sklearn.datasets import fetch_openml
import os

print("🚢 타이타닉 데이터 다운로드 중...")

# 타이타닉 데이터 다운로드
titanic = fetch_openml(name='titanic', version=1, as_frame=True)

# 데이터프레임으로 변환
df = titanic.frame

print(f"📊 데이터 형태: {df.shape}")
print(f"📋 컬럼: {list(df.columns)}")
print(f"🎯 타겟 변수: survived")

# datasets 폴더에 저장
output_path = "datasets/titanic.csv"
df.to_csv(output_path, index=False)

print(f"✅ 타이타닉 데이터가 '{output_path}'에 저장되었습니다!")
print(f"📁 파일 크기: {os.path.getsize(output_path) / 1024:.1f} KB")

# 데이터 미리보기
print("\n📋 데이터 미리보기:")
print(df.head())
print(f"\n📊 데이터 정보:")
print(df.info()) 