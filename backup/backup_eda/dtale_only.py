import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import dtale

# 타이타닉 데이터 로드
print("🚢 타이타닉 데이터 로딩 중...")
titanic = fetch_openml(name='titanic', version=1, as_frame=True)
df = titanic.frame

print(f"📊 데이터 형태: {df.shape}")

# D-Tale 실행
print("🌐 D-Tale 시작 중...")
d = dtale.show(df, name="Titanic Dataset")

print("✅ D-Tale이 시작되었습니다!")
print(f"🌐 브라우저에서 다음 URL로 접속하세요: {d._url}")
print("💡 브라우저가 자동으로 열리지 않으면 위 URL을 복사해서 접속하세요!")

# 브라우저 자동 열기 시도
import webbrowser
try:
    webbrowser.open(d._url)
    print("🌐 브라우저를 자동으로 열었습니다!")
except:
    print("❌ 브라우저 자동 열기 실패. 위 URL을 수동으로 복사해서 접속하세요.")

print("\n⏳ D-Tale이 실행 중입니다. Ctrl+C를 눌러서 종료하세요.") 