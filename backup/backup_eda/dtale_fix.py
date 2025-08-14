import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import dtale
import webbrowser
import time

# 타이타닉 데이터 로드
print("🚢 타이타닉 데이터 로딩 중...")
titanic = fetch_openml(name='titanic', version=1, as_frame=True)
df = titanic.frame

print(f"📊 데이터 형태: {df.shape}")

# D-Tale 실행 - 명시적으로 포트 지정
print("🌐 D-Tale 시작 중...")
d = dtale.show(df, name="Titanic Dataset", port=40000)

print("✅ D-Tale이 시작되었습니다!")
print(f"🌐 브라우저에서 다음 URL로 접속하세요: {d._url}")

# 잠시 기다린 후 브라우저 열기
print("⏳ 3초 후 브라우저를 열겠습니다...")
time.sleep(3)

try:
    # 여러 URL 시도
    urls_to_try = [
        d._url,
        "http://localhost:40000",
        "http://127.0.0.1:40000",
        "http://localhost:40001",
        "http://127.0.0.1:40001"
    ]
    
    for url in urls_to_try:
        try:
            print(f"🌐 {url} 시도 중...")
            webbrowser.open(url)
            print(f"✅ {url}에서 브라우저가 열렸습니다!")
            break
        except Exception as e:
            print(f"❌ {url} 실패: {e}")
            continue
    else:
        print("❌ 모든 URL 시도 실패")
        
except Exception as e:
    print(f"❌ 브라우저 자동 열기 실패: {e}")

print("\n💡 수동으로 다음 URL들을 브라우저에 복사해서 시도해보세요:")
for url in ["http://localhost:40000", "http://127.0.0.1:40000", "http://localhost:40001", "http://127.0.0.1:40001"]:
    print(f"  {url}")

print("\n⏳ D-Tale이 실행 중입니다. Ctrl+C를 눌러서 종료하세요.") 