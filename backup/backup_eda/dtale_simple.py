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

# D-Tale 실행 - 더 간단한 방법
print("🌐 D-Tale 시작 중...")

# 명시적으로 포트와 호스트 지정
d = dtale.show(
    df, 
    name="Titanic Dataset",
    port=40000,
    host='localhost'
)

print("✅ D-Tale이 시작되었습니다!")
print(f"🌐 브라우저에서 다음 URL로 접속하세요: {d._url}")

# 잠시 기다린 후 브라우저 열기
print("⏳ 5초 후 브라우저를 열겠습니다...")
time.sleep(5)

try:
    # localhost URL 시도
    localhost_url = "http://localhost:40000"
    print(f"🌐 {localhost_url} 시도 중...")
    webbrowser.open(localhost_url)
    print(f"✅ {localhost_url}에서 브라우저가 열렸습니다!")
    
except Exception as e:
    print(f"❌ 브라우저 자동 열기 실패: {e}")

print("\n💡 만약 여전히 접속이 안 되면 다음을 시도해보세요:")
print("1. 브라우저에서 직접 http://localhost:40000 접속")
print("2. http://127.0.0.1:40000 접속")
print("3. 다른 포트로 시도: http://localhost:40001")

print("\n⏳ D-Tale이 실행 중입니다. Ctrl+C를 눌러서 종료하세요.")

# 무한 루프로 서버 유지
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n�� D-Tale을 종료합니다.") 