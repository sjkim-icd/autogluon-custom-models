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

# 여러 포트로 D-Tale 시도
ports_to_try = [40000, 40001, 40002, 40003, 40004]

for port in ports_to_try:
    try:
        print(f"\n🌐 포트 {port}로 D-Tale 시작 시도...")
        
        d = dtale.show(
            df, 
            name="Titanic Dataset",
            port=port,
            host='localhost'
        )
        
        print(f"✅ 포트 {port}에서 D-Tale이 시작되었습니다!")
        print(f"🌐 브라우저에서 다음 URL로 접속하세요: http://localhost:{port}")
        
        # 브라우저 열기
        time.sleep(2)
        webbrowser.open(f"http://localhost:{port}")
        
        print(f"✅ http://localhost:{port}에서 브라우저가 열렸습니다!")
        print(f"⏳ D-Tale이 포트 {port}에서 실행 중입니다. Ctrl+C를 눌러서 종료하세요.")
        
        # 성공하면 루프 종료
        break
        
    except Exception as e:
        print(f"❌ 포트 {port} 실패: {e}")
        continue

print("\n💡 만약 모든 포트가 실패하면 다음을 시도해보세요:")
print("1. 브라우저에서 직접 http://localhost:40000 접속")
print("2. http://127.0.0.1:40000 접속")
print("3. 다른 포트들: 40001, 40002, 40003, 40004")

# 무한 루프로 서버 유지
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n�� D-Tale을 종료합니다.") 