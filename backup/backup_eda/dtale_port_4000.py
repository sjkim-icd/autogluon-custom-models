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

# 포트 4000으로 D-Tale 시도
print("\n🌐 포트 4000으로 D-Tale 시작 시도...")

try:
    d = dtale.show(
        df, 
        name="Titanic Dataset",
        port=4000,
        host='localhost'
    )
    
    print(f"✅ 포트 4000에서 D-Tale이 시작되었습니다!")
    print(f"🌐 브라우저에서 다음 URL로 접속하세요: http://localhost:4000")
    
    # 브라우저 열기
    time.sleep(2)
    webbrowser.open("http://localhost:4000")
    
    print(f"✅ http://localhost:4000에서 브라우저가 열렸습니다!")
    print(f"⏳ D-Tale이 포트 4000에서 실행 중입니다. Ctrl+C를 눌러서 종료하세요.")
    
except Exception as e:
    print(f"❌ 포트 4000 실패: {e}")
    
    # 다른 포트들도 시도
    ports_to_try = [4000, 40000, 40001, 40002, 40003]
    
    for port in ports_to_try:
        if port == 4000:  # 이미 시도했으므로 건너뛰기
            continue
            
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

print("\n💡 접속 URL들:")
print("1. http://localhost:4000")
print("2. http://127.0.0.1:4000")
print("3. http://localhost:40000")
print("4. http://127.0.0.1:40000")

# 무한 루프로 서버 유지
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n�� D-Tale을 종료합니다.") 