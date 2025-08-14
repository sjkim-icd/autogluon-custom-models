import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import dtale
import os
import warnings
warnings.filterwarnings('ignore')

# 타이타닉 데이터 로드
print("🚢 타이타닉 데이터 로딩 중...")
titanic = fetch_openml(name='titanic', version=1, as_frame=True)
df = titanic.frame

print(f"📊 데이터 형태: {df.shape}")

# ============================================================================
# D-TALE + NGROK 적용
# ============================================================================
print("\n" + "="*60)
print("🌐 D-TALE + NGROK 적용")
print("="*60)

# ngrok import 및 설정
try:
    from pyngrok import ngrok
    import subprocess
    import time
    
    print("🔧 ngrok 설정 중...")
    
    # ngrok authtoken 설정
    auth_token = "2p7DLlnWujpi4eay9sYyr4ZDyYY_7i5Wgfqcj6Ekp5eax7E6M"
    
    try:
        # ngrok authtoken 설정
        ngrok.set_auth_token(auth_token)
        print("✅ ngrok authtoken 설정 완료")
    except Exception as e:
        print(f"⚠️ ngrok authtoken 설정 실패: {e}")
        print("💡 수동으로 ngrok authtoken을 설정하세요.")
    
    # 기존 ngrok 프로세스 종료 (Windows)
    try:
        subprocess.run(['taskkill', '/f', '/im', 'ngrok.exe'], 
                      capture_output=True, check=False)
        print("✅ 기존 ngrok 프로세스 종료")
    except:
        pass
    
    # D-Tale 실행 (0.0.0.0으로 설정하여 외부 접속 허용)
    print("🌐 D-Tale 시작 중...")
    d = dtale.show(df, name="Titanic Dataset", host="0.0.0.0", port=4000)
    
    print("✅ D-Tale이 시작되었습니다!")
    print(f"🌐 로컬 URL: http://localhost:4000")
    
    # ngrok을 통해 포트 터널링
    print("🔗 ngrok 터널 생성 중...")
    public_url = ngrok.connect(4000)
    
    print("✅ ngrok 터널이 생성되었습니다!")
    print(f"🌐 외부 접속 URL: {public_url}")
    print(f"💡 위 URL을 복사해서 브라우저에서 접속하세요!")
    
    # 브라우저 자동 열기
    import webbrowser
    time.sleep(2)
    try:
        webbrowser.open(public_url)
        print("🌐 브라우저를 자동으로 열었습니다!")
    except:
        print("❌ 브라우저 자동 열기 실패. 수동으로 접속하세요.")
    
    print("\n" + "="*60)
    print("📋 접속 URL 요약:")
    print(f"• 로컬 접속: http://localhost:4000")
    print(f"• 외부 접속: {public_url}")
    print("="*60)
    
    print("\n⏳ D-Tale이 실행 중입니다. Ctrl+C를 눌러서 종료하세요.")
    
    # 무한 루프로 서버 유지
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 D-Tale을 종료합니다.")
        # ngrok 터널 종료
        ngrok.kill()
        print("✅ ngrok 터널을 종료했습니다.")

except ImportError:
    print("❌ pyngrok이 설치되지 않았습니다.")
    print("💡 다음 명령어로 설치하세요: pip install pyngrok")
    
    # ngrok 없이 D-Tale만 실행
    print("\n🌐 D-Tale만 실행합니다...")
    d = dtale.show(df, name="Titanic Dataset", port=4000, host='localhost')
    
    print("✅ D-Tale이 시작되었습니다!")
    print(f"🌐 브라우저에서 다음 URL로 접속하세요: http://localhost:4000")
    
    # 브라우저 자동 열기
    import webbrowser
    import time
    time.sleep(2)
    try:
        webbrowser.open("http://localhost:4000")
        print("🌐 브라우저를 자동으로 열었습니다!")
    except:
        print("❌ 브라우저 자동 열기 실패. 수동으로 접속하세요.")
    
    print("\n⏳ D-Tale이 실행 중입니다. Ctrl+C를 눌러서 종료하세요.")
    
    # 무한 루프로 서버 유지
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n�� D-Tale을 종료합니다.") 