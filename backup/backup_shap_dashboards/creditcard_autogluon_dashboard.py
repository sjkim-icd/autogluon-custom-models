from autogluon.tabular import TabularPredictor, TabularDataset
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. AutoGluon 모델 로드 (기존 모델 사용)
def load_existing_model():
    """기존 AutoGluon 모델 로드"""
    print("=== 기존 AutoGluon 모델 로드 ===")
    
    # 기존 모델 경로들
    model_paths = [
        r"C:\Users\woori\Desktop\autogluon_env_cursor\models\CUSTOM_NN_TORCH_hpo",
        r"C:\Users\woori\Desktop\autogluon_env_cursor\models\five_models_experiment"
    ]
    
    for path in model_paths:
        try:
            predictor = TabularPredictor.load(path)
            print(f"✅ 모델 로드 성공: {path}")
            return predictor, path
        except Exception as e:
            print(f"❌ 모델 로드 실패: {path} - {e}")
    
    raise Exception("사용 가능한 모델을 찾을 수 없습니다.")

# 2. 래퍼 클래스 (explainerdashboard가 요구하는 API에 맞추기)
class AutoGluonWrapper:
    def __init__(self, predictor):
        self.predictor = predictor
    
    def predict(self, X):
        return self.predictor.predict(X).values  # numpy array
    
    def predict_proba(self, X):
        return self.predictor.predict_proba(X).values  # numpy array

# 3. Credit Card Fraud 데이터 생성
def create_creditcard_data():
    """Credit Card Fraud 데이터 생성"""
    print("\n=== Credit Card Fraud 데이터 생성 ===")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Credit Card Fraud 데이터셋 특성들
    feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # 데이터 생성
    data = {}
    
    # Time (시간)
    data['Time'] = np.random.exponential(1000, n_samples)
    
    # V1-V28 (PCA 변환된 특성들)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Amount (거래 금액)
    data['Amount'] = np.random.exponential(100, n_samples)
    
    # Class (타겟 - 대부분 정상 거래, 일부 사기)
    data['Class'] = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    
    df = pd.DataFrame(data)
    print(f"✅ 데이터 생성 완료: {df.shape}")
    print(f"특성 목록: {list(df.columns)}")
    print(f"타겟 분포: {df['Class'].value_counts().to_dict()}")
    
    return df

def create_dashboard(predictor, data):
    """대시보드 생성"""
    print("\n=== 대시보드 생성 ===")
    
    try:
        # 데이터 준비
        label = "Class"
        X = data.drop(columns=[label])
        y = data[label]
        
        print(f"특성 데이터 크기: {X.shape}")
        print(f"타겟 데이터 크기: {y.shape}")
        
        # 래퍼 생성
        wrapped_model = AutoGluonWrapper(predictor)
        print("✅ AutoGluon 래퍼 생성 완료!")
        
        # 모델 예측 테스트
        try:
            predictions = wrapped_model.predict(X.head(5))
            proba = wrapped_model.predict_proba(X.head(5))
            print("✅ 모델 예측 테스트 성공!")
            print(f"예측 형태: {predictions.shape}")
            print(f"확률 예측 형태: {proba.shape}")
        except Exception as e:
            print(f"❌ 모델 예측 테스트 실패: {e}")
            return
        
        # 4. ExplainerDashboard 연결
        print("ClassifierExplainer 생성 중...")
        explainer = ClassifierExplainer(
            model=wrapped_model,
            X=X,
            y=y,
            model_output='probability'
        )
        print("✅ ClassifierExplainer 생성 완료!")
        
        # 대시보드 생성 및 실행
        print("ExplainerDashboard 생성 중...")
        dashboard = ExplainerDashboard(
            explainer,
            title="AutoGluon Credit Card Fraud Model - SHAP Analysis",
            whatif=True,  # What-if 분석 활성화
            shap_interaction=False,  # SHAP interaction 비활성화 (성능 향상)
            mode='inline'
        )
        
        print("✅ 대시보드 생성 완료!")
        print("대시보드를 실행합니다...")
        print("브라우저에서 http://localhost:8054 으로 접속하세요.")
        
        # 대시보드 실행 (다른 포트 사용)
        dashboard.run(port=8054, use_waitress=False)
        
    except Exception as e:
        print(f"❌ 대시보드 생성 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    print("=== AutoGluon Credit Card Fraud 모델 SHAP 대시보드 생성 ===")
    
    try:
        # 1. 기존 모델 로드
        predictor, model_path = load_existing_model()
        
        # 2. 모델 정보 출력
        print(f"\n모델 경로: {model_path}")
        leaderboard = predictor.leaderboard()
        print("리더보드:")
        print(leaderboard)
        
        # 3. Credit Card Fraud 데이터 생성
        data = create_creditcard_data()
        
        # 4. 대시보드 생성
        create_dashboard(predictor, data)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 