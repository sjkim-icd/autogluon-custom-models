import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import warnings
warnings.filterwarnings('ignore')

# AutoGluon 모델을 scikit-learn 호환 래퍼로 감싸기
class AutoGluonWrapper:
    def __init__(self, predictor):
        self.predictor = predictor
    
    def predict(self, X):
        return self.predictor.predict(X).values  # numpy array
    
    def predict_proba(self, X):
        return self.predictor.predict_proba(X).values  # numpy array

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

def create_sample_data():
    """샘플 데이터 생성"""
    print("\n=== 샘플 데이터 생성 ===")
    
    np.random.seed(42)
    n_samples = 500
    
    # Titanic 데이터셋 특성들 (모델이 기대하는 특성명으로)
    sample_data = {
        'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.25, 0.25, 0.5]),
        'name': [f"Passenger_{i}" for i in range(n_samples)],
        'sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'age': np.clip(np.random.normal(30, 15, n_samples), 0, 80),
        'sibsp': np.clip(np.random.poisson(0.5, n_samples), 0, 8),
        'parch': np.clip(np.random.poisson(0.4, n_samples), 0, 6),
        'ticket': [f"Ticket_{i}" for i in range(n_samples)],
        'fare': np.clip(np.random.exponential(50, n_samples), 10, 500),
        'cabin': [f"Cabin_{i}" if np.random.random() > 0.7 else np.nan for i in range(n_samples)],
        'embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1]),
        'boat': [f"Boat_{i}" if np.random.random() > 0.6 else np.nan for i in range(n_samples)],
        'body': [i if np.random.random() > 0.8 else np.nan for i in range(n_samples)],
        'home.dest': [f"Destination_{i}" if np.random.random() > 0.5 else np.nan for i in range(n_samples)],
        'Survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # 타겟 컬럼 추가
    }
    
    df = pd.DataFrame(sample_data)
    print(f"샘플 데이터 생성 완료: {df.shape}")
    return df

def create_dashboard(predictor, test_df):
    """대시보드 생성"""
    print("\n=== 대시보드 생성 ===")
    
    try:
        # 래퍼 생성
        wrapped_model = AutoGluonWrapper(predictor)
        print("✅ AutoGluon 래퍼 생성 완료!")
        
        # 데이터 준비
        X = test_df.drop('Survived', axis=1)
        y = test_df['Survived']
        
        print(f"테스트 데이터 준비: {X.shape}")
        print(f"타겟 분포: {np.bincount(y)}")
        
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
        
        # ClassifierExplainer 생성
        explainer = ClassifierExplainer(
            model=wrapped_model,
            X=X,
            y=y,
            model_output='probability'
        )
        print("✅ ClassifierExplainer 생성 완료!")
        
        # ExplainerDashboard 생성
        dashboard = ExplainerDashboard(
            explainer,
            title="AutoGluon Model - SHAP Analysis",
            whatif=True,  # What-if 분석 활성화
            shap_interaction=True,  # SHAP interaction 활성화
            mode='inline'
        )
        
        print("✅ 대시보드 생성 완료!")
        print("대시보드를 실행합니다...")
        print("브라우저에서 http://localhost:8052 으로 접속하세요.")
        
        # 대시보드 실행 (다른 포트 사용)
        dashboard.run(port=8052, use_waitress=False)
        
    except Exception as e:
        print(f"❌ 대시보드 생성 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    print("=== 기존 AutoGluon 모델 SHAP 대시보드 생성 ===")
    
    try:
        # 1. 기존 모델 로드
        predictor, model_path = load_existing_model()
        
        # 2. 모델 정보 출력
        print(f"\n모델 경로: {model_path}")
        leaderboard = predictor.leaderboard()
        print("리더보드:")
        print(leaderboard)
        
        # 3. 샘플 데이터 생성
        test_df = create_sample_data()
        
        # 4. 대시보드 생성
        create_dashboard(predictor, test_df)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 