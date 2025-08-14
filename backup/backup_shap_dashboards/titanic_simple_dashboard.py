from autogluon.tabular import TabularPredictor, TabularDataset
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. Titanic 데이터 로드 및 모델 학습
def load_titanic_data():
    """Titanic 데이터 로드"""
    print("=== Titanic 데이터 로드 ===")
    
    try:
        # 실제 Titanic 데이터 로드
        url = "https://autogluon.s3.amazonaws.com/datasets/Inc/titanic/train.csv"
        train_data = pd.read_csv(url)
        print(f"✅ 데이터 로드 성공: {train_data.shape}")
        return train_data
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        print("샘플 데이터 생성...")
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_samples = 500
        
        sample_data = {
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.25, 0.25, 0.5]),
            'Name': [f"Passenger_{i}" for i in range(n_samples)],
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
            'Age': np.clip(np.random.normal(30, 15, n_samples), 0, 80),
            'SibSp': np.clip(np.random.poisson(0.5, n_samples), 0, 8),
            'Parch': np.clip(np.random.poisson(0.4, n_samples), 0, 6),
            'Ticket': [f"Ticket_{i}" for i in range(n_samples)],
            'Fare': np.clip(np.random.exponential(50, n_samples), 10, 500),
            'Cabin': [f"Cabin_{i}" if np.random.random() > 0.7 else np.nan for i in range(n_samples)],
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1])
        }
        
        train_data = pd.DataFrame(sample_data)
        print(f"✅ 샘플 데이터 생성 완료: {train_data.shape}")
        return train_data

def train_titanic_model(train_data):
    """Titanic 모델 학습"""
    print("\n=== Titanic 모델 학습 ===")
    
    # 학습/테스트 분할
    train_size = int(0.8 * len(train_data))
    train_df = train_data.iloc[:train_size]
    test_df = train_data.iloc[train_size:]
    
    print(f"학습 데이터 크기: {train_df.shape}")
    print(f"테스트 데이터 크기: {test_df.shape}")
    
    # AutoGluon 모델 학습 (빠른 학습을 위해 시간 제한)
    predictor = TabularPredictor(
        label='Survived',
        eval_metric='f1',
        verbosity=2
    ).fit(
        train_data=train_df,
        time_limit=30,  # 30초 제한
        presets='best_quality',
        raise_on_no_models_fitted=False  # 모델 학습 실패해도 계속 진행
    )
    
    # 성능 평가
    try:
        train_score = predictor.evaluate(train_df)
        test_score = predictor.evaluate(test_df)
        print(f"학습 성능: {train_score}")
        print(f"테스트 성능: {test_score}")
    except Exception as e:
        print(f"성능 평가 실패: {e}")
    
    return predictor, test_df

# 2. 래퍼 클래스 (explainerdashboard가 요구하는 API에 맞추기)
class AutoGluonWrapper:
    def __init__(self, predictor):
        self.predictor = predictor
    
    def predict(self, X):
        return self.predictor.predict(X).values  # numpy array
    
    def predict_proba(self, X):
        return self.predictor.predict_proba(X).values  # numpy array

def create_dashboard(predictor, test_df):
    """대시보드 생성"""
    print("\n=== 대시보드 생성 ===")
    
    try:
        # 데이터 준비
        label = "Survived"
        X = test_df.drop(columns=[label])
        y = test_df[label]
        
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
            title="AutoGluon Titanic Model - SHAP Analysis",
            whatif=True,  # What-if 분석 활성화
            shap_interaction=False,  # SHAP interaction 비활성화 (성능 향상)
            mode='inline'
        )
        
        print("✅ 대시보드 생성 완료!")
        print("대시보드를 실행합니다...")
        print("브라우저에서 http://localhost:8055 으로 접속하세요.")
        
        # 대시보드 실행 (다른 포트 사용)
        dashboard.run(port=8055, use_waitress=False)
        
    except Exception as e:
        print(f"❌ 대시보드 생성 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    print("=== Titanic 데이터로 AutoGluon 모델 학습 및 SHAP 대시보드 생성 ===")
    
    try:
        # 1. Titanic 데이터 로드
        train_data = load_titanic_data()
        
        # 2. 모델 학습
        predictor, test_df = train_titanic_model(train_data)
        
        # 3. 모델 정보 출력
        try:
            leaderboard = predictor.leaderboard()
            print("\n리더보드:")
            print(leaderboard)
        except Exception as e:
            print(f"리더보드 로드 실패: {e}")
        
        # 4. 대시보드 생성
        create_dashboard(predictor, test_df)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 