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

def load_titanic_data():
    """Titanic 데이터 로드"""
    print("=== Titanic 데이터 로드 ===")
    
    # Titanic 데이터 URL
    url = "https://autogluon.s3.amazonaws.com/datasets/Inc/titanic/train.csv"
    
    try:
        df = pd.read_csv(url)
        print(f"데이터 로드 완료: {df.shape}")
        return df
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        print("샘플 데이터 생성...")
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_samples = 1000
        
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
        
        df = pd.DataFrame(sample_data)
        print(f"샘플 데이터 생성 완료: {df.shape}")
        return df

def train_autogluon_model(df):
    """AutoGluon 모델 학습"""
    print("\n=== AutoGluon 모델 학습 ===")
    
    # 학습/테스트 분할
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"학습 데이터 크기: {train_df.shape}")
    print(f"테스트 데이터 크기: {test_df.shape}")
    
    # AutoGluon 모델 학습
    predictor = TabularPredictor(
        label='Survived',
        eval_metric='f1',
        verbosity=2
    ).fit(
        train_data=train_df,
        time_limit=60,  # 1분 제한
        presets='best_quality'
    )
    
    # 성능 평가
    train_score = predictor.evaluate(train_df)
    test_score = predictor.evaluate(test_df)
    
    print(f"학습 성능: {train_score}")
    print(f"테스트 성능: {test_score}")
    
    return predictor, test_df

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
            title="AutoGluon Titanic Model - SHAP Analysis",
            whatif=True,  # What-if 분석 활성화
            shap_interaction=True,  # SHAP interaction 활성화
            mode='inline'
        )
        
        print("✅ 대시보드 생성 완료!")
        print("대시보드를 실행합니다...")
        print("브라우저에서 http://localhost:8051 으로 접속하세요.")
        
        # 대시보드 실행 (다른 포트 사용)
        dashboard.run(port=8051, use_waitress=False)
        
    except Exception as e:
        print(f"❌ 대시보드 생성 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    print("=== AutoGluon Titanic 모델 SHAP 대시보드 생성 ===")
    
    try:
        # 1. 데이터 로드
        df = load_titanic_data()
        
        # 2. AutoGluon 모델 학습
        predictor, test_df = train_autogluon_model(df)
        
        # 3. 대시보드 생성
        create_dashboard(predictor, test_df)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 