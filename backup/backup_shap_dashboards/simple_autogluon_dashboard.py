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

# 3. Titanic 데이터 로드 및 전처리
def load_titanic_data():
    """Titanic 데이터 로드"""
    print("\n=== Titanic 데이터 로드 ===")
    
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

def preprocess_data(train_data):
    """데이터 전처리"""
    print("\n=== 데이터 전처리 ===")
    
    # 필요한 특성만 선택 (모델이 기대하는 특성명으로)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    # 특성명을 소문자로 변경 (모델이 기대하는 형식)
    train_data_renamed = train_data.copy()
    train_data_renamed.columns = [col.lower() for col in train_data_renamed.columns]
    
    # 누락된 특성들 추가 (모델이 기대하는 특성들)
    if 'boat' not in train_data_renamed.columns:
        train_data_renamed['boat'] = [f"Boat_{i}" if np.random.random() > 0.6 else np.nan for i in range(len(train_data_renamed))]
    
    if 'body' not in train_data_renamed.columns:
        train_data_renamed['body'] = [i if np.random.random() > 0.8 else np.nan for i in range(len(train_data_renamed))]
    
    if 'home.dest' not in train_data_renamed.columns:
        train_data_renamed['home.dest'] = [f"Destination_{i}" if np.random.random() > 0.5 else np.nan for i in range(len(train_data_renamed))]
    
    # 결측값 처리
    train_data_renamed['age'].fillna(train_data_renamed['age'].median(), inplace=True)
    train_data_renamed['fare'].fillna(train_data_renamed['fare'].median(), inplace=True)
    train_data_renamed['embarked'].fillna('S', inplace=True)
    
    # 범주형 변수 처리 (NaN 값을 문자열로 변경)
    categorical_cols = ['sex', 'embarked', 'cabin', 'boat', 'home.dest']
    for col in categorical_cols:
        if col in train_data_renamed.columns:
            train_data_renamed[col] = train_data_renamed[col].fillna('unknown')
            train_data_renamed[col] = train_data_renamed[col].astype(str)
    
    # 수치형 변수 처리
    numerical_cols = ['pclass', 'age', 'sibsp', 'parch', 'fare']
    for col in numerical_cols:
        if col in train_data_renamed.columns:
            train_data_renamed[col] = pd.to_numeric(train_data_renamed[col], errors='coerce')
            train_data_renamed[col] = train_data_renamed[col].fillna(train_data_renamed[col].median())
    
    print(f"전처리 완료: {train_data_renamed.shape}")
    print(f"특성 목록: {list(train_data_renamed.columns)}")
    print(f"타겟 분포: {train_data_renamed['survived'].value_counts().to_dict()}")
    
    return train_data_renamed

def create_dashboard(predictor, train_data):
    """대시보드 생성"""
    print("\n=== 대시보드 생성 ===")
    
    try:
        # 데이터 준비
        label = "survived"
        X = train_data.drop(columns=[label])
        y = train_data[label]
        
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
            shap_interaction=True,  # SHAP interaction 활성화
            mode='inline'
        )
        
        print("✅ 대시보드 생성 완료!")
        print("대시보드를 실행합니다...")
        print("브라우저에서 http://localhost:8053 으로 접속하세요.")
        
        # 대시보드 실행 (다른 포트 사용)
        dashboard.run(port=8053, use_waitress=False)
        
    except Exception as e:
        print(f"❌ 대시보드 생성 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    print("=== AutoGluon Titanic 모델 SHAP 대시보드 생성 ===")
    
    try:
        # 1. 기존 모델 로드
        predictor, model_path = load_existing_model()
        
        # 2. 모델 정보 출력
        print(f"\n모델 경로: {model_path}")
        leaderboard = predictor.leaderboard()
        print("리더보드:")
        print(leaderboard)
        
        # 3. Titanic 데이터 로드
        train_data = load_titanic_data()
        
        # 4. 데이터 전처리
        train_data_processed = preprocess_data(train_data)
        
        # 5. 대시보드 생성
        create_dashboard(predictor, train_data_processed)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 