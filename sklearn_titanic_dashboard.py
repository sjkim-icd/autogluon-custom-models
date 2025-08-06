import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import warnings
warnings.filterwarnings('ignore')

def load_titanic_data():
    """Titanic 데이터 로드 및 전처리"""
    print("=== Titanic 데이터 로드 ===")
    
    # Titanic 데이터 URL
    url = "https://autogluon.s3.amazonaws.com/datasets/Inc/titanic/train.csv"
    
    try:
        df = pd.read_csv(url)
        print(f"데이터 로드 완료: {df.shape}")
        print(f"특성 목록: {list(df.columns)}")
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

def preprocess_data(df):
    """데이터 전처리"""
    print("\n=== 데이터 전처리 ===")
    
    # 필요한 특성만 선택
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features].copy()
    y = df['Survived']
    
    # 결측값 처리
    X['Age'].fillna(X['Age'].median(), inplace=True)
    X['Fare'].fillna(X['Fare'].median(), inplace=True)
    X['Embarked'].fillna('S', inplace=True)
    
    # 범주형 변수 인코딩
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    X['Sex'] = le_sex.fit_transform(X['Sex'])
    X['Embarked'] = le_embarked.fit_transform(X['Embarked'])
    
    print(f"전처리 완료: {X.shape}")
    print(f"특성 목록: {list(X.columns)}")
    print(f"타겟 분포: {np.bincount(y)}")
    
    return X, y

def train_model(X, y):
    """모델 학습"""
    print("\n=== 모델 학습 ===")
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # RandomForest 모델 학습
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 성능 평가
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"학습 정확도: {train_score:.4f}")
    print(f"테스트 정확도: {test_score:.4f}")
    
    return model, X_test, y_test

def create_dashboard(model, X, y):
    """대시보드 생성"""
    print("\n=== 대시보드 생성 ===")
    
    try:
        # ClassifierExplainer 생성
        explainer = ClassifierExplainer(
            model=model,
            X=X,
            y=y,
            model_output='probability'
        )
        print("✅ ClassifierExplainer 생성 완료!")
        
        # ExplainerDashboard 생성
        dashboard = ExplainerDashboard(
            explainer,
            title="Titanic Survival Prediction - SHAP Analysis",
            whatif=True,  # What-if 분석 활성화
            shap_interaction=True,  # SHAP interaction 활성화
            mode='inline'
        )
        
        print("✅ 대시보드 생성 완료!")
        print("대시보드를 실행합니다...")
        print("브라우저에서 http://localhost:8050 으로 접속하세요.")
        
        # 대시보드 실행
        dashboard.run(port=8050, use_waitress=False)
        
    except Exception as e:
        print(f"❌ 대시보드 생성 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    print("=== Titanic 데이터로 SHAP 대시보드 생성 ===")
    
    try:
        # 1. 데이터 로드
        df = load_titanic_data()
        
        # 2. 데이터 전처리
        X, y = preprocess_data(df)
        
        # 3. 모델 학습
        model, X_test, y_test = train_model(X, y)
        
        # 4. 대시보드 생성
        create_dashboard(model, X_test, y_test)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 