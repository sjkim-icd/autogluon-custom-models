import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import warnings
warnings.filterwarnings('ignore')
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.datasets import load_breast_cancer, load_iris

# 1. sklearn 데이터 로드 및 모델 학습
def load_sklearn_data():
    """sklearn 데이터 로드"""
    print("=== sklearn 데이터 로드 ===")
    
    try:
        # Breast Cancer 데이터셋 사용 (이진 분류)
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        
        # 데이터프레임 생성
        df = pd.concat([X, y], axis=1)
        print(f"✅ Breast Cancer 데이터 로드 성공: {df.shape}")
        print(f"특성 수: {len(data.feature_names)}")
        print(f"타겟 분포: {df['target'].value_counts().to_dict()}")
        return df
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None

def train_sklearn_model(df):
    """sklearn 데이터로 모델 학습"""
    print("\n=== AutoGluon 모델 학습 ===")
    
    # 학습/테스트 분할
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"학습 데이터 크기: {train_df.shape}")
    print(f"테스트 데이터 크기: {test_df.shape}")
    
    # AutoGluon 모델 학습 (더 간단한 설정)
    predictor = TabularPredictor(
        label='target',
        eval_metric='f1',
        verbosity=2
    ).fit(
        train_data=train_df,
        time_limit=60,  # 60초로 증가
        presets='medium_quality',  # 더 빠른 프리셋
        dynamic_stacking=False,  # 동적 스태킹 비활성화
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
        label = "target"
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
            title="AutoGluon Breast Cancer Model - SHAP Analysis",
            whatif=True,  # What-if 분석 활성화
            shap_interaction=False,  # SHAP interaction 비활성화 (성능 향상)
            mode='inline'
        )
        
        print("✅ 대시보드 생성 완료!")
        print("대시보드를 실행합니다...")
        print("브라우저에서 http://localhost:8056 으로 접속하세요.")
        
        # 대시보드 실행 (다른 포트 사용)
        dashboard.run(port=8056, use_waitress=False)
        
    except Exception as e:
        print(f"❌ 대시보드 생성 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    print("=== sklearn 데이터로 AutoGluon 모델 학습 및 SHAP 대시보드 생성 ===")
    
    try:
        # 1. sklearn 데이터 로드
        df = load_sklearn_data()
        if df is None:
            print("데이터 로드 실패로 종료합니다.")
            return
        
        # 2. 모델 학습
        predictor, test_df = train_sklearn_model(df)
        
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