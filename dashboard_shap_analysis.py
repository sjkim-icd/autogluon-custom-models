import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import warnings
warnings.filterwarnings('ignore')

def main():
    """대시보드 SHAP 분석"""
    print("=== Dashboard SHAP 분석 ===")
    
    try:
        # 1. 모델 로드
        print("커스텀 모델 로드 중...")
        model_path = r"C:\Users\woori\Desktop\autogluon_env_cursor\models\CUSTOM_NN_TORCH_hpo"
        predictor = TabularPredictor.load(model_path)
        print("✅ 모델 로드 완료!")
        
        # 2. 샘플 데이터 생성 (Titanic 데이터셋)
        print("\n=== 샘플 데이터 생성 ===")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Titanic 데이터셋 특성들 (모든 특성 포함)
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
            'home.dest': [f"Destination_{i}" if np.random.random() > 0.5 else np.nan for i in range(n_samples)]
        }
        
        X_sample = pd.DataFrame(sample_data)
        
        # 가상의 타겟 생성 (모델 예측 기반)
        predictions = predictor.predict_proba(X_sample)
        y_sample = (predictions.iloc[:, 1] > 0.5).astype(int)
        
        print(f"샘플 데이터 생성 완료: {X_sample.shape}")
        print(f"특성 목록: {list(X_sample.columns)}")
        print(f"타겟 분포: {np.bincount(y_sample)}")
        
        # 3. ClassifierExplainer 생성 (수치형 특성만 사용)
        print("\n=== ClassifierExplainer 생성 ===")
        
        # 수치형 특성만 선택
        numeric_features = ['pclass', 'age', 'sibsp', 'parch', 'fare']
        X_numeric = X_sample[numeric_features].copy()
        
        try:
            # AutoGluon 모델을 위한 예측 함수 래퍼
            def model_predict(X):
                if isinstance(X, np.ndarray):
                    # 수치형 특성만 있으면 나머지는 기본값으로 채움
                    X_df = pd.DataFrame(X, columns=numeric_features)
                    # 나머지 특성들을 기본값으로 추가
                    for col in X_sample.columns:
                        if col not in numeric_features:
                            if col in ['sex', 'embarked']:
                                X_df[col] = 'unknown'
                            elif col in ['name', 'ticket', 'cabin', 'boat', 'home.dest']:
                                X_df[col] = 'unknown'
                            elif col == 'body':
                                X_df[col] = np.nan
                else:
                    X_df = X
                proba = predictor.predict_proba(X_df)
                return proba.iloc[:, 1].values
            
            # ClassifierExplainer 생성
            explainer = ClassifierExplainer(
                model=model_predict,
                X=X_numeric,
                y=y_sample,
                model_output='probability'
            )
            print("✅ ClassifierExplainer 생성 완료!")
            
        except Exception as e:
            print(f"❌ ClassifierExplainer 생성 실패: {e}")
            print("대안 방법 시도...")
            
            # 대안: 더 간단한 방법
            try:
                explainer = ClassifierExplainer(
                    model=lambda x: predictor.predict_proba(pd.DataFrame(x, columns=numeric_features)).iloc[:, 1].values,
                    X=X_numeric,
                    y=y_sample,
                    model_output='probability'
                )
                print("✅ ClassifierExplainer 생성 완료! (대안 방법)")
            except Exception as e2:
                print(f"❌ 대안 방법도 실패: {e2}")
                return
        
        # 4. 대시보드 생성 및 실행
        print("\n=== 대시보드 생성 ===")
        
        try:
            # ExplainerDashboard 생성
            dashboard = ExplainerDashboard(
                explainer,
                title="AutoGluon Custom Model SHAP Analysis",
                whatif=False,  # What-if 분석 비활성화
                shap_interaction=False,  # SHAP interaction 비활성화
                mode='inline'  # 인라인 모드로 실행
            )
            
            print("✅ 대시보드 생성 완료!")
            print("대시보드를 실행합니다...")
            print("브라우저에서 http://localhost:8050 으로 접속하세요.")
            
            # 대시보드 실행
            dashboard.run(port=8050, use_waitress=False)
            
        except Exception as e:
            print(f"❌ 대시보드 실행 실패: {e}")
            print("대안: 간단한 SHAP 분석만 실행...")
            
            # 간단한 SHAP 분석
            shap_values = explainer.shap_values()
            print(f"SHAP 값 계산 완료: {shap_values.shape}")
            
            # 특성 중요도 출력
            feature_importance = explainer.feature_importance()
            print("\n=== 특성 중요도 ===")
            print(feature_importance.head(10))
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 