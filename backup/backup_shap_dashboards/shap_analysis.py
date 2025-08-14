import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import warnings
warnings.filterwarnings('ignore')

def load_data_and_model():
    """데이터와 학습된 모델을 로드합니다."""
    print("=== 데이터 및 모델 로드 ===")
    
    # 데이터 로드
    df = pd.read_csv('creditcard.csv')
    print(f"데이터 크기: {df.shape}")
    print(f"클래스 분포:\n{df['Class'].value_counts()}")
    
    # 학습/테스트 분할
    train_df = df.iloc[:227845]
    test_df = df.iloc[227845:]
    
    print(f"학습 데이터 크기: {train_df.shape}")
    print(f"테스트 데이터 크기: {test_df.shape}")
    
    # 모델 로드
    predictor = TabularPredictor.load("models/five_models_experiment")
    print("모델 로드 완료!")
    
    return train_df, test_df, predictor

def create_shap_explainer(train_df, test_df, predictor):
    """SHAP Explainer를 생성합니다."""
    print("\n=== SHAP Explainer 생성 ===")
    
    # 특성 데이터 준비
    X_train = train_df.drop('Class', axis=1)
    X_test = test_df.drop('Class', axis=1)
    y_train = train_df['Class']
    y_test = test_df['Class']
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # ClassifierExplainer 생성
    explainer = ClassifierExplainer(
        model=predictor,
        X=X_test,  # SHAP 계산을 위한 백그라운드 데이터
        y=y_test,
        model_output='probability'  # 확률 출력
    )
    
    print("SHAP Explainer 생성 완료!")
    return explainer

def run_shap_analysis(explainer):
    """SHAP 분석을 실행합니다."""
    print("\n=== SHAP 분석 실행 ===")
    
    # SHAP 값 계산
    print("SHAP 값 계산 중...")
    shap_values = explainer.shap_values()
    print(f"SHAP 값 형태: {shap_values.shape}")
    
    # 특성 중요도 확인
    feature_importance = explainer.feature_importance()
    print("\n=== 특성 중요도 (SHAP) ===")
    print(feature_importance.head(10))
    
    return explainer

def create_dashboard(explainer):
    """대시보드를 생성하고 실행합니다."""
    print("\n=== 대시보드 생성 ===")
    
    # ExplainerDashboard 생성
    dashboard = ExplainerDashboard(
        explainer,
        title="AutoGluon Ensemble Model SHAP Analysis",
        whatif=False,  # What-if 분석 비활성화 (성능 향상)
        shap_interaction=False,  # SHAP interaction 비활성화
        mode='inline'  # 인라인 모드로 실행
    )
    
    print("대시보드 생성 완료!")
    print("대시보드를 실행합니다...")
    
    # 대시보드 실행
    dashboard.run(port=8050, use_waitress=False)
    
    return dashboard

def main():
    """메인 함수"""
    print("=== AutoGluon Ensemble Model SHAP 분석 ===")
    
    try:
        # 1. 데이터 및 모델 로드
        train_df, test_df, predictor = load_data_and_model()
        
        # 2. SHAP Explainer 생성
        explainer = create_shap_explainer(train_df, test_df, predictor)
        
        # 3. SHAP 분석 실행
        explainer = run_shap_analysis(explainer)
        
        # 4. 대시보드 생성 및 실행
        dashboard = create_dashboard(explainer)
        
        print("\n=== 분석 완료! ===")
        print("대시보드가 http://localhost:8050 에서 실행 중입니다.")
        print("브라우저에서 해당 주소로 접속하여 SHAP 분석 결과를 확인하세요.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 