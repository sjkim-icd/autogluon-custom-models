import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import shap
import warnings
warnings.filterwarnings('ignore')

def main():
    """간단한 SHAP 분석"""
    print("=== AutoGluon Ensemble Model SHAP 분석 ===")
    
    try:
        # 1. 데이터 로드
        print("데이터 로드 중...")
        df = pd.read_csv('creditcard.csv')
        train_df = df.iloc[:227845]
        test_df = df.iloc[227845:]
        
        X_test = test_df.drop('Class', axis=1)
        y_test = test_df['Class']
        
        print(f"테스트 데이터 크기: {X_test.shape}")
        
        # 2. 모델 로드
        print("모델 로드 중...")
        predictor = TabularPredictor.load("models/five_models_experiment")
        
        # 3. SHAP 분석
        print("SHAP 분석 시작...")
        
        # 백그라운드 데이터로 SHAP 계산
        background_data = X_test.sample(min(100, len(X_test)), random_state=42)
        
        # SHAP Explainer 생성
        explainer = shap.TreeExplainer(predictor) if hasattr(predictor, 'predict_proba') else shap.KernelExplainer(
            model=lambda x: predictor.predict_proba(x)[:, 1], 
            data=background_data.values,
            link="identity"
        )
        
        # SHAP 값 계산
        print("SHAP 값 계산 중...")
        shap_values = explainer.shap_values(X_test.values)
        
        print(f"SHAP 값 형태: {np.array(shap_values).shape}")
        
        # 특성 중요도 계산
        feature_importance = np.abs(shap_values).mean(0)
        feature_names = X_test.columns
        
        # 상위 10개 특성 출력
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\n=== 상위 10개 특성 중요도 (SHAP) ===")
        print(importance_df.head(10))
        
        # SHAP 요약 플롯 저장
        print("SHAP 요약 플롯 생성 중...")
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("SHAP 요약 플롯이 'shap_summary_plot.png'에 저장되었습니다.")
        
        # 개별 샘플 SHAP 플롯 (첫 번째 샘플)
        print("개별 샘플 SHAP 플롯 생성 중...")
        plt.figure(figsize=(10, 6))
        shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            X_test.iloc[0],
            show=False
        )
        plt.tight_layout()
        plt.savefig('shap_individual_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("개별 샘플 SHAP 플롯이 'shap_individual_plot.png'에 저장되었습니다.")
        
        print("\n=== SHAP 분석 완료! ===")
        print("생성된 파일:")
        print("- shap_summary_plot.png: 전체 특성 중요도")
        print("- shap_individual_plot.png: 개별 샘플 분석")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 