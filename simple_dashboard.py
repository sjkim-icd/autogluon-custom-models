import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import shap
import matplotlib.pyplot as plt
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

def main():
    """간단한 대시보드"""
    print("=== Simple Dashboard 생성 ===")
    
    try:
        # 1. 모델 로드
        print("커스텀 모델 로드 중...")
        model_path = r"C:\Users\woori\Desktop\autogluon_env_cursor\models\CUSTOM_NN_TORCH_hpo"
        predictor = TabularPredictor.load(model_path)
        print("✅ 모델 로드 완료!")
        
        # 2. 샘플 데이터 생성
        print("\n=== 샘플 데이터 생성 ===")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Titanic 데이터셋 특성들
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
        
        # 가상의 타겟 생성
        predictions = predictor.predict_proba(X_sample)
        y_sample = (predictions.iloc[:, 1] > 0.5).astype(int)
        
        print(f"샘플 데이터 생성 완료: {X_sample.shape}")
        
        # 3. SHAP 분석
        print("\n=== SHAP 분석 ===")
        
        # 수치형 특성만 사용
        numeric_features = ['pclass', 'age', 'sibsp', 'parch', 'fare']
        X_numeric = X_sample[numeric_features].copy()
        
        # SHAP Explainer 생성
        def model_predict(X):
            if isinstance(X, np.ndarray):
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
        
        # 백그라운드 데이터
        background_data = X_numeric.sample(min(50, len(X_numeric)), random_state=42)
        
        explainer = shap.KernelExplainer(
            model=model_predict,
            data=background_data,
            link="identity"
        )
        
        # SHAP 값 계산
        sample_for_shap = X_numeric.sample(min(20, len(X_numeric)), random_state=42)
        shap_values = explainer.shap_values(sample_for_shap)
        
        print(f"✅ SHAP 값 계산 완료! 형태: {np.array(shap_values).shape}")
        
        # 4. 시각화 생성
        print("\n=== 시각화 생성 ===")
        
        # SHAP 요약 플롯
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, sample_for_shap, show=False)
        plt.tight_layout()
        plt.savefig('dashboard_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ SHAP 요약 플롯 저장 완료")
        
        # 특성 중요도 차트
        feature_importance = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame({
            'feature': numeric_features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('SHAP Importance')
        plt.title('Feature Importance (SHAP)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('dashboard_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 특성 중요도 차트 저장 완료")
        
        # 개별 샘플 SHAP 플롯
        plt.figure(figsize=(10, 6))
        shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            sample_for_shap.iloc[0],
            show=False
        )
        plt.tight_layout()
        plt.savefig('dashboard_individual_shap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 개별 샘플 SHAP 플롯 저장 완료")
        
        # 5. 결과 출력
        print("\n=== 분석 결과 ===")
        print("특성 중요도 (SHAP):")
        print(importance_df)
        
        print(f"\n모델 성능: F1 Score = {predictor.leaderboard().iloc[0]['score_val']:.4f}")
        
        print("\n=== 생성된 파일 ===")
        print("- dashboard_shap_summary.png: SHAP 요약 플롯")
        print("- dashboard_feature_importance.png: 특성 중요도 차트")
        print("- dashboard_individual_shap.png: 개별 샘플 분석")
        
        # 6. 간단한 HTML 대시보드 생성
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AutoGluon Model SHAP Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .feature-importance {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AutoGluon Custom Model SHAP Analysis</h1>
                    <p>모델 성능: F1 Score = {predictor.leaderboard().iloc[0]['score_val']:.4f}</p>
                </div>
                
                <div class="section">
                    <h2>특성 중요도 (SHAP)</h2>
                    <div class="feature-importance">
                        <table>
                            <tr><th>특성</th><th>중요도</th></tr>
                            {''.join([f'<tr><td>{row["feature"]}</td><td>{row["importance"]:.4f}</td></tr>' for _, row in importance_df.iterrows()])}
                        </table>
                    </div>
                </div>
                
                <div class="section">
                    <h2>시각화</h2>
                    <p>다음 이미지 파일들을 확인하세요:</p>
                    <ul>
                        <li><strong>dashboard_shap_summary.png</strong>: SHAP 요약 플롯</li>
                        <li><strong>dashboard_feature_importance.png</strong>: 특성 중요도 차트</li>
                        <li><strong>dashboard_individual_shap.png</strong>: 개별 샘플 분석</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>주요 인사이트</h2>
                    <ul>
                        <li>가장 중요한 특성: <strong>{importance_df.iloc[0]['feature']}</strong> (중요도: {importance_df.iloc[0]['importance']:.4f})</li>
                        <li>두 번째 중요한 특성: <strong>{importance_df.iloc[1]['feature']}</strong> (중요도: {importance_df.iloc[1]['importance']:.4f})</li>
                        <li>세 번째 중요한 특성: <strong>{importance_df.iloc[2]['feature']}</strong> (중요도: {importance_df.iloc[2]['importance']:.4f})</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open('dashboard_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("✅ HTML 대시보드 생성 완료: dashboard_report.html")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 