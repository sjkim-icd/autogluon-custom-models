import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def permutation_importance(predictor, X, y, n_repeats=5):
    """순열 중요도 계산"""
    from sklearn.inspection import permutation_importance
    
    # 예측 함수 래퍼
    def predict_wrapper(X):
        return predictor.predict_proba(X)[:, 1]
    
    # 순열 중요도 계산
    result = permutation_importance(
        estimator=None,
        X=X,
        y=y,
        scoring=predict_wrapper,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=1
    )
    
    return result

def main():
    """특성 중요도 분석"""
    print("=== Custom Model 특성 중요도 분석 ===")
    
    try:
        # 1. 모델 로드
        print("커스텀 모델 로드 중...")
        model_path = r"C:\Users\woori\Desktop\autogluon_env_cursor\models\CUSTOM_NN_TORCH_hpo"
        predictor = TabularPredictor.load(model_path)
        print("✅ 모델 로드 완료!")
        
        # 2. 모델 정보 확인
        print("\n=== 모델 정보 ===")
        leaderboard = predictor.leaderboard()
        print("리더보드:")
        print(leaderboard)
        
        # 3. 샘플 데이터 생성 (Titanic 데이터셋)
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
        
        # 가상의 타겟 생성 (모델 예측 기반)
        predictions = predictor.predict_proba(X_sample)
        y_sample = (predictions[:, 1] > 0.5).astype(int)
        
        print(f"샘플 데이터 생성 완료: {X_sample.shape}")
        print(f"타겟 분포: {np.bincount(y_sample)}")
        
        # 4. 특성 중요도 분석
        print("\n=== 특성 중요도 분석 ===")
        
        # 수치형 특성만 선택
        numeric_features = ['pclass', 'age', 'sibsp', 'parch', 'fare']
        X_numeric = X_sample[numeric_features].copy()
        
        # 결측값 처리
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        print("수치형 특성으로 중요도 계산 중...")
        
        # 순열 중요도 계산
        try:
            from sklearn.inspection import permutation_importance
            from sklearn.ensemble import RandomForestClassifier
            
            # 간단한 랜덤 포레스트로 중요도 계산
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_numeric, y_sample)
            
            # 특성 중요도
            feature_importance = rf.feature_importances_
            
            # 결과 정리
            importance_df = pd.DataFrame({
                'feature': numeric_features,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print("\n=== 특성 중요도 (Random Forest) ===")
            print(importance_df)
            
        except Exception as e:
            print(f"순열 중요도 계산 실패: {e}")
            
            # 대안: 상관관계 기반 중요도
            print("상관관계 기반 중요도 계산...")
            
            correlations = []
            for feature in numeric_features:
                corr = np.corrcoef(X_numeric[feature], y_sample)[0, 1]
                correlations.append(abs(corr))
            
            importance_df = pd.DataFrame({
                'feature': numeric_features,
                'importance': correlations
            }).sort_values('importance', ascending=False)
            
            print("\n=== 특성 중요도 (상관관계) ===")
            print(importance_df)
        
        # 5. 시각화
        print("\n=== 시각화 생성 ===")
        
        try:
            # 특성 중요도 바 차트
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Importance')
            plt.title('Feature Importance Analysis')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('custom_model_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 특성 중요도 차트가 'custom_model_feature_importance.png'에 저장되었습니다.")
            
            # 특성별 분포 시각화
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, feature in enumerate(numeric_features[:6]):
                axes[i].hist(X_numeric[feature], bins=20, alpha=0.7)
                axes[i].set_title(f'{feature} Distribution')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('custom_model_feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 특성 분포 차트가 'custom_model_feature_distributions.png'에 저장되었습니다.")
            
        except Exception as e:
            print(f"❌ 시각화 생성 실패: {e}")
        
        # 6. 모델 성능 분석
        print("\n=== 모델 성능 분석 ===")
        
        # 예측 확률 분포
        predictions = predictor.predict_proba(X_sample)
        proba_positive = predictions[:, 1]
        
        plt.figure(figsize=(10, 6))
        plt.hist(proba_positive, bins=50, alpha=0.7)
        plt.xlabel('Predicted Probability (Survived)')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig('custom_model_prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 예측 확률 분포가 'custom_model_prediction_distribution.png'에 저장되었습니다.")
        
        print("\n=== 분석 완료! ===")
        print("생성된 파일:")
        print("- custom_model_feature_importance.png: 특성 중요도")
        print("- custom_model_feature_distributions.png: 특성 분포")
        print("- custom_model_prediction_distribution.png: 예측 확률 분포")
        
        # 7. 추가 정보
        print(f"\n모델 성능 점수: {leaderboard.iloc[0]['score_val']:.4f}")
        print(f"평가 지표: {leaderboard.iloc[0]['eval_metric']}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 