import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import shap
import warnings
warnings.filterwarnings('ignore')

def main():
    """커스텀 모델의 SHAP 분석"""
    print("=== Custom Model SHAP 분석 ===")
    
    try:
        # 1. 모델 로드
        print("커스텀 모델 로드 중...")
        model_path = r"C:\Users\woori\Desktop\autogluon_env_cursor\models\CUSTOM_NN_TORCH_hpo"
        print(f"모델 경로 확인: {model_path}")
        
        # 경로 존재 확인
        import os
        if os.path.exists(model_path):
            print("✅ 모델 경로 존재 확인")
        else:
            print("❌ 모델 경로가 존재하지 않습니다!")
            return
        
        predictor = TabularPredictor.load(model_path)
        print("✅ 모델 로드 완료!")
        
        # 2. 모델 정보 확인
        print("\n=== 모델 정보 ===")
        print(f"모델 경로: {model_path}")
        
        # 리더보드 확인
        try:
            leaderboard = predictor.leaderboard()
            print("\n리더보드:")
            print(leaderboard)
        except Exception as e:
            print(f"리더보드 로드 실패: {e}")
        
        # 3. 샘플 데이터 생성 (Titanic 데이터셋 특성들)
        print("\n=== 샘플 데이터 생성 ===")
        
        # Titanic 데이터셋의 특성들
        feature_names = [
            'pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest'
        ]
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_samples = 1000
        
        # 각 특성별로 적절한 분포 생성
        sample_data = {}
        
        # pclass (1, 2, 3)
        sample_data['pclass'] = np.random.choice([1, 2, 3], n_samples, p=[0.25, 0.25, 0.5])
        
        # name (문자열)
        sample_data['name'] = [f"Passenger_{i}" for i in range(n_samples)]
        
        # sex (male, female)
        sample_data['sex'] = np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4])
        
        # age (20-80)
        sample_data['age'] = np.random.normal(30, 15, n_samples)
        sample_data['age'] = np.clip(sample_data['age'], 0, 80)
        
        # sibsp (0-8)
        sample_data['sibsp'] = np.random.poisson(0.5, n_samples)
        sample_data['sibsp'] = np.clip(sample_data['sibsp'], 0, 8)
        
        # parch (0-6)
        sample_data['parch'] = np.random.poisson(0.4, n_samples)
        sample_data['parch'] = np.clip(sample_data['parch'], 0, 6)
        
        # ticket (문자열)
        sample_data['ticket'] = [f"Ticket_{i}" for i in range(n_samples)]
        
        # fare (10-500)
        sample_data['fare'] = np.random.exponential(50, n_samples)
        sample_data['fare'] = np.clip(sample_data['fare'], 10, 500)
        
        # cabin (문자열, 일부는 NaN)
        sample_data['cabin'] = [f"Cabin_{i}" if np.random.random() > 0.7 else np.nan for i in range(n_samples)]
        
        # embarked (S, C, Q)
        sample_data['embarked'] = np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1])
        
        # boat (문자열, 일부는 NaN)
        sample_data['boat'] = [f"Boat_{i}" if np.random.random() > 0.6 else np.nan for i in range(n_samples)]
        
        # body (숫자, 일부는 NaN)
        sample_data['body'] = [i if np.random.random() > 0.8 else np.nan for i in range(n_samples)]
        
        # home.dest (문자열, 일부는 NaN)
        sample_data['home.dest'] = [f"Destination_{i}" if np.random.random() > 0.5 else np.nan for i in range(n_samples)]
        
        # 데이터프레임 생성
        X_sample = pd.DataFrame(sample_data)
        
        print(f"샘플 데이터 생성 완료: {X_sample.shape}")
        print(f"특성 목록: {list(X_sample.columns)}")
        
        # 4. 모델 예측 테스트
        print("\n=== 모델 예측 테스트 ===")
        try:
            predictions = predictor.predict_proba(X_sample.head(5))
            print("샘플 예측 확률:")
            print(predictions)
            print("✅ 예측 성공!")
        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            return
        
        # 5. SHAP 분석
        print("\n=== SHAP 분석 시작 ===")
        
        # 백그라운드 데이터로 SHAP 계산
        background_data = X_sample.sample(min(50, len(X_sample)), random_state=42)
        print(f"백그라운드 데이터 크기: {background_data.shape}")
        
        # SHAP Explainer 생성
        print("SHAP Explainer 생성 중...")
        
        try:
            # AutoGluon 모델을 위한 예측 함수 래퍼
            def model_predict(X):
                if isinstance(X, np.ndarray):
                    # numpy array를 DataFrame으로 변환
                    X_df = pd.DataFrame(X, columns=feature_names)
                else:
                    X_df = X
                proba = predictor.predict_proba(X_df)
                return proba.iloc[:, 1].values  # 올바른 인덱싱
            
            # SHAP Explainer 생성
            explainer = shap.KernelExplainer(
                model=model_predict, 
                data=background_data,
                link="identity"
            )
            print("✅ SHAP Explainer 생성 완료!")
        except Exception as e:
            print(f"❌ SHAP Explainer 생성 실패: {e}")
            print("대안 방법으로 시도...")
            
            # 대안: 더 간단한 방법
            try:
                def model_predict_alt(X):
                    X_df = pd.DataFrame(X, columns=feature_names)
                    proba = predictor.predict_proba(X_df)
                    return proba.iloc[:, 1].values
                
                explainer = shap.KernelExplainer(
                    model=model_predict_alt, 
                    data=background_data.values,
                    link="identity"
                )
                print("✅ SHAP Explainer 생성 완료! (대안 방법)")
            except Exception as e2:
                print(f"❌ 대안 방법도 실패: {e2}")
                return
        
        # SHAP 값 계산 (일부 샘플만)
        print("SHAP 값 계산 중...")
        sample_for_shap = X_sample.sample(min(20, len(X_sample)), random_state=42)
        
        try:
            shap_values = explainer.shap_values(sample_for_shap)
            print(f"✅ SHAP 값 계산 완료! 형태: {np.array(shap_values).shape}")
        except Exception as e:
            print(f"❌ SHAP 값 계산 실패: {e}")
            return
        
        # 특성 중요도 계산
        feature_importance = np.abs(shap_values).mean(0)
        
        # 상위 10개 특성 출력
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\n=== 상위 10개 특성 중요도 (SHAP) ===")
        print(importance_df.head(10))
        
        # 6. 시각화
        print("\n=== SHAP 시각화 생성 ===")
        import matplotlib.pyplot as plt
        
        try:
            # SHAP 요약 플롯
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, sample_for_shap, show=False)
            plt.tight_layout()
            plt.savefig('custom_model_shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ SHAP 요약 플롯이 'custom_model_shap_summary.png'에 저장되었습니다.")
        except Exception as e:
            print(f"❌ SHAP 요약 플롯 생성 실패: {e}")
        
        try:
            # 개별 샘플 SHAP 플롯
            plt.figure(figsize=(10, 6))
            shap.force_plot(
                explainer.expected_value, 
                shap_values[0], 
                sample_for_shap.iloc[0],
                show=False
            )
            plt.tight_layout()
            plt.savefig('custom_model_shap_individual.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 개별 샘플 SHAP 플롯이 'custom_model_shap_individual.png'에 저장되었습니다.")
        except Exception as e:
            print(f"❌ 개별 샘플 SHAP 플롯 생성 실패: {e}")
        
        try:
            # 특성 중요도 바 차트
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('SHAP Importance')
            plt.title('Top 15 Feature Importance (SHAP)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('custom_model_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 특성 중요도 차트가 'custom_model_feature_importance.png'에 저장되었습니다.")
        except Exception as e:
            print(f"❌ 특성 중요도 차트 생성 실패: {e}")
        
        print("\n=== SHAP 분석 완료! ===")
        print("생성된 파일:")
        print("- custom_model_shap_summary.png: 전체 특성 중요도")
        print("- custom_model_shap_individual.png: 개별 샘플 분석")
        print("- custom_model_feature_importance.png: 특성 중요도 차트")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 