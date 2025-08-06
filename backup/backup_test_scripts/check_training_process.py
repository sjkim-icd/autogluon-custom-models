import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import os

def check_training_process():
    """학습 과정을 자세히 분석"""
    print("=== 학습 과정 분석 ===")
    
    # 데이터 로드
    data = pd.read_csv('datasets/creditcard.csv')
    
    # AutoGluon과 동일한 검증 데이터 분할
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        data, 
        test_size=0.0219,  # AutoGluon의 holdout_frac
        random_state=42, 
        stratify=data['Class']
    )
    
    print(f"훈련 데이터: {len(train_data)}개")
    print(f"검증 데이터: {len(val_data)}개")
    print(f"검증 데이터 Class 분포: {val_data['Class'].value_counts().to_dict()}")
    
    # Focal Loss HPO 결과 확인
    try:
        focal_predictor = TabularPredictor.load('models/focal_hpo_test')
        
        print("\n=== Focal Loss HPO 모델들의 상세 분석 ===")
        
        # 리더보드에서 모델 이름 가져오기
        leaderboard = focal_predictor.leaderboard()
        focal_models = [row['model'] for idx, row in leaderboard.iterrows() if 'CUSTOM_FOCAL_DL' in row['model']]
        
        print(f"발견된 Focal Loss 모델들: {focal_models}")
        
        for model_name in focal_models:
            try:
                print(f"\n--- {model_name} 분석 ---")
                
                # 모델별 예측
                pred_proba = focal_predictor.predict_proba(val_data, model=model_name)
                pred_class = focal_predictor.predict(val_data, model=model_name)
                
                # 예측 결과 분석
                print(f"예측 분포: {pred_class.value_counts().to_dict()}")
                print(f"확률 분포:")
                print(f"  Class 0 확률: {pred_proba.iloc[:, 0].describe()}")
                print(f"  Class 1 확률: {pred_proba.iloc[:, 1].describe()}")
                
                # 실제 vs 예측 비교
                from sklearn.metrics import classification_report, confusion_matrix
                print(f"\n분류 리포트:")
                print(classification_report(val_data['Class'], pred_class))
                
                print(f"혼동 행렬:")
                print(confusion_matrix(val_data['Class'], pred_class))
                
                # F1 스코어 계산
                from sklearn.metrics import f1_score
                f1 = f1_score(val_data['Class'], pred_class)
                print(f"F1 Score: {f1:.4f}")
                
            except Exception as e:
                print(f"{model_name}: 에러 - {e}")
        
        # 모델들 간 예측 비교
        print("\n=== 모델들 간 예측 비교 ===")
        predictions = {}
        
        for model_name in focal_models:
            try:
                pred_proba = focal_predictor.predict_proba(val_data, model=model_name)
                pred_class = focal_predictor.predict(val_data, model=model_name)
                predictions[model_name] = {
                    'proba': pred_proba,
                    'class': pred_class
                }
            except Exception as e:
                print(f"{model_name}: 에러 - {e}")
        
        # 모델들 간 유사도 분석
        if len(predictions) >= 2:
            model_names = list(predictions.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    
                    # 클래스 예측 일치도
                    class_agreement = (predictions[model1]['class'] == predictions[model2]['class']).mean()
                    
                    # 확률 상관관계
                    proba_corr = predictions[model1]['proba'].iloc[:, 1].corr(
                        predictions[model2]['proba'].iloc[:, 1]
                    )
                    
                    print(f"\n{model1} vs {model2}:")
                    print(f"  클래스 일치도: {class_agreement:.4f}")
                    print(f"  확률 상관관계: {proba_corr:.4f}")
                    
                    if class_agreement > 0.95:
                        print(f"  ⚠️  경고: 모델들이 거의 동일한 예측을 함!")
                    
                    if abs(proba_corr) > 0.95:
                        print(f"  ⚠️  경고: 모델들의 확률 분포가 매우 유사함!")
        
        # 기존 모델과 비교
        print("\n=== 기존 모델과 비교 ===")
        try:
            old_predictor = TabularPredictor.load('models/five_models_experiment')
            
            # 기존 CUSTOM_FOCAL_DL과 비교
            old_focal_pred = old_predictor.predict(val_data, model='CUSTOM_FOCAL_DL')
            old_focal_proba = old_predictor.predict_proba(val_data, model='CUSTOM_FOCAL_DL')
            
            print("기존 CUSTOM_FOCAL_DL 예측 분포:", old_focal_pred.value_counts().to_dict())
            
            # 새로운 Focal Loss 모델들과 비교
            for model_name in focal_models:
                try:
                    new_pred = predictions[model_name]['class']
                    agreement = (old_focal_pred == new_pred).mean()
                    print(f"기존 vs {model_name} 일치도: {agreement:.4f}")
                    
                    if agreement > 0.95:
                        print(f"  ⚠️  경고: 기존 모델과 거의 동일한 예측!")
                        
                except Exception as e:
                    print(f"{model_name} 비교 에러: {e}")
                    
        except Exception as e:
            print(f"기존 모델 로드 에러: {e}")
            
    except Exception as e:
        print(f"Focal Loss 모델 로드 에러: {e}")

if __name__ == "__main__":
    check_training_process() 