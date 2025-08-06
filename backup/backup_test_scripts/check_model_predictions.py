import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

def check_model_predictions():
    """모델들의 예측 결과를 비교"""
    print("=== 모델 예측 결과 비교 ===")
    
    # 데이터 로드
    data = pd.read_csv('datasets/creditcard.csv')
    
    # 검증 데이터 분할 (AutoGluon과 동일하게)
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(
        data, 
        test_size=0.0219,  # AutoGluon의 holdout_frac
        random_state=42, 
        stratify=data['Class']
    )
    
    print(f"훈련 데이터: {len(train_data)}개")
    print(f"검증 데이터: {len(test_data)}개")
    print(f"검증 데이터 Class 분포: {test_data['Class'].value_counts().to_dict()}")
    
    # 기존 모델 로드
    try:
        predictor = TabularPredictor.load('models/five_models_experiment')
        
        # 각 모델의 예측 결과 확인
        models_to_check = ['CUSTOM_FOCAL_DL', 'CUSTOM_NN_TORCH', 'DCNV2', 'RandomForest']
        
        print("\n=== 각 모델의 검증 데이터 예측 결과 ===")
        predictions = {}
        
        for model_name in models_to_check:
            try:
                # 모델별 예측
                pred_proba = predictor.predict_proba(test_data, model=model_name)
                pred_class = predictor.predict(test_data, model=model_name)
                
                predictions[model_name] = {
                    'proba': pred_proba,
                    'class': pred_class,
                    'f1_score': predictor.evaluate(test_data, model=model_name)['f1']
                }
                
                print(f"\n{model_name}:")
                print(f"  F1 Score: {predictions[model_name]['f1_score']:.4f}")
                print(f"  예측 분포: {pred_class.value_counts().to_dict()}")
                print(f"  확률 분포: {pred_proba.iloc[:, 1].describe()}")
                
            except Exception as e:
                print(f"{model_name}: 에러 - {e}")
        
        # 예측 결과 비교
        print("\n=== 예측 결과 유사도 분석 ===")
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
                    
                    print(f"{model1} vs {model2}:")
                    print(f"  클래스 일치도: {class_agreement:.4f}")
                    print(f"  확률 상관관계: {proba_corr:.4f}")
        
    except Exception as e:
        print(f"모델 로드 에러: {e}")
        
        # Focal Loss HPO 결과 확인
        try:
            focal_predictor = TabularPredictor.load('models/focal_hpo_test')
            
            print("\n=== Focal Loss HPO 모델들의 예측 결과 ===")
            focal_models = [model for model in focal_predictor.get_model_names() if 'CUSTOM_FOCAL_DL' in model]
            
            focal_predictions = {}
            for model_name in focal_models:
                try:
                    pred_proba = focal_predictor.predict_proba(test_data, model=model_name)
                    pred_class = focal_predictor.predict(test_data, model=model_name)
                    
                    focal_predictions[model_name] = {
                        'proba': pred_proba,
                        'class': pred_class,
                        'f1_score': focal_predictor.evaluate(test_data, model=model_name)['f1']
                    }
                    
                    print(f"\n{model_name}:")
                    print(f"  F1 Score: {focal_predictions[model_name]['f1_score']:.4f}")
                    print(f"  예측 분포: {pred_class.value_counts().to_dict()}")
                    print(f"  확률 분포: {pred_proba.iloc[:, 1].describe()}")
                    
                except Exception as e:
                    print(f"{model_name}: 에러 - {e}")
            
            # Focal Loss 모델들 간 비교
            if len(focal_predictions) >= 2:
                print("\n=== Focal Loss 모델들 간 예측 유사도 ===")
                focal_model_names = list(focal_predictions.keys())
                for i in range(len(focal_model_names)):
                    for j in range(i+1, len(focal_model_names)):
                        model1, model2 = focal_model_names[i], focal_model_names[j]
                        
                        class_agreement = (focal_predictions[model1]['class'] == focal_predictions[model2]['class']).mean()
                        proba_corr = focal_predictions[model1]['proba'].iloc[:, 1].corr(
                            focal_predictions[model2]['proba'].iloc[:, 1]
                        )
                        
                        print(f"{model1} vs {model2}:")
                        print(f"  클래스 일치도: {class_agreement:.4f}")
                        print(f"  확률 상관관계: {proba_corr:.4f}")
                        
        except Exception as e:
            print(f"Focal Loss 모델 로드 에러: {e}")

if __name__ == "__main__":
    check_model_predictions() 