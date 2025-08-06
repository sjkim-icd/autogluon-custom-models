import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import optuna
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_deepfm_torch_model import TabularDeepFMTorchModel
from custom_models.tabular_dcnv2_torch_model import TabularDCNv2TorchModel
from custom_models.tabular_dcnv2_fuxictr_torch_model_fixed import TabularDCNv2FuxiCTRTorchModel
from custom_models.focal_loss_implementation import CustomFocalDLModel
from custom_models.custom_nn_torch_model import CustomNNTorchModel
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import time
import json

# 모델 등록
ag_model_registry.add(TabularDeepFMTorchModel)
ag_model_registry.add(TabularDCNv2TorchModel)
ag_model_registry.add(TabularDCNv2FuxiCTRTorchModel)
ag_model_registry.add(CustomFocalDLModel)
ag_model_registry.add(CustomNNTorchModel)

def load_data():
    """Titanic 데이터 로드"""
    print("Titanic 데이터 로드 중...")
    titanic = fetch_openml("titanic", version=1, as_frame=True)
    df = titanic.frame
    
    # 데이터 전처리
    df = df.dropna(subset=['survived'])
    df['survived'] = df['survived'].astype(int)
    
    # 데이터 분할
    train_data, test_data = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['survived']
    )
    
    print(f"훈련 데이터: {len(train_data)}개")
    print(f"테스트 데이터: {len(test_data)}개")
    
    return train_data, test_data

def stage1_individual_hpo(train_data, test_data):
    """1단계: 개별 모델 최적화"""
    print("=== 1단계: 개별 모델 최적화 ===")
    
    best_params = {}
    model_types = ['DCNV2', 'CUSTOM_FOCAL_DL', 'RF']
    
    for model_type in model_types:
        print(f"\n🔍 {model_type} 개별 최적화 시작")
        print("=" * 50)
        
        # Optuna study 생성
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{model_type}_stage1_study'
        )
        
        # Objective 함수 생성
        def objective(trial):
            # 모델별 하이퍼파라미터 정의
            if model_type == 'DCNV2':
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                    'dropout_prob': trial.suggest_categorical('dropout_prob', [0.1, 0.2, 0.3]),
                    'num_layers': trial.suggest_categorical('num_layers', [3, 4, 5]),
                    'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
                    'num_epochs': trial.suggest_categorical('num_epochs', [15, 20, 25]),
                }
            elif model_type == 'CUSTOM_FOCAL_DL':
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                    'dropout_prob': trial.suggest_categorical('dropout_prob', [0.1, 0.2, 0.3]),
                    'num_layers': trial.suggest_categorical('num_layers', [3, 4, 5]),
                    'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
                    'num_epochs': trial.suggest_categorical('num_epochs', [15, 20, 25]),
                    'focal_alpha': trial.suggest_categorical('focal_alpha', [0.25, 0.5, 0.75, 1.0]),
                    'focal_gamma': trial.suggest_categorical('focal_gamma', [1.0, 2.0, 3.0]),
                }
            elif model_type == 'RF':
                params = {
                    'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
                    'max_depth': trial.suggest_categorical('max_depth', [5, 10, 15, None]),
                    'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),
                    'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4]),
                }
            
            # AutoGluon Predictor 설정
            predictor = TabularPredictor(
                label='survived',
                problem_type='binary',
                eval_metric='f1',
                path=f"models/stage1_{model_type}_{trial.number}",
                verbosity=2
            )
            
            # 모델 학습
            try:
                predictor.fit(
                    train_data=train_data,
                    hyperparameters={model_type: params},
                    time_limit=300,  # 5분 제한
                    presets='medium_quality'
                )
                
                # 검증 성능 평가
                val_scores = predictor.leaderboard()
                if len(val_scores) > 0:
                    best_score = val_scores.iloc[0]['score_val']
                    print(f"Trial {trial.number}: {model_type} - F1 = {best_score:.4f}")
                    return best_score
                else:
                    return 0.0
                    
            except Exception as e:
                print(f"Trial {trial.number} 실패: {e}")
                return 0.0
        
        # Optuna 최적화 실행
        study.optimize(
            objective,
            n_trials=15,  # 15번 시도
            timeout=900,  # 15분 제한
            show_progress_bar=True
        )
        
        # 결과 저장
        best_params[model_type] = {
            'best_value': study.best_value,
            'best_params': study.best_params
        }
        
        print(f"\n📊 {model_type} 최적화 결과:")
        print(f"최고 성능: {study.best_value:.4f}")
        print(f"최적 하이퍼파라미터: {study.best_params}")
    
    return best_params

def stage2_ensemble_hpo(train_data, test_data, best_individual_params):
    """2단계: 앙상블 최적화"""
    print("\n=== 2단계: 앙상블 최적화 ===")
    
    def objective(trial):
        # 어떤 모델들을 사용할지 결정
        use_models = {}
        for model_type in ['DCNV2', 'CUSTOM_FOCAL_DL', 'RF']:
            use_models[model_type] = trial.suggest_categorical(f'use_{model_type.lower()}', [True, False])
        
        # 최소 1개 모델은 사용
        if not any(use_models.values()):
            use_models['RF'] = True
        
        # 최적화된 하이퍼파라미터 사용
        hyperparameters = {}
        for model_type, use_model in use_models.items():
            if use_model and model_type in best_individual_params:
                hyperparameters[model_type] = best_individual_params[model_type]['best_params']
        
        # 앙상블 가중치 최적화 (선택사항)
        ensemble_weights = {}
        if len(hyperparameters) > 1:
            for model_type in hyperparameters.keys():
                weight = trial.suggest_float(f'weight_{model_type.lower()}', 0.1, 1.0)
                ensemble_weights[model_type] = weight
        
        # AutoGluon Predictor 설정
        predictor = TabularPredictor(
            label='survived',
            problem_type='binary',
            eval_metric='f1',
            path=f"models/stage2_ensemble_{trial.number}",
            verbosity=2
        )
        
        # 모델 학습
        try:
            predictor.fit(
                train_data=train_data,
                hyperparameters=hyperparameters,
                time_limit=600,  # 10분 제한
                presets='medium_quality'
            )
            
            # 검증 성능 평가
            val_scores = predictor.leaderboard()
            if len(val_scores) > 0:
                best_score = val_scores.iloc[0]['score_val']
                used_models = list(hyperparameters.keys())
                print(f"Trial {trial.number}: 앙상블 - F1 = {best_score:.4f}")
                print(f"사용된 모델: {used_models}")
                return best_score
            else:
                return 0.0
                
        except Exception as e:
            print(f"Trial {trial.number} 실패: {e}")
            return 0.0
    
    # Optuna study 생성
    study = optuna.create_study(
        direction='maximize',
        study_name='stage2_ensemble_study'
    )
    
    # Optuna 최적화 실행
    study.optimize(
        objective,
        n_trials=20,  # 20번 시도
        timeout=1800,  # 30분 제한
        show_progress_bar=True
    )
    
    return study

def run_two_stage_hpo():
    """2단계 HPO 실험"""
    print("=== 2단계 Optuna HPO 실험 시작 ===")
    
    # 데이터 로드
    train_data, test_data = load_data()
    
    # 1단계: 개별 모델 최적화
    best_individual_params = stage1_individual_hpo(train_data, test_data)
    
    # 결과 저장
    with open('best_individual_params.json', 'w') as f:
        json.dump(best_individual_params, f, indent=2)
    
    print("\n📊 1단계 결과 요약:")
    for model_type, result in best_individual_params.items():
        print(f"{model_type}: F1 = {result['best_value']:.4f}")
    
    # 2단계: 앙상블 최적화
    ensemble_study = stage2_ensemble_hpo(train_data, test_data, best_individual_params)
    
    # 최종 결과 출력
    print(f"\n📊 2단계 앙상블 최적화 결과:")
    print(f"최고 성능: {ensemble_study.best_value:.4f}")
    print(f"최적 하이퍼파라미터: {ensemble_study.best_params}")
    
    # 최적 앙상블 구성
    best_ensemble_params = {}
    best_params = ensemble_study.best_params
    
    for model_type in ['DCNV2', 'CUSTOM_FOCAL_DL', 'RF']:
        if best_params.get(f'use_{model_type.lower()}', False):
            best_ensemble_params[model_type] = best_individual_params[model_type]['best_params']
    
    print(f"\n🏆 최종 앙상블 구성:")
    for model_type, params in best_ensemble_params.items():
        print(f"{model_type}: {params}")
    
    # 최종 모델 학습
    print("\n🚀 최종 앙상블 모델 학습...")
    final_predictor = TabularPredictor(
        label='survived',
        problem_type='binary',
        eval_metric='f1',
        path="models/final_two_stage_ensemble",
        verbosity=2
    )
    
    final_predictor.fit(
        train_data=train_data,
        hyperparameters=best_ensemble_params,
        time_limit=1200,  # 20분 제한
        presets='medium_quality'
    )
    
    # 최종 성능 평가
    final_scores = final_predictor.leaderboard()
    print(f"\n🏆 최종 앙상블 성능:")
    print(final_scores)
    
    # 테스트 데이터로 평가
    test_predictions = final_predictor.predict(test_data)
    test_probabilities = final_predictor.predict_proba(test_data)
    
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    
    test_f1 = f1_score(test_data['survived'], test_predictions)
    test_acc = accuracy_score(test_data['survived'], test_predictions)
    test_prec = precision_score(test_data['survived'], test_predictions)
    test_rec = recall_score(test_data['survived'], test_predictions)
    
    print(f"\n📊 테스트 데이터 성능:")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    
    print("\n✅ 2단계 Optuna HPO 실험 완료!")

if __name__ == "__main__":
    run_two_stage_hpo() 