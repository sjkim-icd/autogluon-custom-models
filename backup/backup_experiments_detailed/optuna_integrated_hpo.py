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

def create_integrated_objective(train_data, test_data):
    """통합 Optuna objective 함수"""
    
    def objective(trial):
        # 모든 모델의 하이퍼파라미터를 한번에 탐색
        hyperparameters = {}
        
        # DCNV2 하이퍼파라미터
        if trial.suggest_categorical('use_dcnv2', [True, False]):
            hyperparameters['DCNV2'] = {
                'learning_rate': trial.suggest_float('dcnv2_lr', 1e-4, 1e-2, log=True),
                'weight_decay': trial.suggest_float('dcnv2_wd', 1e-6, 1e-3, log=True),
                'dropout_prob': trial.suggest_categorical('dcnv2_dropout', [0.1, 0.2, 0.3]),
                'num_layers': trial.suggest_categorical('dcnv2_layers', [3, 4, 5]),
                'hidden_size': trial.suggest_categorical('dcnv2_hidden', [128, 256, 512]),
                'num_epochs': trial.suggest_categorical('dcnv2_epochs', [15, 20, 25]),
            }
        
        # Focal Loss 하이퍼파라미터
        if trial.suggest_categorical('use_focal', [True, False]):
            hyperparameters['CUSTOM_FOCAL_DL'] = {
                'learning_rate': trial.suggest_float('focal_lr', 1e-4, 1e-2, log=True),
                'weight_decay': trial.suggest_float('focal_wd', 1e-6, 1e-3, log=True),
                'dropout_prob': trial.suggest_categorical('focal_dropout', [0.1, 0.2, 0.3]),
                'num_layers': trial.suggest_categorical('focal_layers', [3, 4, 5]),
                'hidden_size': trial.suggest_categorical('focal_hidden', [128, 256, 512]),
                'num_epochs': trial.suggest_categorical('focal_epochs', [15, 20, 25]),
                'focal_alpha': trial.suggest_categorical('focal_alpha', [0.25, 0.5, 0.75, 1.0]),
                'focal_gamma': trial.suggest_categorical('focal_gamma', [1.0, 2.0, 3.0]),
            }
        
        # RF 하이퍼파라미터
        if trial.suggest_categorical('use_rf', [True, False]):
            hyperparameters['RF'] = {
                'n_estimators': trial.suggest_categorical('rf_n_estimators', [50, 100, 200]),
                'max_depth': trial.suggest_categorical('rf_max_depth', [5, 10, 15, None]),
                'min_samples_split': trial.suggest_categorical('rf_min_split', [2, 5, 10]),
                'min_samples_leaf': trial.suggest_categorical('rf_min_leaf', [1, 2, 4]),
            }
        
        # 최소 1개 모델은 사용
        if not hyperparameters:
            hyperparameters['RF'] = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
            }
        
        # AutoGluon Predictor 설정
        predictor = TabularPredictor(
            label='survived',
            problem_type='binary',
            eval_metric='f1',
            path=f"models/optuna_integrated_{trial.number}",
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
                print(f"Trial {trial.number}: 앙상블 - F1 = {best_score:.4f}")
                print(f"사용된 모델: {list(hyperparameters.keys())}")
                return best_score
            else:
                return 0.0
                
        except Exception as e:
            print(f"Trial {trial.number} 실패: {e}")
            return 0.0
    
    return objective

def run_integrated_hpo():
    """통합 Optuna HPO 실험"""
    print("=== 통합 Optuna HPO 실험 시작 ===")
    
    # 데이터 로드
    train_data, test_data = load_data()
    
    # 통합 Optuna study 생성
    study = optuna.create_study(
        direction='maximize',
        study_name='integrated_optuna_study'
    )
    
    # Objective 함수 생성
    objective = create_integrated_objective(train_data, test_data)
    
    # Optuna 최적화 실행
    study.optimize(
        objective,
        n_trials=20,  # 20번 시도
        timeout=3600,  # 1시간 제한
        show_progress_bar=True
    )
    
    # 결과 출력
    print(f"\n📊 통합 최적화 결과:")
    print(f"최고 성능: {study.best_value:.4f}")
    print(f"최적 하이퍼파라미터: {study.best_params}")
    
    # 최적 하이퍼파라미터로 최종 모델 학습
    print("\n🚀 최적 하이퍼파라미터로 최종 모델 학습...")
    best_hyperparameters = {}
    
    # 최적 파라미터에서 모델별 하이퍼파라미터 추출
    best_params = study.best_params
    
    if best_params.get('use_dcnv2', False):
        best_hyperparameters['DCNV2'] = {
            'learning_rate': best_params['dcnv2_lr'],
            'weight_decay': best_params['dcnv2_wd'],
            'dropout_prob': best_params['dcnv2_dropout'],
            'num_layers': best_params['dcnv2_layers'],
            'hidden_size': best_params['dcnv2_hidden'],
            'num_epochs': best_params['dcnv2_epochs'],
        }
    
    if best_params.get('use_focal', False):
        best_hyperparameters['CUSTOM_FOCAL_DL'] = {
            'learning_rate': best_params['focal_lr'],
            'weight_decay': best_params['focal_wd'],
            'dropout_prob': best_params['focal_dropout'],
            'num_layers': best_params['focal_layers'],
            'hidden_size': best_params['focal_hidden'],
            'num_epochs': best_params['focal_epochs'],
            'focal_alpha': best_params['focal_alpha'],
            'focal_gamma': best_params['focal_gamma'],
        }
    
    if best_params.get('use_rf', False):
        best_hyperparameters['RF'] = {
            'n_estimators': best_params['rf_n_estimators'],
            'max_depth': best_params['rf_max_depth'],
            'min_samples_split': best_params['rf_min_split'],
            'min_samples_leaf': best_params['rf_min_leaf'],
        }
    
    # 최종 모델 학습
    final_predictor = TabularPredictor(
        label='survived',
        problem_type='binary',
        eval_metric='f1',
        path="models/optuna_final_ensemble",
        verbosity=2
    )
    
    final_predictor.fit(
        train_data=train_data,
        hyperparameters=best_hyperparameters,
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
    
    print("\n✅ 통합 Optuna HPO 실험 완료!")

if __name__ == "__main__":
    run_integrated_hpo() 