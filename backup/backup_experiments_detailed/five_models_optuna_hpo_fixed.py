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

def create_optuna_objective(model_type, train_data, test_data):
    """Optuna objective 함수 생성"""
    
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
                'max_batch_size': trial.suggest_categorical('max_batch_size', [256, 512, 1024]),
            }
        elif model_type == 'CUSTOM_FOCAL_DL':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'dropout_prob': trial.suggest_categorical('dropout_prob', [0.1, 0.2, 0.3]),
                'num_layers': trial.suggest_categorical('num_layers', [3, 4, 5]),
                'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
                'num_epochs': trial.suggest_categorical('num_epochs', [15, 20, 25]),
                'max_batch_size': trial.suggest_categorical('max_batch_size', [256, 512, 1024]),
                'focal_alpha': trial.suggest_categorical('focal_alpha', [0.25, 0.5, 0.75, 1.0]),
                'focal_gamma': trial.suggest_categorical('focal_gamma', [1.0, 2.0, 3.0]),
            }
        elif model_type == 'RF':  # RandomForest -> RF로 수정
            params = {
                'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
                'max_depth': trial.suggest_categorical('max_depth', [5, 10, 15, None]),
                'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),
                'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4]),
            }
        else:
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'dropout_prob': trial.suggest_categorical('dropout_prob', [0.1, 0.2, 0.3]),
                'num_layers': trial.suggest_categorical('num_layers', [3, 4, 5]),
                'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
                'num_epochs': trial.suggest_categorical('num_epochs', [15, 20, 25]),
                'max_batch_size': trial.suggest_categorical('max_batch_size', [256, 512, 1024]),
            }
        
        # AutoGluon Predictor 설정
        predictor = TabularPredictor(
            label='survived',
            problem_type='binary',
            eval_metric='f1',
            path=f"models/optuna_{model_type}_{trial.number}",
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
    
    return objective

def safe_visualization(study, model_type):
    """안전한 시각화 함수"""
    try:
        import matplotlib.pyplot as plt
        
        # 최적화 히스토리
        try:
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.write_image(f'optuna_{model_type}_history.png')
            print(f"📈 최적화 히스토리 저장: optuna_{model_type}_history.png")
        except Exception as e:
            print(f"⚠️ 최적화 히스토리 시각화 실패: {e}")
        
        # 파라미터 중요도 (안전한 버전)
        try:
            # 최소 2개 이상의 성공한 trial이 있는지 확인
            successful_trials = [t for t in study.trials if t.value is not None and t.value > 0]
            if len(successful_trials) >= 2:
                fig2 = optuna.visualization.plot_param_importances(study)
                fig2.write_image(f'optuna_{model_type}_importance.png')
                print(f"📊 파라미터 중요도 저장: optuna_{model_type}_importance.png")
            else:
                print(f"⚠️ 파라미터 중요도 시각화 건너뜀 (성공한 trial 부족)")
        except Exception as e:
            print(f"⚠️ 파라미터 중요도 시각화 실패: {e}")
            
    except ImportError:
        print("📈 matplotlib 없음 - 시각화 건너뜀")

def run_optuna_hpo():
    """Optuna를 사용한 HPO 실험"""
    print("=== Optuna HPO 실험 시작 ===")
    
    # 데이터 로드
    train_data, test_data = load_data()
    
    # 모델 타입별 Optuna 실험
    model_types = ['DCNV2', 'CUSTOM_FOCAL_DL', 'RF']  # RandomForest -> RF로 수정
    
    for model_type in model_types:
        print(f"\n🔍 {model_type} Optuna HPO 시작")
        print("=" * 50)
        
        # Optuna study 생성
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{model_type}_optuna_study'
        )
        
        # Objective 함수 생성
        objective = create_optuna_objective(model_type, train_data, test_data)
        
        # Optuna 최적화 실행
        study.optimize(
            objective,
            n_trials=10,  # 10번 시도
            timeout=1800,  # 30분 제한
            show_progress_bar=True
        )
        
        # 결과 출력
        print(f"\n📊 {model_type} 최적화 결과:")
        print(f"최고 성능: {study.best_value:.4f}")
        print(f"최적 하이퍼파라미터: {study.best_params}")
        
        # 안전한 시각화
        safe_visualization(study, model_type)
    
    print("\n✅ Optuna HPO 실험 완료!")

if __name__ == "__main__":
    run_optuna_hpo() 