import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import optuna
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_dcnv2_torch_model import TabularDCNv2TorchModel
from custom_models.tabular_dcnv2_fuxictr_torch_model_fixed import TabularDCNv2FuxiCTRTorchModel
from custom_models.focal_loss_implementation import CustomFocalDLModel
from custom_models.custom_nn_torch_model import CustomNNTorchModel
from custom_models.advanced_focal_loss import AdvancedFocalDLModel
from custom_models.adafocal_loss import AdaFocalDL  # AdaFocal 모델 추가

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import time
import json
from datetime import datetime
import argparse

# 모델 등록
ag_model_registry.add(TabularDCNv2TorchModel)
ag_model_registry.add(TabularDCNv2FuxiCTRTorchModel)
ag_model_registry.add(CustomFocalDLModel)
ag_model_registry.add(CustomNNTorchModel)
ag_model_registry.add(AdvancedFocalDLModel)
ag_model_registry.add(AdaFocalDL)  # AdaFocal 모델 등록

def load_titanic_data():
    """Titanic 데이터 로드"""
    print("📊 Titanic 데이터 로드 중...")
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
    
    print(f"✅ 훈련 데이터: {len(train_data)}개")
    print(f"✅ 테스트 데이터: {len(test_data)}개")
    
    return train_data, test_data

def load_credit_card_data():
    """Credit Card 데이터 로드 및 전처리"""
    print(" Credit Card 데이터 로드 중...")
    
    # 데이터 로드
    df = pd.read_csv("datasets/creditcard.csv")
    
    print(f"📈 데이터 형태: {df.shape}")
    print(f" 타겟 분포:")
    print(df['Class'].value_counts())
    
    # 데이터 분할
    train_data, test_data = train_test_split(
        df, test_size=0.2, random_state=42, 
        stratify=df['Class']
    )
    
    print(f"✅ 학습 데이터: {train_data.shape[0]}개")
    print(f"✅ 테스트 데이터: {test_data.shape[0]}개")
    
    return train_data, test_data

def load_data(dataset_name):
    """데이터셋 선택에 따른 데이터 로드"""
    if dataset_name.lower() == 'titanic':
        return load_titanic_data()
    elif dataset_name.lower() == 'creditcard':
        return load_credit_card_data()
    else:
        raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")

def individual_model_hpo(train_data, test_data, experiment_name):
    """개별 모델 최적화 (통합 DB 사용)"""
    print("=== 개별 모델 최적화 (통합 DB) ===")
    
    best_params = {}
    model_types = ['DCNV2', 'DCNV2_FUXICTR', 'CUSTOM_FOCAL_DL', 'CUSTOM_NN_TORCH', 'ADVANCED_FOCAL_DL', 'ADAFOCAL_DL', 'RF']  # 7개 모델로 확장
    # model_types = [ 'ADAFOCAL_DL']  # 7개 모델로 확장
    
    # 실험별 DB 경로 자동 구성
    db_dir = f'optuna_studies/{experiment_name}'
    os.makedirs(db_dir, exist_ok=True)
    unified_db_path = f'sqlite:///{db_dir}/all_studies.db'
    
    print(f"🔍 DB 경로: {db_dir}/all_studies.db")
    
    for model_type in model_types:
        print(f"\n🔍 {model_type} 최적화 시작")
        print("=" * 50)
        
        # Optuna study 생성 (통합 DB 사용)
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{model_type}_hpo_study',
            storage=unified_db_path,  # 하나의 DB 파일 사용
            load_if_exists=True
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
            elif model_type == 'DCNV2_FUXICTR':
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
            elif model_type == 'CUSTOM_NN_TORCH':
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                    'dropout_prob': trial.suggest_categorical('dropout_prob', [0.1, 0.2, 0.3]),
                    'num_layers': trial.suggest_categorical('num_layers', [3, 4, 5]),
                    'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
                    'num_epochs': trial.suggest_categorical('num_epochs', [15, 20, 25]),
                }
            elif model_type == 'ADVANCED_FOCAL_DL':
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                    'dropout_prob': trial.suggest_categorical('dropout_prob', [0.1, 0.2, 0.3]),
                    'num_layers': trial.suggest_categorical('num_layers', [3, 4, 5]),
                    'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
                    'num_epochs': trial.suggest_categorical('num_epochs', [15, 20, 25]),
                    'focal_alpha': trial.suggest_categorical('focal_alpha', [0.25, 0.5, 0.75, 1.0]),
                    'base_gamma': trial.suggest_categorical('base_gamma', [1.0, 1.5, 2.0, 2.5, 3.0]),
                }
            elif model_type == 'ADAFOCAL_DL':  # AdaFocal 모델 하이퍼파라미터 추가
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                    'dropout_prob': trial.suggest_categorical('dropout_prob', [0.1, 0.2, 0.3]),
                    'num_layers': trial.suggest_categorical('num_layers', [3, 4, 5]),
                    'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
                    'num_epochs': trial.suggest_categorical('num_epochs', [15, 20, 25]),
                    'alpha': trial.suggest_categorical('alpha', [0.25, 0.5, 0.75, 1.0]),  # AdaFocal alpha
                    'base_gamma': trial.suggest_categorical('base_gamma', [1.0, 1.5, 2.0, 2.5, 3.0]),  # AdaFocal base_gamma
                    'momentum': trial.suggest_categorical('momentum', [0.8, 0.9, 0.95]),  # AdaFocal momentum
                    'num_bins': trial.suggest_categorical('num_bins', [10, 15, 20]),  # AdaFocal num_bins
                    'calibration_threshold': trial.suggest_categorical('calibration_threshold', [0.1, 0.2, 0.3]),  # AdaFocal Sth
                    'lambda_param': trial.suggest_categorical('lambda_param', [0.5, 1.0, 1.5]),  # AdaFocal λ
                    'gamma_max': trial.suggest_categorical('gamma_max', [15.0, 20.0, 25.0]),  # AdaFocal γmax
                    'gamma_min': trial.suggest_categorical('gamma_min', [-2.0, -1.5, -1.0]),  # AdaFocal γmin
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
                path=f"models/{model_type}_hpo",
                verbosity=0
            )
            
            # 하이퍼파라미터 설정
            hyperparameters = {
                model_type: params
            }
            
            try:
                # 모델 학습
                predictor.fit(
                    train_data=train_data,
                    hyperparameters=hyperparameters,
                    time_limit=300,  # 5분 제한
                    presets='medium_quality'
                )
                
                # 검증 성능 평가
                val_scores = predictor.leaderboard()
                if len(val_scores) > 0:
                    best_score = val_scores.iloc[0]['score_val']
                    print(f"Trial {trial.number}: F1 = {best_score:.4f}")
                    return best_score
                else:
                    print(f"Trial {trial.number}: 학습 실패")
                    return 0.0
                    
            except Exception as e:
                print(f"Trial {trial.number} 오류: {e}")
                return 0.0
        
        # Optuna 최적화 실행
        study.optimize(
            objective,
            n_trials=3, # 각 모델당 15번
            timeout=600,   # 10분 제한
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

def final_ensemble_with_autogluon(train_data, test_data, best_params):
    """AutoGluon 자동 앙상블로 최종 모델 학습"""
    print("\n=== AutoGluon 자동 앙상블 학습 ===")
    
    # 최적화된 하이퍼파라미터로 앙상블 구성
    hyperparameters = {}
    for model_type, result in best_params.items():
        hyperparameters[model_type] = result['best_params']
    
    print(f"앙상블 구성:")
    for model_type, params in hyperparameters.items():
        print(f"  {model_type}: {params}")
    
    # AutoGluon Predictor 설정
    final_predictor = TabularPredictor(
        label='survived',
        problem_type='binary',
        eval_metric='f1',
        path="models/final_autogluon_ensemble",
        verbosity=4
    )
    
    # 모든 모델을 한번에 학습 (AutoGluon이 자동으로 앙상블)
    final_predictor.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
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
    
    return final_predictor, {
        'test_f1': test_f1,
        'test_acc': test_acc,
        'test_prec': test_prec,
        'test_rec': test_rec
    }

def run_single_stage_hpo_unified_db(experiment_name, dataset_name):  # dataset_name 인자 추가
    """1단계 HPO + AutoGluon 자동 앙상블 실험 실행"""
    print(f"🚀 1단계 HPO + AutoGluon 자동 앙상블 실험 시작: {experiment_name}")
    
    # 데이터 로드 (데이터셋 선택)
    train_data, test_data = load_data(dataset_name)  # dataset_name 전달
    
    # 1단계: 개별 모델 최적화 (통합 DB 사용)
    best_params = individual_model_hpo(train_data, test_data, experiment_name)
    
    # 실험별 결과 폴더 생성
    experiment_results_dir = f"results/{experiment_name}"
    os.makedirs(experiment_results_dir, exist_ok=True)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{experiment_results_dir}/best_individual_params_{experiment_name}_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ 최적 하이퍼파라미터가 '{result_file}'에 저장되었습니다!")
    
    # 2단계: AutoGluon 자동 앙상블
    final_predictor, test_metrics = final_ensemble_with_autogluon(train_data, test_data, best_params)
    
    print(f"\n✅ 1단계 HPO + AutoGluon 자동 앙상블 실험 완료 (통합 DB)!")
    print(f"최종 테스트 F1 Score: {test_metrics['test_f1']:.4f}")
    print(f"통합 DB 파일: optuna_studies/{experiment_name}/all_studies.db")
    
    print(f"\n💡 다음 명령어로 분석 대시보드를 생성할 수 있습니다:")
    print(f"python analysis/create_final_unified_dashboard_excel_fixed.py \"{experiment_name}\"")

def main():
    parser = argparse.ArgumentParser(description='Optuna HPO')
    parser.add_argument('experiment_name', type=str, help='실험 이름')
    parser.add_argument('dataset_name', type=str, choices=['titanic', 'creditcard'], help='데이터셋 이름')  # 추가
    parser.add_argument('--num-trials', type=int, default=20, help='trial 수')
    parser.add_argument('--timeout', type=int, default=3600, help='타임아웃(초)')
    
    args = parser.parse_args()
    
    experiment_name = args.experiment_name
    dataset_name = args.dataset_name  # 추가
    num_trials = args.num_trials
    timeout = args.timeout
    
      # 모델 타입 정의 - 이 부분이 누락되어 있었습니다!
    model_types = [
        'DCNV2', 'DCNV2_FUXICTR', 'CUSTOM_FOCAL_DL', 
        'CUSTOM_NN_TORCH', 'ADVANCED_FOCAL_DL', 'ADAFOCAL_DL', 'RF'
    ]

    print(f" 실험 이름: {experiment_name}")
    print(f"🔄 Trial 수: {num_trials}")
    print(f"⏱️  타임아웃: {timeout}초")
    print(f"📊 실험 구성: {len(model_types)}개 모델 ({', '.join(model_types)})")
    print(f"🎯 데이터셋: Titanic (이진 분류)")
    print(f"⏱️  각 모델당 {num_trials} trials, 총 {num_trials * len(model_types)} trials")
    
    # HPO 실행
    run_single_stage_hpo_unified_db(args.experiment_name, args.dataset_name) 

if __name__ == "__main__":
    main() 