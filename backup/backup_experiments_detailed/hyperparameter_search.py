import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import os
import sys

# 커스텀 모델 import
sys.path.append('.')
from custom_models.tabular_dcnv2_torch_model import TabularDCNv2TorchModel
from custom_models.focal_loss_implementation import CustomFocalDLModel
from custom_models.custom_nn_torch_model import CustomNNTorchModel

def load_data():
    """데이터 로드"""
    print("=== 데이터 로드 ===")
    
    # Credit Card Fraud Detection 데이터셋 로드
    data = pd.read_csv('creditcard.csv')
    
    # 학습/테스트 분할
    train_data = data.iloc[:-56962]  # 마지막 56962개를 테스트로 사용
    test_data = data.iloc[-56962:]
    
    print(f"학습 데이터 크기: {train_data.shape}")
    print(f"테스트 데이터 크기: {test_data.shape}")
    print(f"학습 데이터 클래스 분포:\n{train_data['Class'].value_counts()}")
    print(f"학습 데이터 클래스 비율:\n{train_data['Class'].value_counts(normalize=True)}")
    
    return train_data, test_data

def register_custom_models():
    """커스텀 모델 등록"""
    print("=== 커스텀 모델 등록 ===")
    
    # DCNv2 등록
    TabularDCNv2TorchModel.register()
    
    # CustomFocalDLModel 등록
    CustomFocalDLModel.register()
    
    # CustomNNTorchModel 등록
    CustomNNTorchModel.register()
    
    print("✅ 모든 커스텀 모델 등록 완료")

def create_hyperparameter_configs():
    """하이퍼파라미터 검색을 위한 다양한 설정 생성"""
    
    configs = {
        # === DCNv2 하이퍼파라미터 검색 ===
        "DCNV2_SEARCH": [
            # 기본 설정
            {
                "num_cross_layers": 2,
                "cross_dropout": 0.1,
                "low_rank": 16,
                "deep_output_size": 64,
                "deep_hidden_size": 64,
                "deep_dropout": 0.1,
                "deep_layers": 2,
                'epochs_wo_improve': 5,
                'num_epochs': 20,
                "lr_scheduler": True,
                "scheduler_type": "cosine",
                "lr_scheduler_min_lr": 1e-6,
                "learning_rate": 0.0001,
            },
            # 더 깊은 네트워크
            {
                "num_cross_layers": 3,
                "cross_dropout": 0.2,
                "low_rank": 32,
                "deep_output_size": 128,
                "deep_hidden_size": 128,
                "deep_dropout": 0.2,
                "deep_layers": 3,
                'epochs_wo_improve': 5,
                'num_epochs': 20,
                "lr_scheduler": True,
                "scheduler_type": "cosine",
                "lr_scheduler_min_lr": 1e-6,
                "learning_rate": 0.0001,
            },
            # 더 높은 학습률
            {
                "num_cross_layers": 2,
                "cross_dropout": 0.1,
                "low_rank": 16,
                "deep_output_size": 64,
                "deep_hidden_size": 64,
                "deep_dropout": 0.1,
                "deep_layers": 2,
                'epochs_wo_improve': 5,
                'num_epochs': 20,
                "lr_scheduler": True,
                "scheduler_type": "cosine",
                "lr_scheduler_min_lr": 1e-6,
                "learning_rate": 0.001,
            },
        ],
        
        # === CustomFocalDL 하이퍼파라미터 검색 ===
        "CUSTOM_FOCAL_DL_SEARCH": [
            # 기본 설정
            {
                'max_batch_size': 512,
                'num_epochs': 20,
                'epochs_wo_improve': 5,
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'dropout_prob': 0.1,
                'num_layers': 4,
                'hidden_size': 128,
                'activation': 'relu',
                'lr_scheduler': True,
                'scheduler_type': 'cosine',
                'lr_scheduler_min_lr': 1e-6,
            },
            # 더 깊은 네트워크
            {
                'max_batch_size': 512,
                'num_epochs': 20,
                'epochs_wo_improve': 5,
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'dropout_prob': 0.2,
                'num_layers': 6,
                'hidden_size': 256,
                'activation': 'relu',
                'lr_scheduler': True,
                'scheduler_type': 'cosine',
                'lr_scheduler_min_lr': 1e-6,
            },
            # 더 낮은 학습률
            {
                'max_batch_size': 512,
                'num_epochs': 20,
                'epochs_wo_improve': 5,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
                'weight_decay': 0.0001,
                'dropout_prob': 0.1,
                'num_layers': 4,
                'hidden_size': 128,
                'activation': 'relu',
                'lr_scheduler': True,
                'scheduler_type': 'cosine',
                'lr_scheduler_min_lr': 1e-6,
            },
        ],
        
        # === CustomNNTorch 하이퍼파라미터 검색 ===
        "CUSTOM_NN_TORCH_SEARCH": [
            # 기본 설정
            {
                'max_batch_size': 512,
                'num_epochs': 20,
                'epochs_wo_improve': 5,
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'dropout_prob': 0.1,
                'num_layers': 4,
                'hidden_size': 128,
                'activation': 'relu',
                'lr_scheduler': True,
                'scheduler_type': 'cosine',
                'lr_scheduler_min_lr': 1e-6,
            },
            # 더 깊은 네트워크
            {
                'max_batch_size': 512,
                'num_epochs': 20,
                'epochs_wo_improve': 5,
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'dropout_prob': 0.2,
                'num_layers': 6,
                'hidden_size': 256,
                'activation': 'relu',
                'lr_scheduler': True,
                'scheduler_type': 'cosine',
                'lr_scheduler_min_lr': 1e-6,
            },
            # SGD 옵티마이저
            {
                'max_batch_size': 512,
                'num_epochs': 20,
                'epochs_wo_improve': 5,
                'optimizer': 'sgd',
                'learning_rate': 0.01,
                'weight_decay': 0.0001,
                'dropout_prob': 0.1,
                'num_layers': 4,
                'hidden_size': 128,
                'activation': 'relu',
                'lr_scheduler': True,
                'scheduler_type': 'cosine',
                'lr_scheduler_min_lr': 1e-6,
            },
        ],
    }
    
    return configs

def run_hyperparameter_search():
    """하이퍼파라미터 검색 실행"""
    print("=== 하이퍼파라미터 검색 시작 ===")
    
    # 데이터 로드
    train_data, test_data = load_data()
    
    # 커스텀 모델 등록
    register_custom_models()
    
    # 하이퍼파라미터 설정 생성
    configs = create_hyperparameter_configs()
    
    results = []
    
    # 각 모델별로 하이퍼파라미터 검색
    for model_name, model_configs in configs.items():
        print(f"\n=== {model_name} 하이퍼파라미터 검색 ===")
        
        for i, config in enumerate(model_configs):
            print(f"\n--- 설정 {i+1}/{len(model_configs)} ---")
            print(f"설정: {config}")
            
            # 모델명 생성
            current_model_name = f"{model_name}_config_{i+1}"
            
            # 예측기 생성
            predictor = TabularPredictor(
                label='Class',
                eval_metric='f1',
                path=f"models/hyperparameter_search/{current_model_name}",
                verbosity=2  # 로그 레벨 낮춤
            )
            
            try:
                # AutoGluon 방식으로 하이퍼파라미터 전달
                if model_name == "DCNV2_SEARCH":
                    hyperparameters = {
                        "DCNV2": [config]
                    }
                elif model_name == "CUSTOM_FOCAL_DL_SEARCH":
                    hyperparameters = {
                        "CUSTOM_FOCAL_DL": [config]
                    }
                elif model_name == "CUSTOM_NN_TORCH_SEARCH":
                    hyperparameters = {
                        "CUSTOM_NN_TORCH": [config]
                    }
                
                # 학습 실행
                predictor.fit(
                    train_data=train_data,
                    hyperparameters=hyperparameters,
                    time_limit=300,  # 5분 제한
                    verbosity=2
                )
                
                # 성능 평가
                leaderboard = predictor.leaderboard()
                best_score = leaderboard.iloc[0]['score_val']
                fit_time = leaderboard.iloc[0]['fit_time_marginal']
                
                print(f"✅ 성능: {best_score:.4f}, 학습시간: {fit_time:.2f}초")
                
                # 결과 저장
                results.append({
                    'model_name': current_model_name,
                    'config': config,
                    'f1_score': best_score,
                    'fit_time': fit_time,
                    'config_num': i+1
                })
                
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                results.append({
                    'model_name': current_model_name,
                    'config': config,
                    'f1_score': 0.0,
                    'fit_time': 0.0,
                    'config_num': i+1,
                    'error': str(e)
                })
    
    # 결과 정리 및 출력
    print("\n=== 하이퍼파라미터 검색 결과 ===")
    results_df = pd.DataFrame(results)
    
    # 성공한 실험만 필터링
    successful_results = results_df[results_df['f1_score'] > 0]
    
    if len(successful_results) > 0:
        print("\n📊 성공한 실험 결과:")
        for _, row in successful_results.iterrows():
            print(f"{row['model_name']}: F1={row['f1_score']:.4f}, 시간={row['fit_time']:.2f}초")
        
        # 최고 성능 모델
        best_model = successful_results.loc[successful_results['f1_score'].idxmax()]
        print(f"\n🏆 최고 성능 모델: {best_model['model_name']}")
        print(f"F1 Score: {best_model['f1_score']:.4f}")
        print(f"학습 시간: {best_model['fit_time']:.2f}초")
        print(f"설정: {best_model['config']}")
    
    # 전체 결과 저장
    results_df.to_csv('experiments/hyperparameter_search_results.csv', index=False)
    print(f"\n📁 결과가 'experiments/hyperparameter_search_results.csv'에 저장되었습니다.")
    
    return results_df

if __name__ == "__main__":
    run_hyperparameter_search() 