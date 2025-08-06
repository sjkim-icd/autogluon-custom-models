import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
# from custom_models.tabular_deepfm_torch_model import TabularDeepFMTorchModel  # 백업됨
from custom_models.tabular_dcnv2_torch_model import TabularDCNv2TorchModel
from custom_models.tabular_dcnv2_fuxictr_torch_model_fixed import TabularDCNv2FuxiCTRTorchModel
from custom_models.focal_loss_implementation import CustomFocalDLModel
from custom_models.custom_nn_torch_model import CustomNNTorchModel
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from autogluon.common import space

# 모델 등록
# ag_model_registry.add(TabularDeepFMTorchModel)  # 백업됨
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
    df = df.dropna(subset=['survived'])  # survived가 NaN인 행 제거
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
    print(f"훈련 데이터 Class 분포: {train_data['survived'].value_counts().to_dict()}")
    print(f"테스트 데이터 Class 분포: {test_data['survived'].value_counts().to_dict()}")
    
    return train_data, test_data

def run_five_models_hpo_experiment():
    """5개 모델 HPO 실험 (안정적 버전)"""
    print("=== Titanic 데이터셋 5개 모델 HPO 실험 시작 (안정적 버전) ===")
    
    # 데이터 로드
    train_data, test_data = load_data()
    
    # AutoGluon Predictor 설정
    predictor = TabularPredictor(
        label='survived',
        problem_type='binary',
        eval_metric='f1',
        path="models/five_models_titanic_hpo_fixed",
        verbosity=4
    )
    
    # 모델별 하이퍼파라미터 검색 공간 설정 (간소화)
    hyperparameters = {
        'DCNV2': {
            'learning_rate': space.Real(0.001, 0.01, log=True),
            'weight_decay': space.Real(1e-5, 0.001, log=True),
            'dropout_prob': space.Categorical(0.1, 0.2),
            'num_layers': space.Categorical(3, 4),
            'hidden_size': space.Categorical(128, 256),
            'num_epochs': space.Categorical(15, 20),
            'max_batch_size': space.Categorical(256, 512),
        },
        'DCNV2_FUXICTR': {
            'learning_rate': space.Real(0.001, 0.01, log=True),
            'weight_decay': space.Real(1e-5, 0.001, log=True),
            'dropout_prob': space.Categorical(0.1, 0.2),
            'num_layers': space.Categorical(3, 4),
            'hidden_size': space.Categorical(128, 256),
            'num_epochs': space.Categorical(15, 20),
            'max_batch_size': space.Categorical(256, 512),
        },
        'CUSTOM_FOCAL_DL': {
            'learning_rate': space.Real(0.001, 0.01, log=True),
            'weight_decay': space.Real(1e-5, 0.001, log=True),
            'dropout_prob': space.Categorical(0.1, 0.2),
            'num_layers': space.Categorical(3, 4),
            'hidden_size': space.Categorical(128, 256),
            'num_epochs': space.Categorical(15, 20),
            'max_batch_size': space.Categorical(256, 512),
            'focal_alpha': space.Categorical(0.25, 0.5, 0.75),
            'focal_gamma': space.Categorical(1.0, 2.0),
        },
        'CUSTOM_NN_TORCH': {
            'learning_rate': space.Real(0.001, 0.01, log=True),
            'weight_decay': space.Real(1e-5, 0.001, log=True),
            'dropout_prob': space.Categorical(0.1, 0.2),
            'num_layers': space.Categorical(3, 4),
            'hidden_size': space.Categorical(128, 256),
            'num_epochs': space.Categorical(15, 20),
            'max_batch_size': space.Categorical(256, 512),
        },
        'RF': {
            'n_estimators': space.Categorical(50, 100),
            'max_depth': space.Categorical(5, 10, None),
            'min_samples_split': space.Categorical(2, 5),
            'min_samples_leaf': space.Categorical(1, 2),
        }
    }
    
    # HPO 설정 (안정적 버전)
    hyperparameter_tune_kwargs = {
        'scheduler': 'local',
        'searcher': 'random',
        'num_trials': 2,  # 각 모델당 5번씩 시도 (줄임)
    }
    
    # 모델 학습 (HPO 적용)
    predictor.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        time_limit=600,  # 60분으로 늘림
        presets='medium_quality'  # best_quality 대신 medium_quality 사용
    )
    
    # 성능 평가
    print("\n=== 모델 성능 평가 ===")
    
    # 검증 데이터 성능
    val_scores = predictor.leaderboard()
    print("\n검증 데이터 성능:")
    print(val_scores)
    
    # 테스트 데이터 성능
    test_scores = predictor.evaluate(test_data, silent=True)
    print(f"\n테스트 데이터 성능:")
    print(f"F1 Score: {test_scores['f1']:.4f}")
    print(f"Accuracy: {test_scores['accuracy']:.4f}")
    print(f"Precision: {test_scores['precision']:.4f}")
    print(f"Recall: {test_scores['recall']:.4f}")
    
    # 개별 모델 성능 (상세 버전)
    print("\n=== 개별 모델 성능 ===")
    print("검증 데이터 기준 모델 순위:")
    for idx, row in val_scores.iterrows():
        print(f"  {row['model']}: F1 = {row['score_val']:.4f}")
    
    # Focal Loss vs 일반 CrossEntropy 비교
    print("\n=== Focal Loss vs 일반 CrossEntropy 비교 ===")
    focal_score_val = None
    nn_torch_score_val = None
    
    for idx, row in val_scores.iterrows():
        if 'CUSTOM_FOCAL_DL' in row['model']:
            focal_score_val = row['score_val']
        elif 'CUSTOM_NN_TORCH' in row['model']:
            nn_torch_score_val = row['score_val']
    
    if focal_score_val is not None and nn_torch_score_val is not None:
        print(f"CUSTOM_FOCAL_DL (Focal Loss): {focal_score_val:.4f}")
        print(f"CUSTOM_NN_TORCH (일반 CrossEntropy): {nn_torch_score_val:.4f}")
        if focal_score_val > nn_torch_score_val:
            print("✅ Focal Loss가 더 우수한 성능을 보였습니다!")
        elif focal_score_val < nn_torch_score_val:
            print("❌ 일반 CrossEntropy가 더 우수한 성능을 보였습니다.")
        else:
            print("🤝 두 모델의 성능이 동일합니다.")
    else:
        print("Focal Loss와 일반 CrossEntropy 비교를 위한 데이터가 부족합니다.")
    
    # 예측 테스트
    print("\n=== 예측 테스트 ===")
    predictions = predictor.predict(test_data)
    probabilities = predictor.predict_proba(test_data)
    
    print(f"예측 결과 형태: {predictions.shape}")
    print(f"확률 예측 형태: {probabilities.shape}")
    print(f"예측값 샘플: {predictions.head()}")
    print(f"확률값 샘플:\n{probabilities.head()}")
    
    # 예측 분포 확인
    print(f"\n예측 클래스 분포:")
    print(predictions.value_counts())
    
    print(f"\n실제 클래스 분포:")
    print(test_data['survived'].value_counts())
    
    print(f"\n전체 테스트 데이터 성능:")
    print(f"  F1 Score: {test_scores['f1']:.4f}")
    print(f"  Accuracy: {test_scores['accuracy']:.4f}")
    print(f"  Precision: {test_scores['precision']:.4f}")
    print(f"  Recall: {test_scores['recall']:.4f}")
    
    return predictor, test_data

if __name__ == "__main__":
    predictor, test_data = run_five_models_hpo_experiment() 