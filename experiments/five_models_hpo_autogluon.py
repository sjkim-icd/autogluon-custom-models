import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_dcnv2_torch_model import TabularDCNv2TorchModel
from custom_models.tabular_dcnv2_fuxictr_torch_model_fixed import TabularDCNv2FuxiCTRTorchModel
from custom_models.focal_loss_implementation import CustomFocalDLModel
from custom_models.custom_nn_torch_model import CustomNNTorchModel
from sklearn.model_selection import train_test_split
from autogluon.common import space

# 모델 등록
ag_model_registry.add(TabularDCNv2TorchModel)
ag_model_registry.add(TabularDCNv2FuxiCTRTorchModel)
ag_model_registry.add(CustomFocalDLModel)
ag_model_registry.add(CustomNNTorchModel)

def load_data():
    """데이터 로드"""
    print("=== 데이터 로드 ===")
    
    # 데이터 로드
    df = pd.read_csv("datasets/creditcard.csv")
    df["Class"] = df["Class"].astype("category")

    # 학습/테스트 분할
    train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)
    
    print(f"학습 데이터 크기: {train_data.shape}")
    print(f"테스트 데이터 크기: {test_data.shape}")
    print(f"학습 데이터 클래스 분포:\n{train_data['Class'].value_counts()}")
    print(f"학습 데이터 클래스 비율:\n{train_data['Class'].value_counts(normalize=True)}")
    
    return train_data, test_data

def run_five_models_hpo():
    """5개 모델 HPO 실행"""
    print("=== 5개 모델 HPO 시작 ===")
    
    # 데이터 로드
    train_data, test_data = load_data()
    
    # AutoGluon HPO 실행
    predictor = TabularPredictor(
        label='Class',
        problem_type='binary',
        eval_metric='f1',
        path="models/five_models_hpo_autogluon",
        verbosity=4
    )
    
    print("\n=== AutoGluon HPO 시작 (5개 모델) ===")
    predictor.fit(
        train_data=train_data,
        hyperparameters={
           #DCNv2 - 광범위한 HPO
            "DCNV2": {
                "num_cross_layers": space.Categorical(2, 3, 4),
                "cross_dropout": space.Categorical(0.0, 0.1, 0.2),
                "low_rank": space.Categorical(16, 32, 64),
                "deep_output_size": space.Categorical(64, 128, 256),
                "deep_hidden_size": space.Categorical(64, 128, 256),
                "deep_dropout": space.Categorical(0.1, 0.2, 0.3),
                "deep_layers": space.Categorical(2, 3, 4),
                "learning_rate": space.Real(1e-4, 1e-2, log=True),
                "weight_decay": space.Real(1e-6, 1e-4, log=True),
                "dropout_prob": space.Categorical(0.1, 0.2, 0.3),
                "activation": space.Categorical("relu", "elu"),
                "use_batchnorm": space.Categorical(True, False),
                "hidden_size": space.Categorical(128, 256),
                "num_epochs": space.Categorical(15, 20, 25),
                "epochs_wo_improve": space.Categorical(5, 10),
            },
            
            # # DCNv2 FuxiCTR - MoE 구조 포함
            "DCNV2_FUXICTR": {
                "num_cross_layers": space.Categorical(2, 3, 4),
                "cross_dropout": space.Categorical(0.0, 0.1, 0.2),
                "low_rank": space.Categorical(16, 32, 64),
                "deep_output_size": space.Categorical(64, 128, 256),
                "deep_hidden_size": space.Categorical(64, 128, 256),
                "deep_dropout": space.Categorical(0.1, 0.2, 0.3),
                "deep_layers": space.Categorical(2, 3, 4),
                "learning_rate": space.Real(1e-4, 1e-2, log=True),
                "weight_decay": space.Real(1e-6, 1e-4, log=True),
                "dropout_prob": space.Categorical(0.1, 0.2, 0.3),
                "activation": space.Categorical("relu", "elu"),
                "use_batchnorm": space.Categorical(True, False),
                "hidden_size": space.Categorical(128, 256),
                "num_epochs": space.Categorical(15, 20, 25),
                "epochs_wo_improve": space.Categorical(5, 10),
                # FuxiCTR 특화 파라미터
                "use_low_rank_mixture": space.Categorical(True, False),
                "num_experts": space.Categorical(2, 4, 8),
                "model_structure": space.Categorical("parallel", "stacked"),
            },
            
            # Focal Loss - 불균형 데이터에 특화
            "CUSTOM_FOCAL_DL": {
                "learning_rate": space.Real(1e-4, 1e-2, log=True),
                "weight_decay": space.Real(1e-6, 1e-3, log=True),
                "dropout_prob": space.Categorical(0.1, 0.2, 0.3),
                "num_layers": space.Categorical(3, 4, 5),
                "hidden_size": space.Categorical(128, 256, 512),
                "activation": space.Categorical("relu", "elu"),
                "num_epochs": space.Categorical(15, 20, 25),
                "epochs_wo_improve": space.Categorical(5, 10),
                "max_batch_size": space.Categorical(256, 512, 1024),
                # Focal Loss 특화 파라미터
                "focal_alpha": space.Categorical(0.25, 0.5, 0.75, 1.0),
                "focal_gamma": space.Categorical(1.0, 2.0, 3.0),
            },
            
            # 일반 Neural Network (CrossEntropy)
            "CUSTOM_NN_TORCH": {
                "learning_rate": space.Real(1e-4, 1e-2, log=True),
                "weight_decay": space.Real(1e-6, 1e-3, log=True),
                "dropout_prob": space.Categorical(0.1, 0.2, 0.3),
                "num_layers": space.Categorical(3, 4, 5),
                "hidden_size": space.Categorical(128, 256, 512),
                "activation": space.Categorical("relu", "elu"),
                "num_epochs": space.Categorical(15, 20, 25),
                "epochs_wo_improve": space.Categorical(5, 10),
                "max_batch_size": space.Categorical(256, 512, 1024),
            },
            
            # RandomForest - 트리 기반 모델
            "RF": {
                "n_estimators": space.Categorical(50, 100, 200),
                "max_depth": space.Categorical(5, 10, 15, None),
                "min_samples_split": space.Categorical(2, 5, 10),
                "min_samples_leaf": space.Categorical(1, 2, 4),
                "criterion": space.Categorical("gini", "entropy"),
                "max_features": space.Categorical("sqrt", "log2", None),
            },
        },
        hyperparameter_tune_kwargs={
            'scheduler': 'local',
            'searcher': 'random',  # AutoGluon 1.3.1에서는 random만 지원
            'num_trials': 20 # 각 모델별 1 trial씩
        },
        time_limit=8000,  # 30분 (충분한 시간)
        verbosity=4
    )
    
    return predictor, test_data

def analyze_results(predictor, test_data):
    """결과 분석"""
    print("\n=== HPO 완료! 결과 분석 ===")
    
    # 검증 데이터 기준 리더보드
    print("리더보드 (검증 데이터 기준):")
    leaderboard_val = predictor.leaderboard()
    print(leaderboard_val)
    
    # 테스트 데이터 기준 리더보드
    print("\n리더보드 (테스트 데이터 기준):")
    leaderboard_test = predictor.leaderboard(data=test_data)
    print(leaderboard_test)
    
    # 최고 성능 모델 정보
    print("\n=== 최고 성능 모델 정보 ===")
    best_model_val = leaderboard_val.loc[leaderboard_val['score_val'].idxmax()]
    best_model_test = leaderboard_test.loc[leaderboard_test['score_val'].idxmax()]
    
    print(f"검증 데이터 최고 성능: {best_model_val['model']} (F1 = {best_model_val['score_val']:.4f})")
    print(f"테스트 데이터 최고 성능: {best_model_test['model']} (F1 = {best_model_test['score_val']:.4f})")
    
    # 모델별 성능 비교
    print("\n=== 모델별 성능 비교 (검증 데이터) ===")
    for idx, row in leaderboard_val.iterrows():
        if 'WeightedEnsemble' not in row['model']:
            print(f"{row['model']}: F1 = {row['score_val']:.4f}, 시간 = {row['fit_time_marginal']:.2f}초")
    
    print("\n=== 모델별 성능 비교 (테스트 데이터) ===")
    for idx, row in leaderboard_test.iterrows():
        if 'WeightedEnsemble' not in row['model']:
            print(f"{row['model']}: F1 = {row['score_val']:.4f}")
    
    # Focal Loss vs 일반 CrossEntropy 비교
    print("\n=== Focal Loss vs 일반 CrossEntropy 비교 ===")
    focal_score_val = None
    nn_torch_score_val = None
    
    for idx, row in leaderboard_val.iterrows():
        if 'CUSTOM_FOCAL_DL' in row['model']:
            focal_score_val = row['score_val']
        elif 'CUSTOM_NN_TORCH' in row['model']:
            nn_torch_score_val = row['score_val']
    
    if focal_score_val is not None and nn_torch_score_val is not None:
        print(f"Focal Loss: {focal_score_val:.4f}")
        print(f"일반 CrossEntropy: {nn_torch_score_val:.4f}")
        if focal_score_val > nn_torch_score_val:
            print("✅ Focal Loss가 더 우수한 성능을 보였습니다!")
        elif focal_score_val < nn_torch_score_val:
            print("❌ 일반 CrossEntropy가 더 우수한 성능을 보였습니다.")
        else:
            print("🤝 두 모델의 성능이 동일합니다.")
    
    # DCNv2 vs DCNv2_FUXICTR 비교
    print("\n=== DCNv2 vs DCNv2_FUXICTR 비교 ===")
    dcnv2_score_val = None
    dcnv2_fuxictr_score_val = None
    
    for idx, row in leaderboard_val.iterrows():
        if 'DCNV2' in row['model'] and 'FUXICTR' not in row['model']:
            dcnv2_score_val = row['score_val']
        elif 'DCNV2_FUXICTR' in row['model']:
            dcnv2_fuxictr_score_val = row['score_val']
    
    if dcnv2_score_val is not None and dcnv2_fuxictr_score_val is not None:
        print(f"DCNv2: {dcnv2_score_val:.4f}")
        print(f"DCNv2_FUXICTR: {dcnv2_fuxictr_score_val:.4f}")
        if dcnv2_fuxictr_score_val > dcnv2_score_val:
            print("✅ FuxiCTR 구조가 더 우수한 성능을 보였습니다!")
        elif dcnv2_fuxictr_score_val < dcnv2_score_val:
            print("❌ 기본 DCNv2가 더 우수한 성능을 보였습니다.")
        else:
            print("🤝 두 모델의 성능이 동일합니다.")

if __name__ == "__main__":
    print("=== 5개 모델 AutoGluon HPO 시작 ===")
    print("모델: DCNV2, DCNV2_FUXICTR, CUSTOM_FOCAL_DL, CUSTOM_NN_TORCH, RF")
    print("HPO 방식: AutoGluon 내장 랜덤 서치")
    print("탐색 횟수: 1 trial (빠른 테스트)")
    print("시간 제한: 6분 (빠른 테스트)")
    print("예상 소요 시간: 5-6분")
    print("✅ 빠른 테스트로 모든 모델이 제대로 작동하는지 확인")
    
    # HPO 실행
    predictor, test_data = run_five_models_hpo()
    
    # 결과 분석
    analyze_results(predictor, test_data)
    
    print("\n=== 5개 모델 HPO 완료! ===") 