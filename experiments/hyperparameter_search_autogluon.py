import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import ray
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_dcnv2_torch_model import TabularDCNv2TorchModel
from custom_models.focal_loss_implementation import CustomFocalDLModel
from custom_models.custom_nn_torch_model import CustomNNTorchModel
from sklearn.model_selection import train_test_split
from autogluon.common import space

# Ray 초기화
# ray.init(num_cpus=2, ignore_reinit_error=True)

# 모델 등록
ag_model_registry.add(TabularDCNv2TorchModel)
ag_model_registry.add(CustomFocalDLModel)
ag_model_registry.add(CustomNNTorchModel)

def load_data():
    """데이터 로드"""
    print("=== 데이터 로드 ===")
    
    # 데이터 로드
    df = pd.read_csv("datasets/creditcard.csv")
    df["Class"] = df["Class"].astype("category")  # AutoGluon에서 분류로 인식하게

    # 학습/검증 분리
    train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)
    
    print(f"학습 데이터 크기: {train_data.shape}")
    print(f"테스트 데이터 크기: {test_data.shape}")
    print(f"학습 데이터 클래스 분포:\n{train_data['Class'].value_counts()}")
    print(f"학습 데이터 클래스 비율:\n{train_data['Class'].value_counts(normalize=True)}")
    
    return train_data, test_data

def register_custom_models():
    """커스텀 모델 등록"""
    print("=== 커스텀 모델 등록 ===")
    
    # DCNv2 모델 등록 확인
    from custom_models.tabular_dcnv2_torch_model import TabularDCNv2TorchModel
    from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
    
    # 모델이 이미 등록되어 있는지 확인
    if ag_model_registry.exists(TabularDCNv2TorchModel):
        print("✅ DCNv2 모델이 이미 등록되어 있음")
    else:
        ag_model_registry.add(TabularDCNv2TorchModel)
        print("✅ DCNv2 모델 등록 완료")

def run_hyperparameter_search():
    """하이퍼파라미터 검색 실행 - AutoGluon 형식 [,,] 사용"""
    print("=== 하이퍼파라미터 검색 시작 (AutoGluon 형식) ===")
    
    # 데이터 로드
    train_data, test_data = load_data()
    
    # 커스텀 모델 등록
    register_custom_models()
    
    # AutoGluon 형식으로 하이퍼파라미터 설정 - fit() 안에서 직접 전달
    print("=== 하이퍼파라미터 검색 공간 ===")
    print("DCNv2: 개별 파라미터별 검색 (cross_layers, dropout, learning_rate 등)")
    print("CustomFocalDL: 개별 파라미터별 검색 (layers, hidden_size, learning_rate 등)")
    print("CustomNNTorch: 개별 파라미터별 검색 (optimizer, layers, learning_rate 등)")
    print("RandomForest: 개별 파라미터별 검색 (n_estimators, max_depth, criterion 등)")
    print("총 4개 모델의 하이퍼파라미터 조합 테스트")
    print("제한된 병렬 처리: 2개 CPU 코어 사용 (메모리 부족으로)")
    
    # 예측기 생성
    predictor = TabularPredictor(
        label='Class',
        eval_metric='f1'
        # path="models/hyperparameter_search_autogluon",
        # verbosity=5  # HPO에서 epoch 로그를 보기 위해 높은 verbosity
    )
    
    # 학습 실행 - AutoGluon 형식으로 하이퍼파라미터 전달 (제한된 병렬 처리)
    print("\n=== AutoGluon 하이퍼파라미터 검색 시작 (제한된 병렬 처리) ===")
    predictor.fit(
        train_data=train_data,
        hyperparameters={
            # DCNv2 하이퍼파라미터 검색 공간 - 개별 파라미터별로 []
            "DCNV2": {
                "num_cross_layers": space.Int(2, 3, default=2),
                "cross_dropout": space.Real(0.1, 0.2, default=0.1),
                "low_rank": space.Int(16, 32, default=16),
                "deep_output_size": space.Int(64, 128, default=64),
                "deep_hidden_size": space.Int(64, 128, default=64),
                "deep_dropout": space.Real(0.1, 0.2, default=0.1),
                "deep_layers": space.Int(2, 3, default=2),
                "learning_rate": space.Real(0.0001, 0.001, default=0.0001),
            },
            
            # CustomFocalDL 하이퍼파라미터 검색 공간 - space 형태로 정의
            "CUSTOM_FOCAL_DL": {
                # 고정값들 (HPO에서 제외) - 필요시 주석 해제
                # 'max_batch_size': space.Int(512, 512, default=512),
                # 'num_epochs': space.Int(20, 20, default=20),
                # 'epochs_wo_improve': space.Int(5, 5, default=5),
                # 'optimizer': space.Categorical('adam'),
                # 'weight_decay': space.Real(0.0001, 0.0001, default=0.0001),
                # 'lr_scheduler': space.Categorical(True),
                # 'scheduler_type': space.Categorical('cosine'),
                # 'lr_scheduler_min_lr': space.Real(1e-6, 1e-6, default=1e-6),
                
                # 튜닝 대상 파라미터들
                'learning_rate': space.Real(0.0001, 0.001, default=0.0001),
                'dropout_prob': space.Real(0.1, 0.2, default=0.1),
                'num_layers': space.Int(4, 6, default=4),
                'hidden_size': space.Int(128, 256, default=128),
                'activation': space.Categorical('relu'),
            },
            
            # # CustomNNTorch 하이퍼파라미터 검색 공간 - space 형태로 정의
            "CUSTOM_NN_TORCH": {
                # 고정값들 (HPO에서 제외) - 필요시 주석 해제
                # 'max_batch_size': space.Int(512, 512, default=512),
                # 'num_epochs': space.Int(20, 20, default=20),
                # 'epochs_wo_improve': space.Int(5, 5, default=5),
                # 'weight_decay': space.Real(0.0001, 0.0001, default=0.0001),
                # 'lr_scheduler': space.Categorical(True),
                # 'scheduler_type': space.Categorical('cosine'),
                # 'lr_scheduler_min_lr': space.Real(1e-6, 1e-6, default=1e-6),
                
                # 튜닝 대상 파라미터들
                'optimizer': space.Categorical('adam', 'sgd'),
                'learning_rate': space.Real(0.001, 0.01, default=0.001),
                'dropout_prob': space.Real(0.1, 0.2, default=0.1),
                'num_layers': space.Int(4, 6, default=4),
                'hidden_size': space.Int(128, 256, default=128),
                'activation': space.Categorical('relu'),
            },
            
            # RandomForest 하이퍼파라미터 검색 공간 - space 형태로 정의
            "RF": {
                "n_estimators": space.Int(100, 300, default=100),
                "max_depth": space.Int(10, 20, default=10),
                "min_samples_split": space.Int(2, 10, default=2),
                "min_samples_leaf": space.Int(1, 4, default=1),
                "criterion": space.Categorical("gini", "entropy"),
            },
        },
        # time_limit=180,  # 30분 제한
        hyperparameter_tune_kwargs={
            'scheduler': 'local',
            'searcher': 'random',
            'num_trials': 20,  # 빠른 테스트를 위해 2번 시도로 제한
        },
        num_cpus=2,       # 메모리 부족으로 2개 CPU만 사용
        num_gpus=0,       # GPU 없음
        verbosity=5       # HPO에서 epoch 로그를 보기 위해 높은 verbosity
    )
    
    # 결과 출력
    print("\n=== 하이퍼파라미터 검색 결과 ===")
    leaderboard = predictor.leaderboard()
    print(leaderboard)
    
    # 상세 분석
    print("\n=== 모델별 상세 분석 ===")
    for idx, row in leaderboard.iterrows():
        model_name = row['model']
        score = row['score_val']
        fit_time = row['fit_time_marginal']
        print(f"\n{model_name}:")
        print(f"  - F1 Score: {score:.4f}")
        print(f"  - 학습 시간: {fit_time:.2f}초")
    
    # 최고 성능 모델
    best_row = leaderboard.loc[leaderboard['score_val'].idxmax()]
    print(f"\n🏆 최고 성능 모델: {best_row['model']}")
    print(f"F1 Score: {best_row['score_val']:.4f}")
    print(f"학습 시간: {best_row['fit_time_marginal']:.2f}초")
    
    # 하이퍼파라미터별 성능 비교
    print("\n=== 하이퍼파라미터별 성능 비교 ===")
    for model_type in ['DCNV2', 'CUSTOM_FOCAL_DL', 'CUSTOM_NN_TORCH', 'RF']:
        model_results = leaderboard[leaderboard['model'].str.contains(model_type)]
        if len(model_results) > 0:
            print(f"\n{model_type} 결과:")
            for _, row in model_results.iterrows():
                print(f"  {row['model']}: F1={row['score_val']:.4f}, 시간={row['fit_time_marginal']:.2f}초")
    
    # 예측 테스트
    print("\n=== 예측 테스트 ===")
    predictions = predictor.predict(test_data)
    probabilities = predictor.predict_proba(test_data)
    
    print(f"예측 결과 형태: {predictions.shape}")
    print(f"확률 예측 형태: {probabilities.shape}")
    print(f"예측값 샘플: {predictions.head()}")
    print(f"확률값 샘플:\n{probabilities.head()}")
    
    return predictor, leaderboard

if __name__ == "__main__":
    run_hyperparameter_search() 