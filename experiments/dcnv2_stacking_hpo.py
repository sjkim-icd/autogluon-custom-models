import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.tabular_dcnv2_torch_model import TabularDCNv2TorchModel
from sklearn.model_selection import train_test_split
from autogluon.common import space

# Ray 병렬 처리에서 custom_models 모듈을 찾을 수 있도록 설정
import os
import sys

# 현재 프로젝트 루트 경로를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ray 초기화 시 runtime_env를 사용하여 Python 경로 설정
import ray
if ray.is_initialized():
    ray.shutdown()

ray.init(
    ignore_reinit_error=True,
    runtime_env={
        "env_vars": {
            "PYTHONPATH": project_root
        }
    }
)

# DCNv2 모델만 등록
ag_model_registry.add(TabularDCNv2TorchModel)

# 데이터 로드
df = pd.read_csv("../datasets/creditcard.csv")
df["Class"] = df["Class"].astype("category")  # AutoGluon에서 분류로 인식하게

# 학습/검증 분리
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)

print(f"학습 데이터 크기: {train_data.shape}")
print(f"테스트 데이터 크기: {test_data.shape}")
print(f"학습 데이터 클래스 분포:\n{train_data['Class'].value_counts()}")
print(f"학습 데이터 클래스 비율:\n{train_data['Class'].value_counts(normalize=True)}")

print("\n=== DCNv2 모델 Stacking + 하이퍼파라미터 최적화 실험 ===")

# DCNv2 모델만 사용하여 stacking=2로 학습 (하이퍼파라미터 검색 포함)
predictor = TabularPredictor(
    label="Class", 
    problem_type="binary", 
    eval_metric="f1", 
    path="models/dcnv2_stacking_hpo"
).fit(
    train_data,
    hyperparameters={
        # DCNv2 - 하이퍼파라미터 검색 공간 정의
        "DCNV2": {
            # 크로스 네트워크 파라미터
            "num_cross_layers": space.Int(1, 4),  # 1~4 레이어
            "cross_dropout": space.Real(0.05, 0.3),  # 0.05~0.3 드롭아웃
            "low_rank": space.Int(8, 32),  # 8~32 저차원 분해 크기
            
            # 딥 네트워크 파라미터
            "deep_output_size": space.Int(32, 128),  # 32~128 출력 크기
            "deep_hidden_size": space.Int(32, 128),  # 32~128 은닉층 크기
            "deep_dropout": space.Real(0.05, 0.3),  # 0.05~0.3 드롭아웃
            "deep_layers": space.Int(1, 3),  # 1~3 레이어
            
            # 학습 파라미터
            'epochs_wo_improve': space.Int(5, 15),  # 5~15 조기 종료
            'num_epochs': space.Int(20, 50),  # 20~50 총 에포크
            
            # 학습률 스케줄러 파라미터
            "lr_scheduler": True,
            "scheduler_type": "cosine",
            "lr_scheduler_min_lr": space.Real(1e-7, 1e-5),  # 1e-7~1e-5 최소 학습률
            "learning_rate": space.Real(1e-4, 1e-3),  # 1e-4~1e-3 초기 학습률
        },
    },
    time_limit=3600,  # 1시간 (하이퍼파라미터 검색을 위해 시간 증가)
    verbosity=4,  # 최고 verbosity (가장 자세한 로그)
    num_stack_levels=2,  # stacking=2 설정
    num_bag_folds=2,  # 2-fold cross validation
    num_bag_sets=2,  # bagging 세트 수
    raise_on_no_models_fitted=False,  # 모델 학습 실패 시에도 계속 진행
    hyperparameter_tune_kwargs={
        'scheduler': 'local',
        'searcher': 'random',
        'num_trials': 5,  # 10번의 하이퍼파라미터 조합 시도
        'max_reward': 1.0,  # 최대 보상 (F1 score)
        'time_attr': 'epoch',
        'grace_period': 5,  # 최소 5 에포크 후 조기 종료 가능
        'reduction_factor': 2,  # 성능이 좋지 않으면 2배 빠르게 조기 종료
    }
)

print("\n=== 학습 완료! 결과 확인 ===")
print("리더보드:")
print(predictor.leaderboard())

print("\n=== 모델별 상세 정보 ===")
leaderboard = predictor.leaderboard()
for idx, row in leaderboard.iterrows():
    model_name = row['model']
    score = row['score_val']
    fit_time = row['fit_time_marginal']
    print(f"\n{model_name}:")
    print(f"  - 성능 점수: {score:.4f}")
    print(f"  - 학습 시간: {fit_time:.2f}초")

print("\n=== 최고 성능 모델 ===")
# get_model_best() 대신 leaderboard에서 최고 성능 모델 찾기
best_row = leaderboard.loc[leaderboard['score_val'].idxmax()]
best_model = best_row['model']
print(f"최고 성능 모델: {best_model}")
print(f"성능 점수: {best_row['score_val']:.4f}")

print("\n=== 하이퍼파라미터 최적화 결과 ===")
print("모델별 성능 순위:")
for idx, row in leaderboard.iterrows():
    print(f"{idx+1}. {row['model']}: F1 = {row['score_val']:.4f}")

# Stacking 모델과 개별 모델 비교
stacking_models = [row for idx, row in leaderboard.iterrows() if 'Stack' in row['model'] or 'Ensemble' in row['model']]
individual_models = [row for idx, row in leaderboard.iterrows() if 'Stack' not in row['model'] and 'WeightedEnsemble' not in row['model']]

if stacking_models and individual_models:
    best_stacking = max(stacking_models, key=lambda x: x['score_val'])
    best_individual = max(individual_models, key=lambda x: x['score_val'])
    
    print(f"\n=== Stacking vs 개별 모델 비교 ===")
    print(f"최고 개별 모델: {best_individual['model']} (F1: {best_individual['score_val']:.4f})")
    print(f"최고 Stacking 모델: {best_stacking['model']} (F1: {best_stacking['score_val']:.4f})")
    
    if best_stacking['score_val'] > best_individual['score_val']:
        improvement = ((best_stacking['score_val'] - best_individual['score_val']) / best_individual['score_val']) * 100
        print(f"✅ Stacking이 개별 모델보다 {improvement:.2f}% 성능 향상을 보였습니다!")
    else:
        print("❌ 개별 모델이 Stacking보다 더 우수한 성능을 보였습니다.")

print("\n=== 예측 테스트 ===")
# 테스트 데이터로 예측
predictions = predictor.predict(test_data)
probabilities = predictor.predict_proba(test_data)

print(f"예측 결과 형태: {predictions.shape}")
print(f"확률 예측 형태: {probabilities.shape}")
print(f"예측값 샘플: {predictions.head()}")
print(f"확률값 샘플:\n{probabilities.head()}")

print("\n=== DCNv2 Stacking + HPO 실험 완료! ===") 