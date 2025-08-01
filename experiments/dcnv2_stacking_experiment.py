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

# Ray 초기화 시 Python 경로 설정
import ray
if ray.is_initialized():
    ray.shutdown()

# Ray 초기화 시 runtime_env를 사용하여 Python 경로 설정
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

print("\n=== DCNv2 모델 Stacking 실험 (stacking=2) ===")

# DCNv2 모델만 사용하여 stacking=2로 학습
predictor = TabularPredictor(
    label="Class", 
    problem_type="binary", 
    eval_metric="f1", 
    path="models/dcnv2_stacking_experiment"
).fit(
    train_data,
    hyperparameters={
        # DCNv2 - 수치형 특성에 적합한 모델
        "DCNV2": {
            "num_cross_layers": 3,
            "cross_dropout": 0.1,
            "low_rank": 16,
            "deep_output_size": 64,
            "deep_hidden_size": 64,
            "deep_dropout": 0.1,
            "deep_layers": 2,
            'epochs_wo_improve': 10,  # 조기 종료 설정
            'num_epochs': 30,         # 충분한 학습 시간
            "lr_scheduler": True,
            "scheduler_type": "cosine",
            "lr_scheduler_min_lr": 1e-6,
            "learning_rate": 0.0001,
        },
    },
    time_limit=1800,  # 30분
    verbosity=4,  # 최고 verbosity (가장 자세한 로그)
    num_stack_levels=2,  # stacking=2 설정
    num_bag_folds=2,  # 2-fold cross validation
    num_bag_sets=2,  # bagging 세트 수
    raise_on_no_models_fitted=False,  # 모델 학습 실패 시에도 계속 진행
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

print("\n=== Stacking 모델 성능 분석 ===")
print("모델별 성능 순위:")
for idx, row in leaderboard.iterrows():
    print(f"{idx+1}. {row['model']}: F1 = {row['score_val']:.4f}")

# Stacking 모델과 개별 모델 비교
stacking_models = [row for idx, row in leaderboard.iterrows() if 'Stack' in row['model']]
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

print("\n=== DCNv2 Stacking 실험 완료! ===") 