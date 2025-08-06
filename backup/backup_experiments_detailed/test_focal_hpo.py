import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.tabular.registry import ag_model_registry
from custom_models.focal_loss_implementation import CustomFocalDLModel
from sklearn.model_selection import train_test_split
from autogluon.common import space

# 모델 등록
ag_model_registry.add(CustomFocalDLModel)

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

def run_focal_hpo():
    """Focal Loss HPO 테스트"""
    print("=== Focal Loss HPO 테스트 시작 ===")
    
    # 데이터 로드
    train_data, test_data = load_data()
    
    # AutoGluon HPO 실행
    predictor = TabularPredictor(
        label='Class',
        problem_type='binary',
        eval_metric='f1',
        path="models/focal_hpo_test",
        verbosity=4
    )
    
    print("\n=== Focal Loss HPO 시작 ===")
    predictor.fit(
        train_data=train_data,
        hyperparameters={
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
        },
        hyperparameter_tune_kwargs={
            'scheduler': 'local',
            'searcher': 'random',
            'num_trials': 2,  # 5에서 2로 변경
        },
        time_limit=300,  # 5분 (매우 빠른 테스트)
        verbosity=4
    )
    
    return predictor, test_data

def analyze_focal_results(predictor, test_data):
    """Focal Loss 결과 분석"""
    print("\n=== Focal Loss HPO 완료! 결과 분석 ===")
    
    # 검증 데이터 기준 리더보드
    print("리더보드 (검증 데이터 기준):")
    leaderboard_val = predictor.leaderboard()
    print(leaderboard_val)
    
    # 테스트 데이터 기준 리더보드
    print("\n리더보드 (테스트 데이터 기준):")
    leaderboard_test = predictor.leaderboard(data=test_data)
    print(leaderboard_test)
    
    # 최고 성능 모델 정보
    print("\n=== 최고 성능 Focal Loss 모델 정보 ===")
    best_model_val = leaderboard_val.loc[leaderboard_val['score_val'].idxmax()]
    best_model_test = leaderboard_test.loc[leaderboard_test['score_val'].idxmax()]
    
    print(f"검증 데이터 최고 성능: {best_model_val['model']} (F1 = {best_model_val['score_val']:.4f})")
    print(f"테스트 데이터 최고 성능: {best_model_test['model']} (F1 = {best_model_test['score_val']:.4f})")
    
    # Focal Loss 파라미터 분석
    print("\n=== Focal Loss 파라미터 분석 ===")
    for idx, row in leaderboard_val.iterrows():
        if 'CUSTOM_FOCAL_DL' in row['model']:
            print(f"{row['model']}: F1 = {row['score_val']:.4f}, 시간 = {row['fit_time_marginal']:.2f}초")
    
    print("\n=== Focal Loss vs 이전 결과 비교 ===")
    print("이전 CUSTOM_FOCAL_DL 성능: 0.7500 (75.00%)")
    print(f"현재 최고 성능: {best_model_val['score_val']:.4f} ({best_model_val['score_val']*100:.2f}%)")
    
    if best_model_val['score_val'] > 0.7500:
        improvement = (best_model_val['score_val'] - 0.7500) * 100
        print(f"✅ 성능 향상: +{improvement:.2f}%p")
    else:
        print("❌ 성능 향상 없음")

if __name__ == "__main__":
    print("=== Focal Loss HPO 테스트 시작 ===")
    print("모델: CUSTOM_FOCAL_DL (수정된 버전)")
    print("HPO 방식: AutoGluon 내장 랜덤 서치")
    print("탐색 횟수: 5 trials")
    print("시간 제한: 5분")
    print("목적: Focal Loss 파라미터 수정이 제대로 작동하는지 확인")
    
    # HPO 실행
    predictor, test_data = run_focal_hpo()
    
    # 결과 분석
    analyze_focal_results(predictor, test_data)
    
    print("\n=== Focal Loss HPO 테스트 완료! ===") 