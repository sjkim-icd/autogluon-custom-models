import os
import pickle
import torch
import numpy as np
from autogluon.tabular import TabularPredictor

def check_dcn_model_status():
    """DCN 모델의 학습 상태 확인"""
    
    # 모델 경로
    model_path = "models/five_models_hpo_autogluon"
    
    # 최고 성능 DCN 모델들
    best_dcn_models = [
        "DCNV2\\c30a7_00010",  # 검증 최고
        "DCNV2\\c30a7_00007",  # 테스트 최고
        "DCNV2\\c30a7_00006",  # 검증 2위
    ]
    
    print("=== DCN 모델 학습 상태 확인 ===")
    print()
    
    # Predictor 로드
    try:
        predictor = TabularPredictor.load(model_path)
        print(f"✅ Predictor 로드 성공: {model_path}")
        print()
    except Exception as e:
        print(f"❌ Predictor 로드 실패: {e}")
        return
    
    # 각 모델 확인
    for model_name in best_dcn_models:
        print(f"🔍 모델 확인: {model_name}")
        print("-" * 50)
        
        try:
            # 모델 파일 경로
            model_file_path = os.path.join(model_path, "models", model_name, "model.pkl")
            
            if os.path.exists(model_file_path):
                print(f"✅ 모델 파일 존재: {model_file_path}")
                
                # 모델 로드
                with open(model_file_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                print(f"📊 모델 타입: {type(model_data)}")
                
                # 모델 속성 확인
                if hasattr(model_data, 'model'):
                    model = model_data.model
                    print(f"📈 모델 구조: {type(model)}")
                    
                    # 학습 관련 속성들 확인
                    if hasattr(model, 'epochs_trained'):
                        print(f"🎯 학습된 에포크: {model.epochs_trained}")
                    
                    if hasattr(model, 'best_epoch'):
                        print(f"🏆 최고 성능 에포크: {model.best_epoch}")
                    
                    if hasattr(model, 'early_stopping_counter'):
                        print(f"⏹️ Early stopping 카운터: {model.early_stopping_counter}")
                    
                    if hasattr(model, 'learning_rate'):
                        print(f"📚 학습률: {model.learning_rate}")
                    
                    if hasattr(model, 'weight_decay'):
                        print(f"🔒 Weight decay: {model.weight_decay}")
                    
                    if hasattr(model, 'dropout_prob'):
                        print(f"💧 Dropout: {model.dropout_prob}")
                    
                    # 모델 가중치 확인
                    if hasattr(model, 'state_dict'):
                        state_dict = model.state_dict()
                        print(f"📦 모델 파라미터 수: {len(state_dict)}")
                        
                        # 파라미터 통계
                        total_params = 0
                        for name, param in state_dict.items():
                            if 'weight' in name:
                                total_params += param.numel()
                        print(f"🔢 총 가중치 파라미터: {total_params:,}")
                
                # 모델 성능 확인
                if hasattr(model_data, 'val_score'):
                    print(f"📊 검증 성능: {model_data.val_score}")
                
                if hasattr(model_data, 'test_score'):
                    print(f"📊 테스트 성능: {model_data.test_score}")
                
            else:
                print(f"❌ 모델 파일 없음: {model_file_path}")
                
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
        
        print()
    
    # 전체 모델 리스트 확인
    print("📋 전체 모델 리스트:")
    print("-" * 50)
    
    try:
        leaderboard = predictor.leaderboard()
        dcn_models = leaderboard[leaderboard['model'].str.contains('DCNV2')]
        
        for idx, row in dcn_models.iterrows():
            print(f"{row['model']:<35} | 검증 F1: {row['score_val']:.4f} | 시간: {row['fit_time_marginal']:.1f}초")
    
    except Exception as e:
        print(f"❌ 리더보드 로드 실패: {e}")

def check_model_files():
    """모델 파일 구조 확인"""
    
    model_path = "models/five_models_hpo_autogluon"
    
    print("=== 모델 파일 구조 확인 ===")
    print()
    
    if os.path.exists(model_path):
        print(f"📁 모델 디렉토리: {model_path}")
        
        # DCN 모델 디렉토리 확인
        dcn_path = os.path.join(model_path, "models", "DCNV2")
        if os.path.exists(dcn_path):
            print(f"📁 DCNV2 디렉토리: {dcn_path}")
            
            # 하위 디렉토리 확인
            subdirs = [d for d in os.listdir(dcn_path) if os.path.isdir(os.path.join(dcn_path, d))]
            print(f"📂 DCNV2 하위 디렉토리: {len(subdirs)}개")
            
            for subdir in subdirs[:5]:  # 처음 5개만
                subdir_path = os.path.join(dcn_path, subdir)
                model_file = os.path.join(subdir_path, "model.pkl")
                
                if os.path.exists(model_file):
                    file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
                    print(f"  📄 {subdir}/model.pkl ({file_size:.1f}MB)")
                else:
                    print(f"  ❌ {subdir}/model.pkl (없음)")
        else:
            print(f"❌ DCNV2 디렉토리 없음: {dcn_path}")
    else:
        print(f"❌ 모델 디렉토리 없음: {model_path}")

if __name__ == "__main__":
    check_model_files()
    print()
    check_dcn_model_status() 