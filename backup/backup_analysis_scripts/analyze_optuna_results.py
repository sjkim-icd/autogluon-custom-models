import json
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from datetime import datetime

def load_optuna_results():
    """Optuna 결과 로드"""
    with open('best_individual_params.json', 'r') as f:
        best_params = json.load(f)
    return best_params

def analyze_individual_models():
    """개별 모델 분석"""
    best_params = load_optuna_results()
    
    # 개별 모델 결과 테이블 생성
    individual_results = []
    
    for model_type, result in best_params.items():
        individual_results.append({
            'Model': model_type,
            'Best_F1_Score': result['best_value'],
            'Learning_Rate': result['best_params'].get('learning_rate', 'N/A'),
            'Weight_Decay': result['best_params'].get('weight_decay', 'N/A'),
            'Dropout_Prob': result['best_params'].get('dropout_prob', 'N/A'),
            'Num_Layers': result['best_params'].get('num_layers', 'N/A'),
            'Hidden_Size': result['best_params'].get('hidden_size', 'N/A'),
            'Num_Epochs': result['best_params'].get('num_epochs', 'N/A'),
            'N_Estimators': result['best_params'].get('n_estimators', 'N/A'),
            'Max_Depth': result['best_params'].get('max_depth', 'N/A'),
            'Min_Samples_Split': result['best_params'].get('min_samples_split', 'N/A'),
            'Min_Samples_Leaf': result['best_params'].get('min_samples_leaf', 'N/A'),
            'Focal_Alpha': result['best_params'].get('focal_alpha', 'N/A'),
            'Focal_Gamma': result['best_params'].get('focal_gamma', 'N/A'),
        })
    
    return pd.DataFrame(individual_results)

def analyze_final_ensemble():
    """최종 앙상블 분석"""
    try:
        # 최종 앙상블 모델 로드
        final_predictor = TabularPredictor.load("models/final_autogluon_ensemble")
        
        # 리더보드 가져오기
        leaderboard = final_predictor.leaderboard()
        
        # 앙상블 결과 테이블 생성
        ensemble_results = []
        
        for idx, row in leaderboard.iterrows():
            ensemble_results.append({
                'Model_Name': row['model'],
                'Validation_F1': row['score_val'],
                'Training_Time': row['fit_time'],
                'Prediction_Time': row['pred_time_val'],
                'Stack_Level': row['stack_level'],
                'Can_Infer': row['can_infer'],
                'Fit_Order': row['fit_order']
            })
        
        return pd.DataFrame(ensemble_results)
    
    except Exception as e:
        print(f"앙상블 결과 로드 실패: {e}")
        return pd.DataFrame()

def create_detailed_report():
    """상세 리포트 생성"""
    print("=== Optuna HPO 결과 분석 ===")
    
    # 1. 개별 모델 결과
    individual_df = analyze_individual_models()
    print("\n📊 개별 모델 최적화 결과:")
    print(individual_df.to_string(index=False))
    
    # 2. 최종 앙상블 결과
    ensemble_df = analyze_final_ensemble()
    print("\n🏆 최종 앙상블 결과:")
    print(ensemble_df.to_string(index=False))
    
    # 3. 성능 비교
    print("\n📈 성능 비교:")
    print("=" * 50)
    for _, row in individual_df.iterrows():
        print(f"{row['Model']}: F1 = {row['Best_F1_Score']:.4f}")
    
    if not ensemble_df.empty:
        best_ensemble = ensemble_df.iloc[0]
        print(f"최종 앙상블: F1 = {best_ensemble['Validation_F1']:.4f}")
    
    # 4. 엑셀 파일로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optuna_results_{timestamp}.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 개별 모델 결과
        individual_df.to_excel(writer, sheet_name='Individual_Models', index=False)
        
        # 최종 앙상블 결과
        ensemble_df.to_excel(writer, sheet_name='Final_Ensemble', index=False)
        
        # 요약 시트
        summary_data = {
            'Metric': ['Best Individual Model', 'Best Individual F1', 'Final Ensemble F1', 'Improvement'],
            'Value': [
                individual_df.loc[individual_df['Best_F1_Score'].idxmax(), 'Model'],
                individual_df['Best_F1_Score'].max(),
                ensemble_df['Validation_F1'].iloc[0] if not ensemble_df.empty else 'N/A',
                f"{(ensemble_df['Validation_F1'].iloc[0] - individual_df['Best_F1_Score'].max())*100:.2f}%" if not ensemble_df.empty else 'N/A'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\n✅ 결과가 '{filename}'에 저장되었습니다!")
    
    return individual_df, ensemble_df

def print_optuna_trials_info():
    """Optuna trials 상세 정보 출력"""
    print("\n=== Optuna Trials 상세 정보 ===")
    
    import optuna
    
    # 각 모델별 study 로드
    model_types = ['DCNV2', 'CUSTOM_FOCAL_DL', 'RF']
    
    for model_type in model_types:
        try:
            study = optuna.load_study(study_name=f'{model_type}_hpo_study', storage=None)
            
            print(f"\n🔍 {model_type} Trials 정보:")
            print(f"  총 Trials: {len(study.trials)}")
            print(f"  최고 성능: {study.best_value:.4f}")
            print(f"  최적 파라미터: {study.best_params}")
            
            # 성공한 trials만 필터링
            successful_trials = [t for t in study.trials if t.value is not None]
            print(f"  성공한 Trials: {len(successful_trials)}")
            
            if successful_trials:
                values = [t.value for t in successful_trials]
                print(f"  성능 범위: {min(values):.4f} ~ {max(values):.4f}")
                print(f"  평균 성능: {np.mean(values):.4f}")
                print(f"  표준편차: {np.std(values):.4f}")
            
        except Exception as e:
            print(f"  {model_type} study 로드 실패: {e}")

if __name__ == "__main__":
    # 상세 리포트 생성
    individual_df, ensemble_df = create_detailed_report()
    
    # Optuna trials 상세 정보
    print_optuna_trials_info()
    
    print("\n🎉 분석 완료!") 