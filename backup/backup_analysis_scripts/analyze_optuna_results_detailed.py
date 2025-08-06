import json
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from datetime import datetime
import optuna

def load_optuna_results():
    """Optuna 결과 로드"""
    with open('best_individual_params.json', 'r') as f:
        best_params = json.load(f)
    return best_params

def get_all_trials_data():
    """각 모델별 모든 trial 데이터 수집"""
    model_types = ['DCNV2', 'CUSTOM_FOCAL_DL', 'RF']
    all_trials_data = {}
    
    for model_type in model_types:
        try:
            # Optuna study 로드
            study = optuna.load_study(study_name=f'{model_type}_hpo_study', storage=None)
            
            trials_data = []
            for trial in study.trials:
                if trial.value is not None:  # 성공한 trial만
                    trial_data = {
                        'Trial_Number': trial.number,
                        'F1_Score': trial.value,
                        'Status': 'COMPLETE'
                    }
                    
                    # 모델별 파라미터 추가
                    if model_type == 'DCNV2':
                        trial_data.update({
                            'Learning_Rate': trial.params.get('learning_rate', 'N/A'),
                            'Weight_Decay': trial.params.get('weight_decay', 'N/A'),
                            'Dropout_Prob': trial.params.get('dropout_prob', 'N/A'),
                            'Num_Layers': trial.params.get('num_layers', 'N/A'),
                            'Hidden_Size': trial.params.get('hidden_size', 'N/A'),
                            'Num_Epochs': trial.params.get('num_epochs', 'N/A'),
                        })
                    elif model_type == 'CUSTOM_FOCAL_DL':
                        trial_data.update({
                            'Learning_Rate': trial.params.get('learning_rate', 'N/A'),
                            'Weight_Decay': trial.params.get('weight_decay', 'N/A'),
                            'Dropout_Prob': trial.params.get('dropout_prob', 'N/A'),
                            'Num_Layers': trial.params.get('num_layers', 'N/A'),
                            'Hidden_Size': trial.params.get('hidden_size', 'N/A'),
                            'Num_Epochs': trial.params.get('num_epochs', 'N/A'),
                            'Focal_Alpha': trial.params.get('focal_alpha', 'N/A'),
                            'Focal_Gamma': trial.params.get('focal_gamma', 'N/A'),
                        })
                    elif model_type == 'RF':
                        trial_data.update({
                            'N_Estimators': trial.params.get('n_estimators', 'N/A'),
                            'Max_Depth': trial.params.get('max_depth', 'N/A'),
                            'Min_Samples_Split': trial.params.get('min_samples_split', 'N/A'),
                            'Min_Samples_Leaf': trial.params.get('min_samples_leaf', 'N/A'),
                        })
                    
                    trials_data.append(trial_data)
            
            all_trials_data[model_type] = pd.DataFrame(trials_data)
            print(f"✅ {model_type}: {len(trials_data)}개 trial 데이터 수집 완료")
            
        except Exception as e:
            print(f"❌ {model_type} study 로드 실패: {e}")
            all_trials_data[model_type] = pd.DataFrame()
    
    return all_trials_data

def analyze_individual_models():
    """개별 모델 분석 (최고 성능만)"""
    best_params = load_optuna_results()
    
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
        final_predictor = TabularPredictor.load("models/final_autogluon_ensemble")
        leaderboard = final_predictor.leaderboard()
        
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
    print("=== Optuna HPO 상세 결과 분석 ===")
    
    # 1. 모든 trial 데이터 수집
    all_trials_data = get_all_trials_data()
    
    # 2. 개별 모델 최고 성능 결과
    individual_df = analyze_individual_models()
    
    # 3. 최종 앙상블 결과
    ensemble_df = analyze_final_ensemble()
    
    # 4. 엑셀 파일로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optuna_detailed_results_{timestamp}.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 개별 모델 최고 성능 결과
        individual_df.to_excel(writer, sheet_name='Best_Results', index=False)
        
        # 각 모델별 모든 trial 결과
        for model_type, trials_df in all_trials_data.items():
            if not trials_df.empty:
                sheet_name = f'{model_type}_All_Trials'
                trials_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # 최종 앙상블 결과
        ensemble_df.to_excel(writer, sheet_name='Final_Ensemble', index=False)
        
        # 요약 시트
        summary_data = {
            'Metric': [
                'Total Trials (DCNV2)', 'Total Trials (Focal)', 'Total Trials (RF)',
                'Best Individual Model', 'Best Individual F1', 
                'Final Ensemble F1', 'Improvement'
            ],
            'Value': [
                len(all_trials_data.get('DCNV2', pd.DataFrame())),
                len(all_trials_data.get('CUSTOM_FOCAL_DL', pd.DataFrame())),
                len(all_trials_data.get('RF', pd.DataFrame())),
                individual_df.loc[individual_df['Best_F1_Score'].idxmax(), 'Model'],
                individual_df['Best_F1_Score'].max(),
                ensemble_df['Validation_F1'].iloc[0] if not ensemble_df.empty else 'N/A',
                f"{(ensemble_df['Validation_F1'].iloc[0] - individual_df['Best_F1_Score'].max())*100:.2f}%" if not ensemble_df.empty else 'N/A'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\n✅ 상세 결과가 '{filename}'에 저장되었습니다!")
    
    # 5. 콘솔 출력
    print("\n📊 각 모델별 Trial 수:")
    for model_type, trials_df in all_trials_data.items():
        print(f"  {model_type}: {len(trials_df)}개 trial")
    
    print("\n🏆 최고 성능 모델:")
    best_model = individual_df.loc[individual_df['Best_F1_Score'].idxmax()]
    print(f"  {best_model['Model']}: F1 = {best_model['Best_F1_Score']:.4f}")
    
    return individual_df, ensemble_df, all_trials_data

if __name__ == "__main__":
    individual_df, ensemble_df, all_trials_data = create_detailed_report()
    
    print("\n🎉 상세 분석 완료!")
    print("📋 엑셀 파일에 다음 시트들이 포함됩니다:")
    print("  - Best_Results: 각 모델의 최고 성능 결과")
    print("  - DCNV2_All_Trials: DCNV2의 모든 15개 trial")
    print("  - CUSTOM_FOCAL_DL_All_Trials: Focal Loss의 모든 15개 trial")
    print("  - RF_All_Trials: RandomForest의 모든 15개 trial")
    print("  - Final_Ensemble: 최종 앙상블 결과")
    print("  - Summary: 전체 실험 요약") 