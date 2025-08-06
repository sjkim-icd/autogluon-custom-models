import json
import re

def extract_hyperparams_from_json():
    """JSON 파일에서 0.8235 성능을 달성한 모델의 하이퍼파라미터를 추출"""
    
    # JSON 파일 읽기
    with open('AutogluonModels/ag-20250730_083658/models/DCNV2/experiment_state-2025-07-30_17-37-20.json', 'r') as f:
        data = json.load(f)
    
    # trial_data에서 각 trial 정보 추출
    for trial_info in data['trial_data']:
        if len(trial_info) >= 2:
            trial_json = json.loads(trial_info[0])
            results_json = json.loads(trial_info[1])
            
            trial_id = trial_json.get('trial_id', '')
            validation_performance = results_json.get('last_result', {}).get('validation_performance', 0)
            
            # 0.8235 성능을 달성한 모델 찾기
            if abs(validation_performance - 0.8235) < 0.001:
                print(f"\n=== 0.8235 성능을 달성한 모델: {trial_id} ===")
                
                config = results_json.get('last_result', {}).get('config', {})
                
                # DCNv2 관련 하이퍼파라미터만 추출
                dcnv2_params = {
                    'num_cross_layers': config.get('num_cross_layers'),
                    'cross_dropout': config.get('cross_dropout'),
                    'low_rank': config.get('low_rank'),
                    'deep_output_size': config.get('deep_output_size'),
                    'deep_hidden_size': config.get('deep_hidden_size'),
                    'deep_dropout': config.get('deep_dropout'),
                    'deep_layers': config.get('deep_layers'),
                    'learning_rate': config.get('learning_rate'),
                    'weight_decay': config.get('weight_decay'),
                    'dropout_prob': config.get('dropout_prob'),
                    'activation': config.get('activation'),
                    'optimizer': config.get('optimizer'),
                    'lr_scheduler': config.get('lr_scheduler'),
                    'scheduler_type': config.get('scheduler_type'),
                    'num_epochs': config.get('num_epochs'),
                    'hidden_size': config.get('hidden_size'),
                    'use_batchnorm': config.get('use_batchnorm')
                }
                
                print("DCNv2 하이퍼파라미터:")
                for key, value in dcnv2_params.items():
                    print(f"  {key}: {value}")
                
                print(f"\n검증 성능: {validation_performance}")
                print(f"실험 태그: {results_json.get('last_result', {}).get('experiment_tag', '')}")
                
                return dcnv2_params
    
    print("0.8235 성능을 달성한 모델을 찾을 수 없습니다.")
    return None

if __name__ == "__main__":
    extract_hyperparams_from_json() 