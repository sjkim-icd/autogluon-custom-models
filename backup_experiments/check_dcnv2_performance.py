from autogluon.tabular import TabularPredictor
import pandas as pd

# 두 결과 폴더 확인
folders = ['ag-20250730_082846', 'ag-20250730_083658']

for folder in folders:
    try:
        predictor = TabularPredictor.load(f'AutogluonModels/{folder}')
        lb = predictor.leaderboard()
        
        print(f'\n=== {folder} 결과 ===')
        print(f'전체 모델 수: {len(lb)}')
        
        # DCNV2 모델들만 필터링
        dcnv2_models = lb[lb['model'].str.contains('DCNV2')]
        print(f'DCNV2 모델 수: {len(dcnv2_models)}')
        
        if len(dcnv2_models) > 0:
            print('\nDCNV2 모델 성능 (내림차순):')
            dcnv2_sorted = dcnv2_models.sort_values('score_val', ascending=False)
            for idx, row in dcnv2_sorted.iterrows():
                print(f"  {row['model']}: {row['score_val']:.4f} (F1)")
        
        # 전체 모델 중 최고 성능
        best_model = lb.loc[lb['score_val'].idxmax()]
        print(f'\n전체 최고 성능: {best_model["model"]} - {best_model["score_val"]:.4f}')
        
    except Exception as e:
        print(f'{folder} 로드 실패: {e}') 