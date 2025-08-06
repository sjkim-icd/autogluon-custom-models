from autogluon.tabular import TabularPredictor

try:
    predictor = TabularPredictor.load('models/dcnv2_fuxictr_split')
    print("=== FuxiCTR 스타일 DCNv2 결과 확인 ===")
    print("Leaderboard:")
    print(predictor.leaderboard())
    
    # Test 데이터로 성능 평가
    import pandas as pd
    from sklearn.metrics import f1_score, accuracy_score
    
    test_df = pd.read_csv("datasets/creditcard.csv")
    test_df["Class"] = test_df["Class"].astype("category")
    
    # 데이터 스플릿 (동일한 방식)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(test_df, test_size=0.2, random_state=42, stratify=test_df['Class'])
    
    try:
        test_predictions = predictor.predict(test_df)
        test_f1 = f1_score(test_df['Class'], test_predictions)
        test_accuracy = accuracy_score(test_df['Class'], test_predictions)
        
        print(f"\nTest 성능:")
        print(f"F1 Score: {test_f1:.4f}")
        print(f"Accuracy: {test_accuracy:.4f}")
    except Exception as e:
        print(f"Test 예측 실패: {e}")
        
except Exception as e:
    print(f"결과 확인 실패: {e}") 