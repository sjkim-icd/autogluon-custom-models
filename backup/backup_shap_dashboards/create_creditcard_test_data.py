import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_creditcard_test_data():
    """Credit Card 데이터를 train/test로 분할하고 test 데이터를 별도 파일로 저장"""
    
    print("=== Credit Card 데이터 분할 및 Test 데이터 저장 ===")
    
    # 데이터 로드
    df = pd.read_csv("datasets/creditcard.csv")
    print(f"📊 전체 데이터: {df.shape}")
    print(f"🎯 타겟 분포:")
    print(df['Class'].value_counts())
    
    # 데이터 분할 (stratified)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    train_data, test_data = train_test_split(
        df, test_size=0.2, random_state=42, 
        stratify=df['Class']
    )
    
    print(f"✅ Train 데이터: {train_data.shape}")
    print(f"✅ Test 데이터: {test_data.shape}")
    print(f"📊 Train 클래스 분포:")
    print(train_data['Class'].value_counts())
    print(f"📊 Test 클래스 분포:")
    print(test_data['Class'].value_counts())
    
    # Test 데이터 저장
    test_data_path = "datasets/creditcard_test.csv"
    test_data.to_csv(test_data_path, index=False)
    print(f"✅ Test 데이터 저장 완료: {test_data_path}")
    
    return test_data_path

if __name__ == "__main__":
    create_creditcard_test_data() 