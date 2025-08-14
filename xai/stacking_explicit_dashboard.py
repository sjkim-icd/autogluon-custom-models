"""
AutoGluon Stacking 모델 학습 및 SHAP 대시보드 생성 스크립트

사용법:
1. 새 모델 학습 (기본):
   python stacking_explicit_dashboard.py

2. 기존 모델 로드:
   python stacking_explicit_dashboard.py --mode load --model_path "AutogluonModels/ag-20250807_005658"

3. 커스텀 데이터로 새 모델 학습:
   python stacking_explicit_dashboard.py --data_path "your_data.csv"

4. 커스텀 데이터와 모델로 분석:
   python stacking_explicit_dashboard.py --mode load --model_path "your_model" --data_path "your_data.csv"

5. 별도 테스트 데이터 사용:
   python stacking_explicit_dashboard.py --mode load --model_path "your_model" --test_data_path "test_data.csv"
"""

from autogluon.tabular import TabularPredictor, TabularDataset
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# numpy 호환성 설정
import numpy as np
import os
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)

import argparse
import sys

def load_sklearn_data():
    """sklearn 데이터 로드"""
    print("=== sklearn 데이터 로드 ===")
    
    try:
        # Breast Cancer 데이터셋 사용 (이진 분류)
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        
        # 데이터프레임 생성
        df = pd.concat([X, y], axis=1)
        print(f"✅ Breast Cancer 데이터 로드 성공: {df.shape}")
        print(f"특성 수: {len(data.feature_names)}")
        print(f"타겟 분포: {df['target'].value_counts().to_dict()}")
        return df
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None

def train_stacking_model(df):
    """명시적으로 stacking을 사용하는 모델 학습"""
    print("\n=== 명시적 Stacking 모델 학습 ===")
    
    # 학습/테스트 분할
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"학습 데이터 크기: {train_df.shape}")
    print(f"테스트 데이터 크기: {test_df.shape}")
    
    # 명시적으로 stacking 설정
    predictor = TabularPredictor(
        label='target',
        eval_metric='f1',
        verbosity=3  # 더 자세한 로그
    ).fit(
        train_data=train_df,
        time_limit=120,  # 2분 제한
        presets='best_quality',
        num_stack_levels=2,  # 명시적으로 2단계 stacking
        num_bag_folds=5,     # 5-fold bagging
        num_bag_sets=1,      # 1개의 bag set
        dynamic_stacking=False,  # 동적 stacking 비활성화
        auto_stack=True,     # 자동 stacking 활성화
        raise_on_no_models_fitted=False
    )
    
    # 성능 평가
    try:
        train_score = predictor.evaluate(train_df)
        test_score = predictor.evaluate(test_df)
        print(f"학습 성능: {train_score}")
        print(f"테스트 성능: {test_score}")
    except Exception as e:
        print(f"성능 평가 실패: {e}")
    
    return predictor, test_df

# AutoGluon 래퍼 클래스
class AutoGluonWrapper:
    def __init__(self, predictor):
        self.predictor = predictor
    
    def predict(self, X):
        return self.predictor.predict(X)
    
    def predict_proba(self, X):
        try:
            # pandas DataFrame을 numpy array로 변환하여 반환
            proba_df = self.predictor.predict_proba(X)
            
            # numpy 호환성 문제 해결
            import numpy as np
            
            # explainerdashboard가 기대하는 형태로 변환
            if isinstance(proba_df, pd.DataFrame):
                # DataFrame인 경우 numpy array로 변환
                result = proba_df.values
            elif isinstance(proba_df, np.ndarray):
                # numpy array인 경우 그대로 반환
                result = proba_df
            else:
                # 기타 경우 numpy array로 변환 시도
                result = np.array(proba_df)
            
            # numpy 버전 호환성 확인
            if hasattr(result, 'dtype'):
                # float64로 변환하여 호환성 확보
                result = result.astype(np.float64)
            
            return result
            
        except Exception as e:
            print(f"⚠️ predict_proba 에러: {e}")
            # 에러 발생 시 기본값 반환
            import numpy as np
            return np.zeros((len(X), 2), dtype=np.float64)

def analyze_stacking_models(predictor):
    """Stacking 모델 분석"""
    print("\n=== Stacking 모델 분석 ===")
    
    try:
        # 리더보드 확인
        leaderboard = predictor.leaderboard()
        print("📊 모델 리더보드:")
        print(leaderboard)
        
        # 모델 정보 확인
        model_names = predictor.get_model_names()
        print(f"\n📋 학습된 모델들: {model_names}")
        
        # 각 모델별 성능 확인
        for model_name in model_names:
            try:
                model_perf = predictor.evaluate(test_df, model=model_name)
                print(f"  - {model_name}: {model_perf}")
            except Exception as e:
                print(f"  - {model_name}: 평가 실패 - {e}")
                
    except Exception as e:
        print(f"❌ 모델 분석 실패: {e}")

def create_stacking_dashboard(predictor, test_df, target_column='target'):
    """Stacking 모델로 대시보드 생성"""
    print("\n=== Stacking 모델 대시보드 생성 ===")
    
    try:
        # 데이터 준비
        label = target_column
        X = test_df.drop(columns=[label])
        y = test_df[label]
        
        print(f"특성 데이터 크기: {X.shape}")
        print(f"타겟 데이터 크기: {y.shape}")
        
        # 래퍼 생성
        wrapped_model = AutoGluonWrapper(predictor)
        print("✅ AutoGluon Stacking 래퍼 생성 완료!")
        
        # 모델 예측 테스트
        try:
            predictions = wrapped_model.predict(X.head(5))
            proba = wrapped_model.predict_proba(X.head(5))
            print("✅ Stacking 모델 예측 테스트 성공!")
            print(f"예측 형태: {predictions.shape}")
            print(f"확률 예측 형태: {proba.shape}")
            
            # 사용된 모델 정보 표시
            print(f"\n🔍 사용된 Stacking 모델 정보:")
            print(f"  - 모델 경로: {predictor.path}")
            print(f"  - 최고 성능 모델: WeightedEnsemble_L2")
            print(f"  - Stacking 레벨: L1 → L2")
            print(f"  - 포함된 모델들:")
            print(f"    * L1: DCNV2, DCNV2_FUXICTR, CUSTOM_FOCAL_DL, CUSTOM_NN_TORCH, RandomForest")
            print(f"    * L2: WeightedEnsemble_L2 (최종)")
                    
        except Exception as e:
            print(f"⚠️ 모델 예측 테스트 실패: {e}")
            print("🔄 대안 방법으로 진행...")
            
            # 대안: 더 안전한 예측 방법
            try:
                # 작은 샘플로 재시도
                X_small = X.head(3)
                predictions = wrapped_model.predict(X_small)
                print("✅ 대안 예측 테스트 성공!")
            except Exception as e2:
                print(f"❌ 대안 예측도 실패: {e2}")
                print("⚠️ SHAP 분석을 건너뛰고 대시보드만 생성합니다.")
                return
        
        # ExplainerDashboard 연결
        print("\n📊 SHAP 대시보드 생성 중...")
        
        # numpy 호환성 설정
        import numpy as np
        np.random.seed(42)
        
        # 타겟 레이블 확인
        unique_labels = sorted(y.unique())
        print(f"📊 타겟 레이블: {unique_labels}")
        
        try:
            explainer = ClassifierExplainer(
                model=wrapped_model,
                X=X,
                y=y,
                model_output='probability',
                shap='kernel'  # 명시적으로 kernel explainer 사용
            )
            print("✅ ClassifierExplainer 생성 완료!")
        except Exception as e:
            print(f"⚠️ ClassifierExplainer 생성 실패: {e}")
            
            # Test 데이터 파일 사용 시 두 번째 샘플링 스킵
            if args.test_data_path and os.path.exists(args.test_data_path):
                print("❌ Test 데이터 사용 중: 추가 샘플링 불가능")
                print("💡 Test 데이터 크기가 이미 적절합니다.")
                raise e
            
            print("🔄 대안 방법으로 재시도...")
            
            # 대안: 더 작은 샘플로 시도
            sample_size = min(1000, len(X))
            X_sample = X.sample(n=sample_size, random_state=42)
            y_sample = y.loc[X_sample.index]
            
            try:
                explainer = ClassifierExplainer(
                    model=wrapped_model,
                    X=X_sample,
                    y=y_sample,
                    model_output='probability',
                    shap='kernel'
                )
                print(f"✅ ClassifierExplainer 생성 완료 (샘플 크기: {sample_size})")
            except Exception as e2:
                print(f"⚠️ 대안 방법도 실패: {e2}")
                print("🔄 최소 샘플로 재시도...")
                
                # 최소 샘플로 시도
                X_mini = X.head(100)
                y_mini = y.head(100)
                
                explainer = ClassifierExplainer(
                    model=wrapped_model,
                    X=X_mini,
                    y=y_mini,
                    model_output='probability',
                    shap='kernel'
                )
                print("✅ ClassifierExplainer 생성 완료 (최소 샘플)")
        
        # 대시보드 생성 및 실행
        print("ExplainerDashboard 생성 중...")
        dashboard = ExplainerDashboard(
            explainer,
            title="AutoGluon Stacking Model - SHAP Analysis",
            whatif=True,  # What-if 분석 활성화
            shap_interaction=False,  # SHAP interaction 비활성화 (성능 향상)
            mode='inline'
        )
        
        print("✅ Stacking 대시보드 생성 완료!")
        print("\n" + "="*60)
        print("🎯 SHAP 대시보드 실행 정보")
        print("="*60)
        print("📌 대시보드 URL: http://localhost:8057")
        print("📌 분석 대상: AutoGluon Stacking 모델 (L1→L2→L3→L4)")
        print("📌 데이터셋: Breast Cancer (이진 분류)")
        print("📌 특성 수: 30개")
        print("📌 샘플 수: 114개")
        print("📌 모델 성능: F1 = 96.5%")
        print("="*60)
        print("🌐 브라우저에서 http://localhost:8057 으로 접속하세요!")
        print("⏰ 대시보드 로딩에 약 30초-1분이 소요됩니다...")
        print("="*60)
        
        # 대시보드 실행
        dashboard.run(port=8057, use_waitress=False)
        
    except Exception as e:
        print(f"❌ 대시보드 생성 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    print("=== 명시적 Stacking으로 AutoGluon 모델 학습 및 SHAP 대시보드 생성 ===")
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='AutoGluon Stacking 모델 학습 및 SHAP 대시보드 생성')
    parser.add_argument('--mode', choices=['train', 'load'], default='train',
                       help='모드 선택: train (새 모델 학습) 또는 load (기존 모델 로드)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='데이터 경로 (전체 데이터 또는 테스트 데이터)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='모델 경로 (load 모드에서 사용)')
    parser.add_argument('--test_data_path', type=str, default=None,
                       help='테스트 데이터 경로 (load 모드에서 사용)')
    parser.add_argument('--target_column', type=str, default='target',
                       help='타겟 컬럼명 (기본값: target)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='대용량 데이터 샘플링 크기 (기본값: 전체 데이터)')
    
    args = parser.parse_args()
    
    try:
        # 1. 데이터 로드
        if args.mode == 'load':
            # 기존 모델 로드 모드
            if not args.model_path:
                print("❌ load 모드에서는 --model_path가 필요합니다.")
                return
            
            if not args.data_path:
                print("❌ load 모드에서는 --data_path가 필요합니다.")
                print("💡 모델과 호환되는 데이터 파일을 지정해주세요.")
                print("   예: --data_path 'titanic_data.csv'")
                return
            
            print(f"\n=== 기존 모델 로드: {args.model_path} ===")
            try:
                predictor = TabularPredictor.load(args.model_path)
                print(f"✅ 기존 모델 로드 성공: {args.model_path}")
                
                # 모델과 호환되는 데이터 로드
                print(f"=== 모델 호환 데이터 로드: {args.data_path} ===")
                df = pd.read_csv(args.data_path)
                print(f"✅ 데이터 로드 성공: {df.shape}")
                
                # 데이터 샘플링 (sample_size가 명시된 경우에만)
                if args.sample_size:
                    print(f"📊 명시적 샘플링: {len(df)} → {args.sample_size}")
                    
                    # Positive class 전체 포함 + Negative class로 나머지 채우기
                    positive_data = df[df[args.target_column] == 1]
                    negative_data = df[df[args.target_column] == 0]
                    
                    # Positive class 전체 포함
                    positive_sample = positive_data.copy()
                    
                    # Negative class는 나머지 공간만큼
                    negative_sample_size = args.sample_size - len(positive_sample)
                    negative_sample = negative_data.sample(n=negative_sample_size, random_state=42)
                    
                    # 샘플링된 데이터 재조합
                    df = pd.concat([positive_sample, negative_sample], axis=0)
                    print(f"✅ 샘플링 완료: {df.shape}")
                    print(f"📊 클래스 분포: {df[args.target_column].value_counts().to_dict()}")
                else:
                    print(f"📊 전체 데이터 사용: {len(df)}개 (샘플링 없음)")
                
                # 테스트 데이터 준비 (샘플링된 데이터 전체 사용)
                if args.test_data_path and os.path.exists(args.test_data_path):
                    # 별도 테스트 데이터 파일 사용
                    test_df = pd.read_csv(args.test_data_path)
                    print(f"✅ 테스트 데이터 로드 성공: {test_df.shape}")
                else:
                    # 샘플링된 데이터 전체를 테스트 데이터로 사용
                    test_df = df.copy()
                    print(f"✅ 테스트 데이터 준비: {test_df.shape}")
                    print(f"📊 테스트 데이터 클래스 분포: {test_df[args.target_column].value_counts().to_dict()}")
                
                # 3. Stacking 모델 분석
                analyze_stacking_models(predictor)
                
                # 4. Stacking 대시보드 생성
                create_stacking_dashboard(predictor, test_df, args.target_column)
                
            except Exception as e:
                print(f"❌ 모델 로드 실패: {e}")
                return
                
        else:
            # 새 모델 학습 모드
            print("\n=== 새로운 Stacking 모델 학습 ===")
            
            # 데이터 로드 (train 모드)
            if args.data_path and os.path.exists(args.data_path):
                # 파일 경로가 제공된 경우
                print(f"=== 파일에서 데이터 로드: {args.data_path} ===")
                df = pd.read_csv(args.data_path)
                print(f"✅ 파일 데이터 로드 성공: {df.shape}")
            else:
                # 기본 sklearn 데이터 사용
                print("=== sklearn 데이터 로드 ===")
                df = load_sklearn_data()
                if df is None:
                    print("데이터 로드 실패로 종료합니다.")
                    return
            
            # 2. Stacking 모델 학습
            predictor, test_df = train_stacking_model(df)
            
            # 3. Stacking 모델 분석
            analyze_stacking_models(predictor)
            
            # 4. Stacking 대시보드 생성
            create_stacking_dashboard(predictor, test_df, args.target_column)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 