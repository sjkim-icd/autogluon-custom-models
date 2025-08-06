import json
import pandas as pd
import numpy as np
import optuna
import optuna.visualization as vis
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_studies():
    """모든 Optuna study 로드"""
    model_types = ['DCNV2', 'CUSTOM_FOCAL_DL', 'RF']
    studies = {}
    
    for model_type in model_types:
        try:
            study = optuna.load_study(
                study_name=f'{model_type}_hpo_study',
                storage=f'sqlite:///optuna_studies/{model_type}_study.db'
            )
            studies[model_type] = study
            print(f"✅ {model_type} study 로드 완료")
        except Exception as e:
            print(f"❌ {model_type} study 로드 실패: {e}")
    
    return studies

def analyze_parameter_importance(studies):
    """파라미터 중요도 분석"""
    print("\n=== 파라미터 중요도 분석 ===")
    
    importance_results = {}
    
    for model_type, study in studies.items():
        try:
            # 파라미터 중요도 계산
            importance = optuna.importance.get_param_importances(study)
            
            print(f"\n🔍 {model_type} 파라미터 중요도:")
            for param, score in importance.items():
                print(f"  {param}: {score:.4f}")
            
            importance_results[model_type] = importance
            
        except Exception as e:
            print(f"❌ {model_type} 파라미터 중요도 계산 실패: {e}")
    
    return importance_results

def analyze_optimization_history(studies):
    """최적화 과정 분석"""
    print("\n=== 최적화 과정 분석 ===")
    
    history_results = {}
    
    for model_type, study in studies.items():
        try:
            trials = study.trials
            successful_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if len(successful_trials) < 2:
                print(f"⚠️ {model_type}: 성공한 trial이 부족하여 분석 불가")
                continue
            
            # 성능 통계
            values = [t.value for t in successful_trials]
            best_value = study.best_value
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            print(f"\n📊 {model_type} 최적화 통계:")
            print(f"  최고 성능: {best_value:.4f}")
            print(f"  평균 성능: {mean_value:.4f}")
            print(f"  표준편차: {std_value:.4f}")
            print(f"  성능 범위: {min(values):.4f} ~ {max(values):.4f}")
            print(f"  성공률: {len(successful_trials)}/{len(trials)} ({len(successful_trials)/len(trials)*100:.1f}%)")
            
            # 수렴성 분석
            convergence_analysis = analyze_convergence(successful_trials)
            print(f"  수렴성: {convergence_analysis}")
            
            history_results[model_type] = {
                'best_value': best_value,
                'mean_value': mean_value,
                'std_value': std_value,
                'convergence': convergence_analysis
            }
            
        except Exception as e:
            print(f"❌ {model_type} 최적화 과정 분석 실패: {e}")
    
    return history_results

def analyze_convergence(trials):
    """수렴성 분석"""
    if len(trials) < 5:
        return "데이터 부족"
    
    # 마지막 5개 trial의 성능 변화 확인
    recent_values = [t.value for t in trials[-5:]]
    improvement = recent_values[-1] - recent_values[0]
    
    if improvement > 0.01:
        return "개선 중"
    elif abs(improvement) < 0.005:
        return "수렴됨"
    else:
        return "불안정"

def analyze_parameter_correlations(studies):
    """파라미터 상관관계 분석"""
    print("\n=== 파라미터 상관관계 분석 ===")
    
    for model_type, study in studies.items():
        try:
            successful_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if len(successful_trials) < 3:
                continue
            
            print(f"\n🔍 {model_type} 파라미터 상관관계:")
            
            # 주요 파라미터들의 성능에 미치는 영향 분석
            param_effects = {}
            
            for trial in successful_trials:
                for param_name, param_value in trial.params.items():
                    if param_name not in param_effects:
                        param_effects[param_name] = []
                    param_effects[param_name].append((param_value, trial.value))
            
            # 각 파라미터의 성능 영향 분석
            for param_name, values in param_effects.items():
                if len(values) < 3:
                    continue
                
                # 파라미터 값과 성능의 상관관계 계산
                param_values = [v[0] for v in values]
                performances = [v[1] for v in values]
                
                # 간단한 상관관계 분석
                if isinstance(param_values[0], (int, float)):
                    correlation = np.corrcoef(param_values, performances)[0, 1]
                    print(f"  {param_name}: 상관계수 = {correlation:.3f}")
                    
                    # 최고 성능 구간의 파라미터 범위
                    top_performances = sorted(values, key=lambda x: x[1], reverse=True)[:3]
                    top_values = [v[0] for v in top_performances]
                    print(f"    최고 성능 구간: {min(top_values)} ~ {max(top_values)}")
            
        except Exception as e:
            print(f"❌ {model_type} 상관관계 분석 실패: {e}")

def create_advanced_visualizations(studies):
    """고급 시각화 생성"""
    print("\n=== 고급 시각화 생성 ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_type, study in studies.items():
        try:
            print(f"\n📊 {model_type} 시각화 생성 중...")
            
            # 1. 파라미터 중요도
            try:
                fig_importance = vis.plot_param_importances(study)
                fig_importance.write_html(f"optuna_{model_type}_importance_{timestamp}.html")
                print(f"  ✅ 파라미터 중요도: optuna_{model_type}_importance_{timestamp}.html")
            except Exception as e:
                print(f"  ❌ 파라미터 중요도 시각화 실패: {e}")
            
            # 2. 최적화 과정
            try:
                fig_history = vis.plot_optimization_history(study)
                fig_history.write_html(f"optuna_{model_type}_history_{timestamp}.html")
                print(f"  ✅ 최적화 과정: optuna_{model_type}_history_{timestamp}.html")
            except Exception as e:
                print(f"  ❌ 최적화 과정 시각화 실패: {e}")
            
            # 3. 병렬 좌표 플롯
            try:
                fig_parallel = vis.plot_parallel_coordinate(study)
                fig_parallel.write_html(f"optuna_{model_type}_parallel_{timestamp}.html")
                print(f"  ✅ 병렬 좌표: optuna_{model_type}_parallel_{timestamp}.html")
            except Exception as e:
                print(f"  ❌ 병렬 좌표 시각화 실패: {e}")
            
            # 4. 슬라이스 플롯 (주요 파라미터들)
            try:
                # 주요 파라미터들에 대해 슬라이스 플롯 생성
                if model_type == 'DCNV2':
                    params = ['learning_rate', 'weight_decay', 'dropout_prob']
                elif model_type == 'CUSTOM_FOCAL_DL':
                    params = ['learning_rate', 'focal_alpha', 'focal_gamma']
                elif model_type == 'RF':
                    params = ['n_estimators', 'max_depth']
                
                for param in params:
                    try:
                        fig_slice = vis.plot_slice(study, params=[param])
                        fig_slice.write_html(f"optuna_{model_type}_{param}_slice_{timestamp}.html")
                        print(f"  ✅ {param} 슬라이스: optuna_{model_type}_{param}_slice_{timestamp}.html")
                    except:
                        pass
                        
            except Exception as e:
                print(f"  ❌ 슬라이스 플롯 실패: {e}")
            
        except Exception as e:
            print(f"❌ {model_type} 시각화 생성 실패: {e}")

def generate_recommendations(studies, importance_results, history_results):
    """다음 실험을 위한 권장사항 생성"""
    print("\n=== 다음 실험 권장사항 ===")
    
    for model_type in studies.keys():
        print(f"\n🎯 {model_type} 권장사항:")
        
        # 파라미터 중요도 기반 권장사항
        if model_type in importance_results:
            importance = importance_results[model_type]
            top_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"  📊 가장 중요한 파라미터:")
            for param, score in top_params:
                print(f"    - {param}: {score:.4f}")
        
        # 최적화 과정 기반 권장사항
        if model_type in history_results:
            history = history_results[model_type]
            print(f"  📈 최적화 상태:")
            print(f"    - 최고 성능: {history['best_value']:.4f}")
            print(f"    - 성능 변동성: {history['std_value']:.4f}")
            print(f"    - 수렴 상태: {history['convergence']}")
            
            # 권장사항
            if history['std_value'] > 0.05:
                print(f"    💡 권장: 더 많은 trial 필요 (높은 변동성)")
            elif history['convergence'] == "수렴됨":
                print(f"    💡 권장: 현재 설정으로 충분히 최적화됨")
            else:
                print(f"    💡 권장: 더 세밀한 파라미터 탐색 필요")

def create_comprehensive_report(studies):
    """종합 리포트 생성"""
    print("=== Optuna 고급 분석 리포트 ===")
    
    # 1. 파라미터 중요도 분석
    importance_results = analyze_parameter_importance(studies)
    
    # 2. 최적화 과정 분석
    history_results = analyze_optimization_history(studies)
    
    # 3. 파라미터 상관관계 분석
    analyze_parameter_correlations(studies)
    
    # 4. 시각화 생성
    create_advanced_visualizations(studies)
    
    # 5. 권장사항 생성
    generate_recommendations(studies, importance_results, history_results)
    
    # 6. 요약 리포트 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optuna_advanced_analysis_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== Optuna 고급 분석 리포트 ===\n\n")
        
        f.write("1. 파라미터 중요도 분석\n")
        f.write("=" * 50 + "\n")
        for model_type, importance in importance_results.items():
            f.write(f"\n{model_type}:\n")
            for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {param}: {score:.4f}\n")
        
        f.write("\n\n2. 최적화 과정 분석\n")
        f.write("=" * 50 + "\n")
        for model_type, history in history_results.items():
            f.write(f"\n{model_type}:\n")
            f.write(f"  최고 성능: {history['best_value']:.4f}\n")
            f.write(f"  평균 성능: {history['mean_value']:.4f}\n")
            f.write(f"  표준편차: {history['std_value']:.4f}\n")
            f.write(f"  수렴 상태: {history['convergence']}\n")
    
    print(f"\n✅ 종합 리포트가 '{filename}'에 저장되었습니다!")
    print("📊 생성된 HTML 시각화 파일들:")
    print("  - 파라미터 중요도: optuna_*_importance_*.html")
    print("  - 최적화 과정: optuna_*_history_*.html")
    print("  - 병렬 좌표: optuna_*_parallel_*.html")
    print("  - 슬라이스 플롯: optuna_*_*_slice_*.html")

if __name__ == "__main__":
    # 모든 study 로드
    studies = load_studies()
    
    if not studies:
        print("❌ 로드할 study가 없습니다!")
    else:
        # 종합 분석 실행
        create_comprehensive_report(studies)
        
        print("\n🎉 Optuna 고급 분석 완료!")
        print("📋 주요 분석 결과:")
        print("  - 파라미터 중요도: 어떤 파라미터가 성능에 가장 큰 영향")
        print("  - 최적화 과정: 수렴성과 안정성 분석")
        print("  - 파라미터 상관관계: 파라미터 간 상호작용")
        print("  - 시각화: 인터랙티브 HTML 차트")
        print("  - 권장사항: 다음 실험을 위한 가이드") 