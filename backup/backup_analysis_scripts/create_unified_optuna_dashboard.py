import json
import pandas as pd
import numpy as np
import optuna
import optuna.visualization as vis
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

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

def create_parameter_importance_figures(studies):
    """파라미터 중요도 시각화"""
    figures = {}
    
    for model_type, study in studies.items():
        try:
            fig = vis.plot_param_importances(study)
            fig.update_layout(
                title=f"{model_type} - 파라미터 중요도",
                height=400
            )
            figures[f"{model_type}_importance"] = fig
        except Exception as e:
            print(f"❌ {model_type} 파라미터 중요도 시각화 실패: {e}")
    
    return figures

def create_optimization_history_figures(studies):
    """최적화 과정 시각화"""
    figures = {}
    
    for model_type, study in studies.items():
        try:
            fig = vis.plot_optimization_history(study)
            fig.update_layout(
                title=f"{model_type} - 최적화 과정",
                height=400
            )
            figures[f"{model_type}_history"] = fig
        except Exception as e:
            print(f"❌ {model_type} 최적화 과정 시각화 실패: {e}")
    
    return figures

def create_parallel_coordinate_figures(studies):
    """병렬 좌표 시각화"""
    figures = {}
    
    for model_type, study in studies.items():
        try:
            fig = vis.plot_parallel_coordinate(study)
            fig.update_layout(
                title=f"{model_type} - 병렬 좌표 (모든 trial)",
                height=500
            )
            figures[f"{model_type}_parallel"] = fig
        except Exception as e:
            print(f"❌ {model_type} 병렬 좌표 시각화 실패: {e}")
    
    return figures

def create_slice_figures(studies):
    """슬라이스 플롯 시각화"""
    figures = {}
    
    for model_type, study in studies.items():
        try:
            # 모델별 주요 파라미터들
            if model_type == 'DCNV2':
                params = ['learning_rate', 'weight_decay', 'dropout_prob']
            elif model_type == 'CUSTOM_FOCAL_DL':
                params = ['learning_rate', 'focal_alpha', 'focal_gamma']
            elif model_type == 'RF':
                params = ['n_estimators', 'max_depth']
            
            for param in params:
                try:
                    fig = vis.plot_slice(study, params=[param])
                    fig.update_layout(
                        title=f"{model_type} - {param} 슬라이스",
                        height=400
                    )
                    figures[f"{model_type}_{param}_slice"] = fig
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ {model_type} 슬라이스 플롯 실패: {e}")
    
    return figures

def create_summary_table(studies):
    """요약 테이블 생성"""
    summary_data = []
    
    for model_type, study in studies.items():
        try:
            trials = study.trials
            successful_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if len(successful_trials) >= 2:
                values = [t.value for t in successful_trials]
                best_value = study.best_value
                mean_value = np.mean(values)
                std_value = np.std(values)
                
                summary_data.append({
                    'Model': model_type,
                    'Best_F1': f"{best_value:.4f}",
                    'Mean_F1': f"{mean_value:.4f}",
                    'Std_F1': f"{std_value:.4f}",
                    'Min_F1': f"{min(values):.4f}",
                    'Max_F1': f"{max(values):.4f}",
                    'Trials': f"{len(successful_trials)}/{len(trials)}",
                    'Success_Rate': f"{len(successful_trials)/len(trials)*100:.1f}%"
                })
        except Exception as e:
            print(f"❌ {model_type} 요약 데이터 생성 실패: {e}")
    
    return pd.DataFrame(summary_data)

def create_parameter_importance_table(studies):
    """파라미터 중요도 테이블 생성"""
    importance_data = []
    
    for model_type, study in studies.items():
        try:
            importance = optuna.importance.get_param_importances(study)
            
            for param, score in importance.items():
                importance_data.append({
                    'Model': model_type,
                    'Parameter': param,
                    'Importance': f"{score:.4f}",
                    'Importance_Value': score
                })
        except Exception as e:
            print(f"❌ {model_type} 파라미터 중요도 계산 실패: {e}")
    
    df = pd.DataFrame(importance_data)
    if not df.empty:
        df = df.sort_values(['Model', 'Importance_Value'], ascending=[True, False])
        df = df.drop('Importance_Value', axis=1)
    
    return df

def create_unified_dashboard(studies):
    """통합 대시보드 생성"""
    print("=== Optuna 통합 대시보드 생성 ===")
    
    # 1. 모든 시각화 생성
    print("📊 시각화 생성 중...")
    importance_figs = create_parameter_importance_figures(studies)
    history_figs = create_optimization_history_figures(studies)
    parallel_figs = create_parallel_coordinate_figures(studies)
    slice_figs = create_slice_figures(studies)
    
    # 2. 요약 테이블 생성
    print("📋 요약 테이블 생성 중...")
    summary_df = create_summary_table(studies)
    importance_df = create_parameter_importance_table(studies)
    
    # 3. HTML 대시보드 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optuna_unified_dashboard_{timestamp}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Optuna HPO 통합 대시보드</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            .header p {{
                margin: 10px 0 0 0;
                opacity: 0.9;
                font-size: 1.1em;
            }}
            .section {{
                padding: 30px;
                border-bottom: 1px solid #eee;
            }}
            .section:last-child {{
                border-bottom: none;
            }}
            .section h2 {{
                color: #333;
                margin-bottom: 20px;
                font-size: 1.8em;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 30px;
                margin-top: 20px;
            }}
            .chart-container {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
            }}
            .table-container {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
                overflow-x: auto;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #667eea;
                color: white;
                font-weight: 600;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f0f0f0;
            }}
            .model-section {{
                margin-bottom: 40px;
            }}
            .model-section h3 {{
                color: #667eea;
                font-size: 1.4em;
                margin-bottom: 15px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-card h4 {{
                margin: 0 0 10px 0;
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .stat-card .value {{
                font-size: 1.8em;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Optuna HPO 통합 대시보드</h1>
                <p>DCNV2, Focal Loss, RandomForest 하이퍼파라미터 최적화 결과</p>
                <p>생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section">
                <h2>📊 실험 요약</h2>
                <div class="stats">
    """
    
    # 요약 통계 추가
    if not summary_df.empty:
        total_trials = sum([int(row['Trials'].split('/')[0]) for _, row in summary_df.iterrows()])
        best_performance = max([float(row['Best_F1']) for _, row in summary_df.iterrows()])
        avg_performance = np.mean([float(row['Mean_F1']) for _, row in summary_df.iterrows()])
        
        html_content += f"""
                    <div class="stat-card">
                        <h4>총 Trial 수</h4>
                        <div class="value">{total_trials}</div>
                    </div>
                    <div class="stat-card">
                        <h4>최고 성능</h4>
                        <div class="value">{best_performance:.4f}</div>
                    </div>
                    <div class="stat-card">
                        <h4>평균 성능</h4>
                        <div class="value">{avg_performance:.4f}</div>
                    </div>
        """
    
    html_content += """
                </div>
                
                <div class="table-container">
                    <h3>📋 모델별 성능 요약</h3>
    """
    
    # 요약 테이블 추가
    if not summary_df.empty:
        html_content += summary_df.to_html(index=False, classes='dataframe', border=0)
    
    html_content += """
                </div>
                
                <div class="table-container">
                    <h3>🔍 파라미터 중요도</h3>
    """
    
    # 파라미터 중요도 테이블 추가
    if not importance_df.empty:
        html_content += importance_df.to_html(index=False, classes='dataframe', border=0)
    
    html_content += """
                </div>
            </div>
    """
    
    # 각 모델별 섹션 추가
    for model_type in studies.keys():
        html_content += f"""
            <div class="section">
                <h2>🎯 {model_type} 분석</h2>
        """
        
        # 파라미터 중요도
        if f"{model_type}_importance" in importance_figs:
            html_content += f"""
                <div class="chart-container">
                    <h3>📊 파라미터 중요도</h3>
                    {importance_figs[f"{model_type}_importance"].to_html(full_html=False, include_plotlyjs=False)}
                </div>
            """
        
        # 최적화 과정
        if f"{model_type}_history" in history_figs:
            html_content += f"""
                <div class="chart-container">
                    <h3>📈 최적화 과정</h3>
                    {history_figs[f"{model_type}_history"].to_html(full_html=False, include_plotlyjs=False)}
                </div>
            """
        
        # 병렬 좌표
        if f"{model_type}_parallel" in parallel_figs:
            html_content += f"""
                <div class="chart-container">
                    <h3>🔄 병렬 좌표 (모든 trial)</h3>
                    {parallel_figs[f"{model_type}_parallel"].to_html(full_html=False, include_plotlyjs=False)}
                </div>
            """
        
        # 슬라이스 플롯들
        slice_charts = [k for k in slice_figs.keys() if k.startswith(f"{model_type}_")]
        if slice_charts:
            html_content += f"""
                <div class="grid">
            """
            for chart_key in slice_charts:
                param_name = chart_key.replace(f"{model_type}_", "").replace("_slice", "")
                html_content += f"""
                    <div class="chart-container">
                        <h3>📊 {param_name} 슬라이스</h3>
                        {slice_figs[chart_key].to_html(full_html=False, include_plotlyjs=False)}
                    </div>
                """
            html_content += """
                </div>
            """
        
        html_content += """
            </div>
        """
    
    html_content += """
        </div>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 통합 대시보드가 '{filename}'에 저장되었습니다!")
    return filename

if __name__ == "__main__":
    # 모든 study 로드
    studies = load_studies()
    
    if not studies:
        print("❌ 로드할 study가 없습니다!")
    else:
        # 통합 대시보드 생성
        dashboard_file = create_unified_dashboard(studies)
        
        print("\n🎉 Optuna 통합 대시보드 생성 완료!")
        print("📋 대시보드 내용:")
        print("  - 실험 요약 및 통계")
        print("  - 모델별 성능 요약 테이블")
        print("  - 파라미터 중요도 테이블")
        print("  - 각 모델별 상세 분석:")
        print("    • 파라미터 중요도 차트")
        print("    • 최적화 과정 차트")
        print("    • 병렬 좌표 차트")
        print("    • 주요 파라미터 슬라이스 차트")
        print(f"\n📂 파일 위치: {dashboard_file}")
        print("🌐 브라우저에서 파일을 열어서 인터랙티브 대시보드를 확인하세요!") 