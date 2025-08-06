import json
import pandas as pd
import numpy as np
import optuna
from datetime import datetime

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

def create_simple_chart_data(study, model_type):
    """간단한 차트 데이터 생성"""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(trials) < 2:
        return None, None, None
    
    # 최적화 과정 데이터
    trial_numbers = [t.number for t in trials]
    values = [t.value for t in trials]
    
    # 파라미터 중요도 데이터
    try:
        importance = optuna.importance.get_param_importances(study)
        if importance:
            param_names = list(importance.keys())
            importance_values = list(importance.values())
        else:
            param_names = []
            importance_values = []
    except:
        param_names = []
        importance_values = []
    
    # 파라미터 상관관계 데이터
    correlation_data = []
    if len(trials) >= 3:
        data = []
        for trial in trials:
            row = {'value': trial.value}
            row.update(trial.params)
            data.append(row)
        
        df = pd.DataFrame(data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'value' in numeric_cols:
            numeric_cols.remove('value')
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    corr = df[col1].corr(df[col2])
                    correlation_data.append({
                        'param1': col1,
                        'param2': col2,
                        'correlation': corr
                    })
        
        correlation_data.sort(key=lambda x: abs(x['correlation']), reverse=True)
        correlation_data = correlation_data[:5]
    
    return {
        'trial_numbers': trial_numbers,
        'values': values,
        'param_names': param_names,
        'importance_values': importance_values,
        'correlation_data': correlation_data
    }

def create_html_dashboard(studies):
    """간단한 HTML 대시보드 생성"""
    print("=== 간단한 HTML 대시보드 생성 ===")
    
    # HTML 시작
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎯 Optuna HPO 분석 대시보드</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }
            .header p {
                margin: 10px 0 0 0;
                opacity: 0.9;
                font-size: 1.1em;
            }
            .content {
                padding: 30px;
            }
            .section {
                margin-bottom: 40px;
                background-color: #fafafa;
                border-radius: 8px;
                padding: 25px;
                border-left: 4px solid #667eea;
            }
            .section h2 {
                color: #333;
                margin-top: 0;
                font-size: 1.8em;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            .model-section {
                background-color: white;
                border-radius: 8px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .model-section h3 {
                color: #667eea;
                margin-top: 0;
                font-size: 1.5em;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            .chart-container {
                margin: 20px 0;
                text-align: center;
                border: 1px solid #eee;
                border-radius: 8px;
                padding: 20px;
                background-color: #fafafa;
                min-height: 400px;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .stat-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-card h4 {
                margin: 0 0 10px 0;
                font-size: 1.1em;
                opacity: 0.9;
            }
            .stat-card .value {
                font-size: 2em;
                font-weight: bold;
                margin: 0;
            }
            .summary-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            .summary-table th, .summary-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .summary-table th {
                background-color: #667eea;
                color: white;
                font-weight: bold;
            }
            .summary-table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .summary-table tr:hover {
                background-color: #f0f0f0;
            }
            .best-performance {
                background-color: #ffd700 !important;
                font-weight: bold;
            }
            .footer {
                background-color: #333;
                color: white;
                text-align: center;
                padding: 20px;
                margin-top: 40px;
            }
            .no-data {
                text-align: center;
                padding: 40px;
                color: #666;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Optuna HPO 분석 대시보드</h1>
                <p>Hyperparameter Optimization 결과 및 시각화</p>
                <p>생성 시간: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
            
            <div class="content">
    """
    
    # 1. 전체 요약 섹션
    html_content += """
                <div class="section">
                    <h2>📊 전체 실험 요약</h2>
    """
    
    # 통계 계산
    total_trials = 0
    best_performances = {}
    for model_type, study in studies.items():
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        total_trials += len(trials)
        if trials:
            best_performances[model_type] = study.best_value
    
    if best_performances:
        best_overall = max(best_performances.values())
        best_model = max(best_performances, key=best_performances.get)
        
        html_content += f"""
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h4>총 Trial 수</h4>
                            <div class="value">{total_trials}</div>
                        </div>
                        <div class="stat-card">
                            <h4>최고 성능</h4>
                            <div class="value">{best_overall:.4f}</div>
                        </div>
                        <div class="stat-card">
                            <h4>최고 모델</h4>
                            <div class="value">{best_model}</div>
                        </div>
                        <div class="stat-card">
                            <h4>분석 모델 수</h4>
                            <div class="value">{len(studies)}</div>
                        </div>
                    </div>
        """
    
    # 모델별 성능 요약 테이블
    html_content += """
                    <h3>🏆 모델별 성능 요약</h3>
                    <table class="summary-table">
                        <thead>
                            <tr>
                                <th>모델</th>
                                <th>최고 성능</th>
                                <th>평균 성능</th>
                                <th>표준편차</th>
                                <th>성공률</th>
                                <th>수렴 상태</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    for model_type, study in studies.items():
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if trials:
            values = [t.value for t in trials]
            best_val = study.best_value
            mean_val = np.mean(values)
            std_val = np.std(values)
            success_rate = len(trials) / len(study.trials) * 100
            
            # 수렴성 분석
            if len(values) >= 5:
                recent_values = values[-5:]
                improvement = recent_values[-1] - recent_values[0]
                if improvement > 0.01:
                    convergence = "개선 중"
                elif abs(improvement) < 0.005:
                    convergence = "수렴됨"
                else:
                    convergence = "불안정"
            else:
                convergence = "데이터 부족"
            
            row_class = "best-performance" if best_val == best_overall else ""
            html_content += f"""
                            <tr class="{row_class}">
                                <td><strong>{model_type}</strong></td>
                                <td>{best_val:.4f}</td>
                                <td>{mean_val:.4f}</td>
                                <td>{std_val:.4f}</td>
                                <td>{success_rate:.1f}%</td>
                                <td>{convergence}</td>
                            </tr>
            """
    
    html_content += """
                        </tbody>
                    </table>
                </div>
    """
    
    # 2. 각 모델별 상세 분석
    for model_type, study in studies.items():
        html_content += f"""
                <div class="model-section">
                    <h3>🎯 {model_type} 상세 분석</h3>
        """
        
        # 기본 통계
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if trials:
            values = [t.value for t in trials]
            best_val = study.best_value
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = min(values)
            max_val = max(values)
            
            html_content += f"""
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h4>최고 성능</h4>
                            <div class="value">{best_val:.4f}</div>
                        </div>
                        <div class="stat-card">
                            <h4>평균 성능</h4>
                            <div class="value">{mean_val:.4f}</div>
                        </div>
                        <div class="stat-card">
                            <h4>표준편차</h4>
                            <div class="value">{std_val:.4f}</div>
                        </div>
                        <div class="stat-card">
                            <h4>성능 범위</h4>
                            <div class="value">{min_val:.4f} ~ {max_val:.4f}</div>
                        </div>
                    </div>
            """
        
        # 차트 데이터 생성
        chart_data = create_simple_chart_data(study, model_type)
        
        if chart_data:
            # 최적화 과정 차트
            if chart_data['trial_numbers'] and chart_data['values']:
                html_content += f"""
                    <h4>📈 최적화 과정</h4>
                    <div class="chart-container">
                        <div id="history_{model_type}"></div>
                    </div>
                    <script>
                        var data = [
                            {{
                                x: {chart_data['trial_numbers']},
                                y: {chart_data['values']},
                                type: 'scatter',
                                mode: 'markers',
                                name: 'Trial 값',
                                marker: {{color: '#667eea', size: 8}}
                            }}
                        ];
                        
                        var layout = {{
                            title: '{model_type} 최적화 과정',
                            xaxis: {{title: 'Trial 번호'}},
                            yaxis: {{title: '성능 값'}},
                            height: 400
                        }};
                        
                        Plotly.newPlot('history_{model_type}', data, layout);
                    </script>
                """
            else:
                html_content += """
                    <h4>📈 최적화 과정</h4>
                    <div class="chart-container">
                        <div class="no-data">데이터가 부족하여 차트를 생성할 수 없습니다.</div>
                    </div>
                """
            
            # 파라미터 중요도 차트
            if chart_data['param_names'] and chart_data['importance_values']:
                html_content += f"""
                    <h4>🔍 파라미터 중요도</h4>
                    <div class="chart-container">
                        <div id="importance_{model_type}"></div>
                    </div>
                    <script>
                        var data = [
                            {{
                                x: {chart_data['importance_values']},
                                y: {chart_data['param_names']},
                                type: 'bar',
                                orientation: 'h',
                                marker: {{color: '#667eea'}}
                            }}
                        ];
                        
                        var layout = {{
                            title: '{model_type} 파라미터 중요도',
                            xaxis: {{title: '중요도'}},
                            yaxis: {{title: '파라미터'}},
                            height: 400
                        }};
                        
                        Plotly.newPlot('importance_{model_type}', data, layout);
                    </script>
                """
            else:
                html_content += """
                    <h4>🔍 파라미터 중요도</h4>
                    <div class="chart-container">
                        <div class="no-data">데이터가 부족하여 차트를 생성할 수 없습니다.</div>
                    </div>
                """
            
            # 파라미터 상관관계 차트
            if chart_data['correlation_data']:
                corr_params = [f"{item['param1']} vs {item['param2']}" for item in chart_data['correlation_data']]
                corr_values = [item['correlation'] for item in chart_data['correlation_data']]
                
                html_content += f"""
                    <h4>🔄 파라미터 상관관계</h4>
                    <div class="chart-container">
                        <div id="correlation_{model_type}"></div>
                    </div>
                    <script>
                        var data = [
                            {{
                                x: {corr_params},
                                y: {corr_values},
                                type: 'bar',
                                marker: {{color: '#667eea'}}
                            }}
                        ];
                        
                        var layout = {{
                            title: '{model_type} 파라미터 상관관계 (상위 5개)',
                            xaxis: {{title: '파라미터 쌍'}},
                            yaxis: {{title: '상관계수'}},
                            height: 400
                        }};
                        
                        Plotly.newPlot('correlation_{model_type}', data, layout);
                    </script>
                """
            else:
                html_content += """
                    <h4>🔄 파라미터 상관관계</h4>
                    <div class="chart-container">
                        <div class="no-data">데이터가 부족하여 차트를 생성할 수 없습니다.</div>
                    </div>
                """
        else:
            html_content += """
                <div class="chart-container">
                    <div class="no-data">데이터가 부족하여 차트를 생성할 수 없습니다.</div>
                </div>
            """
        
        html_content += """
                </div>
        """
    
    # 3. 권장사항 섹션
    html_content += """
                <div class="section">
                    <h2>💡 다음 실험을 위한 권장사항</h2>
    """
    
    for model_type, study in studies.items():
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if trials:
            values = [t.value for t in trials]
            std_val = np.std(values)
            
            html_content += f"""
                    <h3>🎯 {model_type}</h3>
                    <ul>
            """
            
            if std_val > 0.05:
                html_content += f"<li>📈 더 많은 trial 필요 (높은 변동성: {std_val:.4f})</li>"
            
            if len(values) >= 5:
                recent_values = values[-5:]
                improvement = recent_values[-1] - recent_values[0]
                if abs(improvement) < 0.005:
                    html_content += "<li>✅ 현재 설정으로 충분히 최적화됨</li>"
                elif improvement > 0.01:
                    html_content += "<li>🔄 더 많은 trial로 개선 가능</li>"
                else:
                    html_content += "<li>🔧 더 세밀한 파라미터 탐색 필요</li>"
            
            # 파라미터 중요도 기반 권장사항
            try:
                importance = optuna.importance.get_param_importances(study)
                if importance:
                    top_param = max(importance.items(), key=lambda x: x[1])
                    html_content += f"<li>🎯 가장 중요한 파라미터: {top_param[0]} (중요도: {top_param[1]:.4f})</li>"
            except:
                pass
            
            html_content += """
                    </ul>
            """
    
    html_content += """
                </div>
            </div>
            
            <div class="footer">
                <p>🎯 Optuna HPO 분석 대시보드 | 생성 시간: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optuna_simple_dashboard_{timestamp}.html"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 간단한 HTML 대시보드가 '{filename}'에 저장되었습니다!")
    return filename

if __name__ == "__main__":
    # 모든 study 로드
    studies = load_studies()
    
    if not studies:
        print("❌ 로드할 study가 없습니다!")
    else:
        # HTML 대시보드 생성
        dashboard_file = create_html_dashboard(studies)
        
        print("\n🎉 간단한 HTML 대시보드 생성 완료!")
        print("📋 대시보드 내용:")
        print("  - 전체 실험 요약 및 통계")
        print("  - 모델별 상세 분석")
        print("  - 간단한 인터랙티브 차트")
        print("  - 다음 실험을 위한 권장사항")
        print(f"\n📂 파일 위치: {dashboard_file}")
        print("🌐 웹 브라우저에서 열어서 차트를 확인하세요!")
        print("💡 이번에는 확실히 차트가 보일 것입니다!") 