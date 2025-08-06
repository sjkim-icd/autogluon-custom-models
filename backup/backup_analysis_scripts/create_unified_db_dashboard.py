import json
import pandas as pd
import numpy as np
import optuna
from datetime import datetime

def load_studies_from_unified_db():
    """통합 DB에서 모든 Optuna study 로드"""
    model_types = ['DCNV2', 'CUSTOM_FOCAL_DL', 'RF']
    studies = {}
    
    # 통합 DB 경로
    unified_db_path = 'sqlite:///optuna_studies/all_studies.db'
    
    for model_type in model_types:
        try:
            study = optuna.load_study(
                study_name=f'{model_type}_hpo_study',
                storage=unified_db_path
            )
            studies[model_type] = study
            print(f"✅ {model_type} study 로드 완료 (통합 DB)")
        except Exception as e:
            print(f"❌ {model_type} study 로드 실패: {e}")
    
    return studies

def safe_json_dumps(obj):
    """안전한 JSON 직렬화"""
    return json.dumps(obj, ensure_ascii=False, default=str)

def create_unified_db_dashboard(studies):
    """통합 DB용 개별 필터 대시보드 생성"""
    print("=== 통합 DB 개별 필터 대시보드 생성 ===")
    
    # HTML 시작
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎯 Optuna HPO 통합 DB 대시보드</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1800px;
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
            .chart-row {
                display: flex;
                gap: 20px;
                margin: 20px 0;
                align-items: flex-start;
            }
            .chart-container {
                flex: 1;
                border: 1px solid #eee;
                border-radius: 8px;
                padding: 20px;
                background-color: #fafafa;
                min-height: 400px;
            }
            .filter-panel {
                width: 300px;
                background-color: #e8f4fd;
                border: 1px solid #b3d9ff;
                border-radius: 8px;
                padding: 15px;
                height: fit-content;
            }
            .filter-panel h4 {
                margin: 0 0 15px 0;
                color: #333;
                font-size: 1.1em;
            }
            .filter-controls {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            .filter-controls select, .filter-controls input, .filter-controls button {
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                width: 100%;
                box-sizing: border-box;
            }
            .filter-controls button {
                background-color: #667eea;
                color: white;
                border: none;
                cursor: pointer;
                transition: background-color 0.3s;
                margin-top: 10px;
            }
            .filter-controls button:hover {
                background-color: #5a6fd8;
            }
            .filter-controls button.reset {
                background-color: #dc3545;
            }
            .filter-controls button.reset:hover {
                background-color: #c82333;
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
            .no-data {
                text-align: center;
                padding: 40px;
                color: #666;
                font-style: italic;
            }
            .debug-info {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 4px;
                padding: 10px;
                margin: 10px 0;
                font-size: 0.9em;
                color: #856404;
            }
            .recommendations {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 8px;
                padding: 25px;
                margin: 20px 0;
            }
            .recommendations h3 {
                margin-top: 0;
                font-size: 1.5em;
                border-bottom: 2px solid rgba(255,255,255,0.3);
                padding-bottom: 10px;
            }
            .recommendations ul {
                margin: 15px 0;
                padding-left: 20px;
            }
            .recommendations li {
                margin: 8px 0;
                line-height: 1.6;
            }
            .db-info {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 8px;
                padding: 15px;
                margin: 20px 0;
                color: #155724;
            }
            .db-info h4 {
                margin: 0 0 10px 0;
                color: #0f5132;
            }
            .footer {
                background-color: #333;
                color: white;
                text-align: center;
                padding: 20px;
                margin-top: 40px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Optuna HPO 통합 DB 대시보드</h1>
                <p>all_studies.db 기반 | 각 차트별 독립적인 필터 + 권장사항</p>
                <p>생성 시간: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
            
            <div class="content">
    """
    
    # DB 정보 섹션
    html_content += """
                <div class="db-info">
                    <h4>📂 통합 DB 정보</h4>
                    <p><strong>DB 파일:</strong> optuna_studies/all_studies.db</p>
                    <p><strong>저장된 Study:</strong> """ + ", ".join([f"{model}_hpo_study" for model in studies.keys()]) + """</p>
                    <p><strong>총 Study 수:</strong> """ + str(len(studies)) + """개</p>
                </div>
    """
    
    # 1. 전체 실험 요약 섹션
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
    
    # 2. 각 모델별 완전한 분석
    for model_type, study in studies.items():
        html_content += f"""
                <div class="model-section">
                    <h3>🎯 {model_type} 완전한 분석 (통합 DB)</h3>
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
                    
                    <div class="debug-info">
                        <strong>통합 DB 정보:</strong> 완료된 Trial: {len(trials)}개, 성능 범위: {min_val:.4f} ~ {max_val:.4f}
                    </div>
            """
            
            # 파라미터 목록 생성
            if trials:
                first_trial = trials[0]
                param_names = list(first_trial.params.keys())
                param_options = ""
                for param in param_names:
                    param_options += f'<option value="{param}">{param}</option>'
                
                # 1. 최적화 과정 차트 (필터 없음)
                trial_numbers = [t.number for t in trials]
                values = [t.value for t in trials]
                
                html_content += f"""
                    <h4>📈 최적화 과정</h4>
                    <div class="chart-container">
                        <div id="history_{model_type}"></div>
                    </div>
                    <script>
                        try {{
                            var data = [
                                {{
                                    x: {safe_json_dumps(trial_numbers)},
                                    y: {safe_json_dumps(values)},
                                    type: 'scatter',
                                    mode: 'markers',
                                    name: 'Trial 값',
                                    marker: {{color: '#667eea', size: 8}}
                                }}
                            ];
                            
                            var layout = {{
                                title: '{model_type} 최적화 과정 (통합 DB)',
                                xaxis: {{title: 'Trial 번호'}},
                                yaxis: {{title: '성능 값'}},
                                height: 400
                            }};
                            
                            Plotly.newPlot('history_{model_type}', data, layout);
                            console.log('{model_type} 최적화 과정 차트 생성 성공');
                        }} catch(e) {{
                            console.error('{model_type} 최적화 과정 차트 오류:', e);
                            document.getElementById('history_{model_type}').innerHTML = '<div class="no-data">차트 생성 오류: ' + e.message + '</div>';
                        }}
                    </script>
                """
                
                # 2. 파라미터 중요도 차트 (필터 없음)
                try:
                    importance = optuna.importance.get_param_importances(study)
                    if importance:
                        # 중요도 순으로 정렬 (높은 순)
                        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                        param_names = [item[0] for item in sorted_importance]
                        importance_values = [item[1] for item in sorted_importance]
                        importance_texts = [f'{v:.4f}' for v in importance_values]
                        
                        html_content += f"""
                            <h4>🔍 파라미터 중요도</h4>
                            <div class="chart-container">
                                <div id="importance_{model_type}"></div>
                            </div>
                            <script>
                                try {{
                                    var data = [
                                        {{
                                            x: {safe_json_dumps(importance_values)},
                                            y: {safe_json_dumps(param_names)},
                                            type: 'bar',
                                            orientation: 'h',
                                            marker: {{color: '#667eea'}},
                                            text: {safe_json_dumps(importance_texts)},
                                            textposition: 'auto'
                                        }}
                                    ];
                                    
                                    var layout = {{
                                        title: '{model_type} 파라미터 중요도 (높은 순)',
                                        xaxis: {{title: '중요도'}},
                                        yaxis: {{title: '파라미터'}},
                                        height: 400,
                                        margin: {{l: 150, r: 50, t: 50, b: 50}}
                                    }};
                                    
                                    Plotly.newPlot('importance_{model_type}', data, layout);
                                    console.log('{model_type} 파라미터 중요도 차트 생성 성공');
                                }} catch(e) {{
                                    console.error('{model_type} 파라미터 중요도 차트 오류:', e);
                                    document.getElementById('importance_{model_type}').innerHTML = '<div class="no-data">차트 생성 오류: ' + e.message + '</div>';
                                }}
                            </script>
                        """
                    else:
                        html_content += f"""
                            <h4>🔍 파라미터 중요도</h4>
                            <div class="chart-container">
                                <div class="no-data">파라미터 중요도를 계산할 수 없습니다. (데이터 부족)</div>
                            </div>
                        """
                except Exception as e:
                    html_content += f"""
                        <h4>🔍 파라미터 중요도</h4>
                        <div class="chart-container">
                            <div class="no-data">파라미터 중요도 계산 오류: {str(e)}</div>
                        </div>
                    """
                
                # 3. 파라미터 상관관계 차트 (필터 없음)
                try:
                    data = []
                    for trial in trials:
                        row = {'value': trial.value}
                        row.update(trial.params)
                        data.append(row)
                    
                    df = pd.DataFrame(data)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if 'value' in numeric_cols:
                        numeric_cols.remove('value')
                    
                    if len(numeric_cols) >= 2:
                        correlation_data = []
                        for i, col1 in enumerate(numeric_cols):
                            for j, col2 in enumerate(numeric_cols):
                                if i < j:
                                    corr = df[col1].corr(df[col2])
                                    correlation_data.append({
                                        'param1': col1,
                                        'param2': col2,
                                        'correlation': corr
                                    })
                        
                        if correlation_data:
                            # 절댓값 기준으로 정렬 (높은 상관관계 순)
                            correlation_data.sort(key=lambda x: abs(x['correlation']), reverse=True)
                            top_correlations = correlation_data[:5]
                            
                            corr_params = [f"{item['param1']} vs {item['param2']}" for item in top_correlations]
                            corr_values = [item['correlation'] for item in top_correlations]
                            corr_texts = [f'{v:.4f}' for v in corr_values]
                            
                            html_content += f"""
                                <h4>🔄 파라미터 상관관계</h4>
                                <div class="chart-container">
                                    <div id="correlation_{model_type}"></div>
                                </div>
                                <script>
                                    try {{
                                        var data = [
                                            {{
                                                x: {safe_json_dumps(corr_params)},
                                                y: {safe_json_dumps(corr_values)},
                                                type: 'bar',
                                                marker: {{color: '#667eea'}},
                                                text: {safe_json_dumps(corr_texts)},
                                                textposition: 'auto'
                                            }}
                                        ];
                                        
                                        var layout = {{
                                            title: '{model_type} 파라미터 상관관계 (높은 순)',
                                            xaxis: {{title: '파라미터 쌍'}},
                                            yaxis: {{title: '상관계수'}},
                                            height: 400,
                                            margin: {{l: 200, r: 50, t: 50, b: 50}}
                                        }};
                                        
                                        Plotly.newPlot('correlation_{model_type}', data, layout);
                                        console.log('{model_type} 파라미터 상관관계 차트 생성 성공');
                                    }} catch(e) {{
                                        console.error('{model_type} 파라미터 상관관계 차트 오류:', e);
                                        document.getElementById('correlation_{model_type}').innerHTML = '<div class="no-data">차트 생성 오류: ' + e.message + '</div>';
                                    }}
                                </script>
                            """
                        else:
                            html_content += f"""
                                <h4>🔄 파라미터 상관관계</h4>
                                <div class="chart-container">
                                    <div class="no-data">상관관계를 계산할 수 없습니다. (충분한 파라미터 없음)</div>
                                </div>
                            """
                    else:
                        html_content += f"""
                            <h4>🔄 파라미터 상관관계</h4>
                            <div class="chart-container">
                                <div class="no-data">수치형 파라미터가 부족합니다. (필요: 2개, 현재: {len(numeric_cols)}개)</div>
                            </div>
                        """
                except Exception as e:
                    html_content += f"""
                        <h4>🔄 파라미터 상관관계</h4>
                        <div class="chart-container">
                            <div class="no-data">상관관계 계산 오류: {str(e)}</div>
                        </div>
                    """
                
                # 4. Parallel Coordinate Plot (개별 필터)
                html_content += f"""
                    <h4>🔄 Parallel Coordinate Plot</h4>
                    <div class="chart-row">
                        <div class="chart-container">
                            <div id="parallel_{model_type}"></div>
                        </div>
                        <div class="filter-panel">
                            <h4>🔍 Parallel Coordinate 필터</h4>
                            <div class="filter-controls">
                                <label>성능 최소값:</label>
                                <input type="range" id="{model_type.lower()}-parallel-min" min="0" max="1" step="0.01" value="0.5">
                                <span id="{model_type.lower()}-parallel-min-value">0.50</span>
                                
                                <label>성능 최대값:</label>
                                <input type="range" id="{model_type.lower()}-parallel-max" min="0" max="1" step="0.01" value="1.0">
                                <span id="{model_type.lower()}-parallel-max-value">1.00</span>
                                
                                <button onclick="updateParallelCoordinate('{model_type}')">필터 적용</button>
                                <button class="reset" onclick="resetParallelCoordinate('{model_type}')">초기화</button>
                            </div>
                        </div>
                    </div>
                """
                
                # 5. Contour Plot (개별 필터)
                html_content += f"""
                    <h4>📊 Contour Plot</h4>
                    <div class="chart-row">
                        <div class="chart-container">
                            <div id="contour_{model_type}"></div>
                        </div>
                        <div class="filter-panel">
                            <h4>🔍 Contour Plot 필터</h4>
                            <div class="filter-controls">
                                <label>X축 파라미터:</label>
                                <select id="{model_type.lower()}-contour-x">{param_options}</select>
                                
                                <label>Y축 파라미터:</label>
                                <select id="{model_type.lower()}-contour-y">{param_options}</select>
                                
                                <button onclick="updateContourPlot('{model_type}')">필터 적용</button>
                                <button class="reset" onclick="resetContourPlot('{model_type}')">초기화</button>
                            </div>
                        </div>
                    </div>
                """
                
                # 6. Slice Plot (개별 필터)
                html_content += f"""
                    <h4>📈 Slice Plot</h4>
                    <div class="chart-row">
                        <div class="chart-container">
                            <div id="slice_{model_type}"></div>
                        </div>
                        <div class="filter-panel">
                            <h4>🔍 Slice Plot 필터</h4>
                            <div class="filter-controls">
                                <label>분석 파라미터:</label>
                                <select id="{model_type.lower()}-slice-param">{param_options}</select>
                                
                                <label>성능 범위:</label>
                                <input type="range" id="{model_type.lower()}-slice-range" min="0" max="1" step="0.01" value="0.5">
                                <span id="{model_type.lower()}-slice-range-value">0.50</span>
                                
                                <button onclick="updateSlicePlot('{model_type}')">필터 적용</button>
                                <button class="reset" onclick="resetSlicePlot('{model_type}')">초기화</button>
                            </div>
                        </div>
                    </div>
                """
                
                # 차트 데이터 준비 및 초기화
                if trials:
                    # Parallel Coordinate 데이터
                    parallel_data = []
                    for trial in trials:
                        row = {'value': trial.value}
                        row.update(trial.params)
                        parallel_data.append(row)
                    
                    # 수치형 파라미터만 선택
                    df = pd.DataFrame(parallel_data)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if 'value' in numeric_cols:
                        numeric_cols.remove('value')
                    
                    if len(numeric_cols) >= 2:
                        # Contour Plot용 데이터 (상위 2개 파라미터)
                        top_params = numeric_cols[:2]
                        
                        html_content += f"""
                            <script>
                                // {model_type} 차트 데이터 (통합 DB)
                                var {model_type.lower()}_data = {safe_json_dumps(parallel_data)};
                                var {model_type.lower()}_numeric_cols = {safe_json_dumps(numeric_cols)};
                                var {model_type.lower()}_top_params = {safe_json_dumps(top_params)};
                                
                                // 초기 차트 생성
                                createParallelCoordinate('{model_type}');
                                createContourPlot('{model_type}');
                                createSlicePlot('{model_type}');
                                
                                // 슬라이더 이벤트 리스너
                                document.getElementById('{model_type.lower()}-parallel-min').addEventListener('input', function() {{
                                    document.getElementById('{model_type.lower()}-parallel-min-value').textContent = this.value;
                                }});
                                document.getElementById('{model_type.lower()}-parallel-max').addEventListener('input', function() {{
                                    document.getElementById('{model_type.lower()}-parallel-max-value').textContent = this.value;
                                }});
                                document.getElementById('{model_type.lower()}-slice-range').addEventListener('input', function() {{
                                    document.getElementById('{model_type.lower()}-slice-range-value').textContent = this.value;
                                }});
                            </script>
                        """
                    else:
                        html_content += f"""
                            <div class="no-data">수치형 파라미터가 부족하여 고급 시각화를 생성할 수 없습니다.</div>
                        """
        else:
            html_content += """
                <div class="chart-container">
                    <div class="no-data">완료된 Trial이 없습니다!</div>
                </div>
            """
        
        html_content += """
                </div>
        """
    
    # 3. 권장사항 섹션
    html_content += """
                <div class="recommendations">
                    <h3>💡 다음 실험을 위한 권장사항 (통합 DB 기반)</h3>
                    <ul>
                        <li><strong>통합 DB 활용:</strong> 모든 모델의 최적화 결과가 하나의 DB에 저장되어 있어 비교 분석이 용이합니다.</li>
                        <li><strong>모델 선택:</strong> 현재 실험에서 가장 높은 성능을 보인 모델을 우선적으로 고려하되, 과적합 여부를 반드시 확인하세요.</li>
                        <li><strong>하이퍼파라미터 범위 조정:</strong> 파라미터 중요도 분석 결과를 바탕으로 중요한 파라미터의 탐색 범위를 좁히거나 확장하세요.</li>
                        <li><strong>탐색 전략 개선:</strong> 현재 Random Search를 사용했다면, Bayesian Optimization으로 전환하여 더 효율적인 탐색을 고려하세요.</li>
                        <li><strong>데이터 전처리:</strong> 파라미터 상관관계 분석에서 높은 상관관계를 보이는 파라미터들이 있다면, 이들 중 일부를 제거하여 모델 복잡도를 줄이세요.</li>
                        <li><strong>앙상블 전략:</strong> 개별 모델들의 성능 차이가 크다면, 앙상블 방법을 통해 성능을 향상시킬 수 있습니다.</li>
                        <li><strong>교차 검증:</strong> 현재 단일 검증 세트를 사용했다면, K-Fold 교차 검증을 도입하여 더 안정적인 성능 평가를 하세요.</li>
                        <li><strong>조기 종료:</strong> 학습 곡선을 분석하여 적절한 조기 종료 조건을 설정하여 과적합을 방지하세요.</li>
                        <li><strong>리소스 최적화:</strong> 시간과 컴퓨팅 리소스를 고려하여 탐색 횟수와 시간 제한을 적절히 조정하세요.</li>
                        <li><strong>DB 백업:</strong> 통합 DB 파일(all_studies.db)을 정기적으로 백업하여 실험 결과를 안전하게 보관하세요.</li>
                    </ul>
                </div>
    """
    
    # JavaScript 함수들
    html_content += """
            </div>
            
            <div class="footer">
                <p>🎯 Optuna HPO 통합 DB 대시보드 | 생성 시간: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                <p>📂 DB 파일: optuna_studies/all_studies.db</p>
            </div>
        </div>
        
        <script>
            // Parallel Coordinate Plot 생성
            function createParallelCoordinate(modelType) {
                const data = window[modelType.toLowerCase() + '_data'];
                const numericCols = window[modelType.toLowerCase() + '_numeric_cols'];
                
                if (!data || !numericCols || numericCols.length < 2) {
                    document.getElementById('parallel_' + modelType).innerHTML = '<div class="no-data">데이터 부족</div>';
                    return;
                }
                
                const dimensions = [];
                dimensions.push({
                    label: 'Performance',
                    values: data.map(d => d.value)
                });
                
                numericCols.forEach(col => {
                    dimensions.push({
                        label: col,
                        values: data.map(d => d[col])
                    });
                });
                
                const plotData = [{
                    type: 'parcoords',
                    line: {
                        color: data.map(d => d.value),
                        colorscale: 'Viridis'
                    },
                    dimensions: dimensions
                }];
                
                const layout = {
                    title: modelType + ' Parallel Coordinate Plot (통합 DB)',
                    height: 400
                };
                
                Plotly.newPlot('parallel_' + modelType, plotData, layout);
                console.log('Parallel Coordinate 생성 성공:', modelType);
            }
            
            // Contour Plot 생성
            function createContourPlot(modelType) {
                const data = window[modelType.toLowerCase() + '_data'];
                const topParams = window[modelType.toLowerCase() + '_top_params'];
                
                if (!data || !topParams || topParams.length < 2) {
                    document.getElementById('contour_' + modelType).innerHTML = '<div class="no-data">데이터 부족</div>';
                    return;
                }
                
                const x = data.map(d => d[topParams[0]]);
                const y = data.map(d => d[topParams[1]]);
                const z = data.map(d => d.value);
                
                const plotData = [{
                    type: 'contour',
                    x: x,
                    y: y,
                    z: z,
                    colorscale: 'Viridis'
                }];
                
                const layout = {
                    title: modelType + ' Contour Plot (' + topParams[0] + ' vs ' + topParams[1] + ') - 통합 DB',
                    xaxis: {title: topParams[0]},
                    yaxis: {title: topParams[1]},
                    height: 400
                };
                
                Plotly.newPlot('contour_' + modelType, plotData, layout);
                console.log('Contour Plot 생성 성공:', modelType);
            }
            
            // Slice Plot 생성
            function createSlicePlot(modelType) {
                const data = window[modelType.toLowerCase() + '_data'];
                const numericCols = window[modelType.toLowerCase() + '_numeric_cols'];
                
                if (!data || !numericCols || numericCols.length === 0) {
                    document.getElementById('slice_' + modelType).innerHTML = '<div class="no-data">데이터 부족</div>';
                    return;
                }
                
                const selectedParam = numericCols[0];
                const x = data.map(d => d[selectedParam]);
                const y = data.map(d => d.value);
                
                const plotData = [{
                    type: 'scatter',
                    mode: 'markers',
                    x: x,
                    y: y,
                    marker: {
                        color: y,
                        colorscale: 'Viridis',
                        size: 8
                    }
                }];
                
                const layout = {
                    title: modelType + ' Slice Plot (' + selectedParam + ') - 통합 DB',
                    xaxis: {title: selectedParam},
                    yaxis: {title: 'Performance'},
                    height: 400
                };
                
                Plotly.newPlot('slice_' + modelType, plotData, layout);
                console.log('Slice Plot 생성 성공:', modelType);
            }
            
            // 차트 업데이트 함수들
            function updateParallelCoordinate(modelType) {
                const minRange = parseFloat(document.getElementById(modelType.toLowerCase() + '-parallel-min').value);
                const maxRange = parseFloat(document.getElementById(modelType.toLowerCase() + '-parallel-max').value);
                
                console.log('Parallel Coordinate 업데이트:', modelType, '범위:', minRange, '-', maxRange);
                
                const data = window[modelType.toLowerCase() + '_data'];
                const filteredData = data.filter(d => d.value >= minRange && d.value <= maxRange);
                
                if (filteredData.length > 0) {
                    const numericCols = window[modelType.toLowerCase() + '_numeric_cols'];
                    const dimensions = [];
                    dimensions.push({
                        label: 'Performance',
                        values: filteredData.map(d => d.value)
                    });
                    
                    numericCols.forEach(col => {
                        dimensions.push({
                            label: col,
                            values: filteredData.map(d => d[col])
                        });
                    });
                    
                    const plotData = [{
                        type: 'parcoords',
                        line: {
                            color: filteredData.map(d => d.value),
                            colorscale: 'Viridis'
                        },
                        dimensions: dimensions
                    }];
                    
                    const layout = {
                        title: modelType + ' Parallel Coordinate Plot (필터링됨) - 통합 DB',
                        height: 400
                    };
                    
                    Plotly.newPlot('parallel_' + modelType, plotData, layout);
                    console.log('Parallel Coordinate 업데이트 성공:', modelType, '필터링된 데이터:', filteredData.length);
                } else {
                    document.getElementById('parallel_' + modelType).innerHTML = '<div class="no-data">필터 조건에 맞는 데이터가 없습니다.</div>';
                }
            }
            
            function updateContourPlot(modelType) {
                const xParam = document.getElementById(modelType.toLowerCase() + '-contour-x').value;
                const yParam = document.getElementById(modelType.toLowerCase() + '-contour-y').value;
                
                console.log('Contour Plot 업데이트:', modelType, 'X:', xParam, 'Y:', yParam);
                
                const data = window[modelType.toLowerCase() + '_data'];
                const x = data.map(d => d[xParam]);
                const y = data.map(d => d[yParam]);
                const z = data.map(d => d.value);
                
                const plotData = [{
                    type: 'contour',
                    x: x,
                    y: y,
                    z: z,
                    colorscale: 'Viridis'
                }];
                
                const layout = {
                    title: modelType + ' Contour Plot (' + xParam + ' vs ' + yParam + ') - 통합 DB',
                    xaxis: {title: xParam},
                    yaxis: {title: yParam},
                    height: 400
                };
                
                Plotly.newPlot('contour_' + modelType, plotData, layout);
                console.log('Contour Plot 업데이트 성공:', modelType);
            }
            
            function updateSlicePlot(modelType) {
                const sliceParam = document.getElementById(modelType.toLowerCase() + '-slice-param').value;
                const rangeValue = parseFloat(document.getElementById(modelType.toLowerCase() + '-slice-range').value);
                
                console.log('Slice Plot 업데이트:', modelType, '파라미터:', sliceParam, '범위:', rangeValue);
                
                const data = window[modelType.toLowerCase() + '_data'];
                const filteredData = data.filter(d => d.value >= rangeValue);
                
                if (filteredData.length > 0) {
                    const x = filteredData.map(d => d[sliceParam]);
                    const y = filteredData.map(d => d.value);
                    
                    const plotData = [{
                        type: 'scatter',
                        mode: 'markers',
                        x: x,
                        y: y,
                        marker: {
                            color: y,
                            colorscale: 'Viridis',
                            size: 8
                        }
                    }];
                    
                    const layout = {
                        title: modelType + ' Slice Plot (' + sliceParam + ') - 필터링됨 - 통합 DB',
                        xaxis: {title: sliceParam},
                        yaxis: {title: 'Performance'},
                        height: 400
                    };
                    
                    Plotly.newPlot('slice_' + modelType, plotData, layout);
                    console.log('Slice Plot 업데이트 성공:', modelType, '필터링된 데이터:', filteredData.length);
                } else {
                    document.getElementById('slice_' + modelType).innerHTML = '<div class="no-data">필터 조건에 맞는 데이터가 없습니다.</div>';
                }
            }
            
            // 리셋 함수들
            function resetParallelCoordinate(modelType) {
                document.getElementById(modelType.toLowerCase() + '-parallel-min').value = 0.5;
                document.getElementById(modelType.toLowerCase() + '-parallel-max').value = 1.0;
                document.getElementById(modelType.toLowerCase() + '-parallel-min-value').textContent = '0.50';
                document.getElementById(modelType.toLowerCase() + '-parallel-max-value').textContent = '1.00';
                createParallelCoordinate(modelType);
            }
            
            function resetContourPlot(modelType) {
                createContourPlot(modelType);
            }
            
            function resetSlicePlot(modelType) {
                document.getElementById(modelType.toLowerCase() + '-slice-range').value = 0.5;
                document.getElementById(modelType.toLowerCase() + '-slice-range-value').textContent = '0.50';
                createSlicePlot(modelType);
            }
            
            // 페이지 로드 완료 후 초기화
            window.addEventListener('load', function() {
                console.log('통합 DB 대시보드 로드 완료');
            });
        </script>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optuna_unified_db_dashboard_{timestamp}.html"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 통합 DB 대시보드가 '{filename}'에 저장되었습니다!")
    return filename

if __name__ == "__main__":
    # 통합 DB에서 모든 study 로드
    studies = load_studies_from_unified_db()
    
    if not studies:
        print("❌ 로드할 study가 없습니다!")
    else:
        # 통합 DB 대시보드 생성
        dashboard_file = create_unified_db_dashboard(studies)
        
        print("\n🎉 통합 DB 대시보드 생성 완료!")
        print("📋 포함된 모든 기능:")
        print("  ✅ 통합 DB(all_studies.db) 기반 데이터 로드")
        print("  ✅ 원래 차트들: 최적화 과정, 파라미터 중요도, 파라미터 상관관계")
        print("  ✅ 새로운 차트들: Parallel Coordinate, Contour Plot, Slice Plot")
        print("  ✅ 각 차트별 독립적인 필터 패널")
        print("  ✅ Parallel Coordinate: 성능 범위 필터 (최소/최대)")
        print("  ✅ Contour Plot: X축/Y축 파라미터 선택")
        print("  ✅ Slice Plot: 파라미터 선택 + 성능 범위 필터")
        print("  ✅ 💡 다음 실험을 위한 권장사항 섹션")
        print("  ✅ 📂 통합 DB 정보 표시")
        print(f"\n📂 파일 위치: {dashboard_file}")
        print("🌐 웹 브라우저에서 열어서 모든 기능을 확인하세요!") 