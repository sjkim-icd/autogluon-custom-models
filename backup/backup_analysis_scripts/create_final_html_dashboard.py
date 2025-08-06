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

def safe_json_dumps(obj):
    """안전한 JSON 직렬화"""
    return json.dumps(obj, ensure_ascii=False, default=str)

def create_final_html_dashboard(studies):
    """최종 HTML 대시보드 생성"""
    print("=== 최종 HTML 대시보드 생성 ===")
    
    # HTML 시작
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎯 Optuna HPO 분석 대시보드 (최종)</title>
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
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Optuna HPO 분석 대시보드 (최종)</h1>
                <p>Hyperparameter Optimization 결과 및 시각화</p>
                <p>생성 시간: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
            
            <div class="content">
    """
    
    # 각 모델별 상세 분석
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
                    
                    <div class="debug-info">
                        <strong>디버그 정보:</strong> 완료된 Trial: {len(trials)}개, 성능 범위: {min_val:.4f} ~ {max_val:.4f}
                    </div>
            """
            
            # 1. 최적화 과정 차트
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
                                title: '{model_type} 최적화 과정',
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
            
            # 2. 파라미터 중요도 차트
            try:
                importance = optuna.importance.get_param_importances(study)
                if importance:
                    param_names = list(importance.keys())
                    importance_values = list(importance.values())
                    
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
            
            # 3. 파라미터 상관관계 차트
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
                        correlation_data.sort(key=lambda x: abs(x['correlation']), reverse=True)
                        top_correlations = correlation_data[:5]
                        
                        corr_params = [f"{item['param1']} vs {item['param2']}" for item in top_correlations]
                        corr_values = [item['correlation'] for item in top_correlations]
                        
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
        else:
            html_content += """
                <div class="chart-container">
                    <div class="no-data">완료된 Trial이 없습니다!</div>
                </div>
            """
        
        html_content += """
                </div>
        """
    
    html_content += """
            </div>
            
            <div class="footer">
                <p>🎯 Optuna HPO 분석 대시보드 (최종) | 생성 시간: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
        </div>
        
        <script>
            // 페이지 로드 완료 후 차트 상태 확인
            window.addEventListener('load', function() {
                console.log('페이지 로드 완료');
                setTimeout(function() {
                    var charts = document.querySelectorAll('[id^="history_"], [id^="importance_"], [id^="correlation_"]');
                    console.log('총 차트 수:', charts.length);
                    charts.forEach(function(chart) {
                        if (chart.children.length === 0) {
                            console.log('빈 차트 발견:', chart.id);
                        }
                    });
                }, 2000);
            });
        </script>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optuna_final_dashboard_{timestamp}.html"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 최종 HTML 대시보드가 '{filename}'에 저장되었습니다!")
    return filename

if __name__ == "__main__":
    # 모든 study 로드
    studies = load_studies()
    
    if not studies:
        print("❌ 로드할 study가 없습니다!")
    else:
        # 최종 HTML 대시보드 생성
        dashboard_file = create_final_html_dashboard(studies)
        
        print("\n🎉 최종 HTML 대시보드 생성 완료!")
        print("📋 개선 사항:")
        print("  - 안전한 JSON 직렬화")
        print("  - JavaScript 오류 처리")
        print("  - 콘솔 로그 추가")
        print("  - 차트 상태 확인 기능")
        print(f"\n📂 파일 위치: {dashboard_file}")
        print("🌐 웹 브라우저에서 열어서 모든 차트가 제대로 표시되는지 확인하세요!")
        print("💡 F12 개발자 도구의 Console 탭에서 오류 메시지를 확인할 수 있습니다!") 