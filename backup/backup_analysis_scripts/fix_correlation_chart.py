import json
import pandas as pd
import numpy as np
import optuna
from datetime import datetime
import os

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

def analyze_parameter_correlation_improved(studies):
    """개선된 파라미터 상관관계 분석"""
    correlation_results = {}
    
    for model_type, study in studies.items():
        try:
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if len(trials) < 3:
                correlation_results[model_type] = {}
                continue
            
            # 숫자형 파라미터만 추출
            numeric_params = {}
            for trial in trials:
                for param, value in trial.params.items():
                    if isinstance(value, (int, float)):
                        if param not in numeric_params:
                            numeric_params[param] = []
                        numeric_params[param].append(value)
            
            # 파라미터가 2개 이상일 때만 상관관계 계산
            if len(numeric_params) < 2:
                correlation_results[model_type] = {}
                print(f"⚠️ {model_type}: 숫자형 파라미터가 부족하여 상관관계 분석 불가")
                continue
            
            # 상관관계 매트릭스 계산
            param_names = list(numeric_params.keys())
            correlation_matrix = {}
            
            for i, param1 in enumerate(param_names):
                for j, param2 in enumerate(param_names):
                    if i < j:  # 중복 제거 (상삼각만)
                        if len(numeric_params[param1]) == len(numeric_params[param2]):
                            corr = np.corrcoef(numeric_params[param1], numeric_params[param2])[0, 1]
                            if not np.isnan(corr):
                                # 더 명확한 라벨링
                                pair_name = f"{param1} ↔ {param2}"
                                correlation_matrix[pair_name] = {
                                    'correlation': corr,
                                    'param1': param1,
                                    'param2': param2,
                                    'abs_correlation': abs(corr)
                                }
            
            correlation_results[model_type] = correlation_matrix
            print(f"✅ {model_type} 상관관계 분석 완료: {len(correlation_matrix)}개 쌍")
            
        except Exception as e:
            print(f"❌ {model_type} 상관관계 분석 실패: {e}")
            correlation_results[model_type] = {}
    
    return correlation_results

def create_debug_correlation_report(studies):
    """상관관계 분석 디버그 리포트 생성"""
    print("=== 파라미터 상관관계 분석 디버그 ===")
    
    correlation_results = analyze_parameter_correlation_improved(studies)
    
    for model_type, correlations in correlation_results.items():
        print(f"\n🎯 {model_type} 상관관계 분석 결과:")
        
        if not correlations:
            print("  ⚠️ 분석 가능한 상관관계가 없습니다.")
            continue
        
        print(f"  📊 총 {len(correlations)}개의 파라미터 쌍 분석됨")
        
        # 상관관계 강도별 정렬
        sorted_correlations = sorted(correlations.items(), 
                                   key=lambda x: x[1]['abs_correlation'], 
                                   reverse=True)
        
        print("  🔍 상관관계 상위 5개:")
        for i, (pair_name, data) in enumerate(sorted_correlations[:5]):
            corr = data['correlation']
            strength = "강함" if abs(corr) > 0.7 else "중간" if abs(corr) > 0.3 else "약함"
            direction = "양의 상관" if corr > 0 else "음의 상관"
            print(f"    {i+1}. {pair_name}: {corr:.4f} ({direction}, {strength})")
    
    return correlation_results

def create_improved_correlation_chart_data(studies):
    """개선된 상관관계 차트용 데이터 생성"""
    correlation_results = analyze_parameter_correlation_improved(studies)
    
    chart_data = {}
    for model_type, correlations in correlation_results.items():
        if correlations:
            # 절댓값 기준으로 정렬 (강한 상관관계부터)
            sorted_correlations = sorted(correlations.items(), 
                                       key=lambda x: x[1]['abs_correlation'], 
                                       reverse=True)
            
            chart_data[model_type] = {
                'pair_names': [item[0] for item in sorted_correlations],
                'correlations': [item[1]['correlation'] for item in sorted_correlations],
                'param1_list': [item[1]['param1'] for item in sorted_correlations],
                'param2_list': [item[1]['param2'] for item in sorted_correlations]
            }
        else:
            chart_data[model_type] = {
                'pair_names': [],
                'correlations': [],
                'param1_list': [],
                'param2_list': []
            }
    
    return chart_data

def create_correlation_test_html():
    """상관관계 차트 테스트용 HTML 생성"""
    studies = load_studies_from_unified_db()
    
    # 디버그 리포트 출력
    correlation_data = create_debug_correlation_report(studies)
    
    # 차트용 데이터 생성
    chart_data = create_improved_correlation_chart_data(studies)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🔗 개선된 파라미터 상관관계 테스트</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .model-section {{
                background-color: #fafafa;
                border-radius: 8px;
                padding: 25px;
                margin-bottom: 30px;
                border-left: 4px solid #667eea;
            }}
            .chart-container {{
                border: 1px solid #eee;
                border-radius: 8px;
                padding: 20px;
                background-color: white;
                margin: 20px 0;
            }}
            .debug-info {{
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 4px;
                padding: 15px;
                margin: 15px 0;
                font-size: 0.9em;
                color: #856404;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔗 개선된 파라미터 상관관계 분석</h1>
                <p>명확한 변수명 표시 + 강도별 정렬</p>
            </div>
    """
    
    for model_type in ['DCNV2', 'CUSTOM_FOCAL_DL', 'RF']:
        model_chart_data = chart_data.get(model_type, {})
        
        html_content += f"""
            <div class="model-section">
                <h2>🎯 {model_type} 파라미터 상관관계</h2>
        """
        
        if model_chart_data.get('correlations'):
            html_content += f"""
                <div class="debug-info">
                    <strong>분석 결과:</strong> {len(model_chart_data['correlations'])}개 파라미터 쌍 |
                    <strong>최고 상관관계:</strong> {max(model_chart_data['correlations'], key=abs):.4f} |
                    <strong>평균 상관관계:</strong> {np.mean([abs(c) for c in model_chart_data['correlations']]):.4f}
                </div>
                <div class="chart-container">
                    <div id="correlation_{model_type}" style="height: 500px;"></div>
                </div>
            """
        else:
            html_content += """
                <div class="debug-info">
                    ⚠️ 분석 가능한 파라미터 상관관계가 없습니다. (숫자형 파라미터 부족 또는 데이터 부족)
                </div>
            """
        
        html_content += "</div>"
    
    html_content += f"""
        </div>
        
        <script>
            const chartData = {json.dumps(chart_data, ensure_ascii=False)};
            
            function createImprovedCorrelationChart(modelType) {{
                const data = chartData[modelType];
                if (!data || !data.correlations || data.correlations.length === 0) {{
                    document.getElementById('correlation_' + modelType).innerHTML = 
                        '<div style="text-align: center; padding: 40px; color: #666;">상관관계 데이터가 없습니다.</div>';
                    return;
                }}
                
                const trace = {{
                    y: data.pair_names,  // Y축: 파라미터 쌍 이름
                    x: data.correlations,  // X축: 상관관계 값
                    type: 'bar',
                    orientation: 'h',
                    marker: {{
                        color: data.correlations.map(corr => {{
                            if (corr > 0.7) return '#d32f2f';      // 강한 양의 상관: 빨간색
                            else if (corr > 0.3) return '#f57c00'; // 중간 양의 상관: 주황색
                            else if (corr > 0) return '#388e3c';   // 약한 양의 상관: 녹색
                            else if (corr > -0.3) return '#1976d2'; // 약한 음의 상관: 파란색
                            else if (corr > -0.7) return '#7b1fa2'; // 중간 음의 상관: 보라색
                            else return '#424242';                  // 강한 음의 상관: 회색
                        }}),
                        line: {{color: '#000', width: 1}}
                    }},
                    text: data.correlations.map(corr => corr.toFixed(3)),
                    textposition: 'auto',
                    hovertemplate: 
                        '<b>%{{y}}</b><br>' +
                        '상관관계: %{{x:.4f}}<br>' +
                        '강도: %{{text}}<br>' +
                        '<extra></extra>'
                }};
                
                const layout = {{
                    title: {{
                        text: `${{modelType}} 파라미터 상관관계 (강도별 정렬)`,
                        font: {{size: 16, color: '#333'}}
                    }},
                    xaxis: {{ 
                        title: 'Correlation Coefficient',
                        range: [-1.1, 1.1],
                        zeroline: true,
                        zerolinecolor: '#000',
                        zerolinewidth: 2,
                        tickformat: '.3f'
                    }},
                    yaxis: {{ 
                        title: 'Parameter Pairs',
                        automargin: true,
                        tickfont: {{size: 11}}
                    }},
                    height: 500,
                    margin: {{l: 200, r: 50, t: 80, b: 80}},
                    plot_bgcolor: '#fafafa',
                    paper_bgcolor: 'white'
                }};
                
                Plotly.newPlot('correlation_' + modelType, [trace], layout);
                console.log(`${{modelType}} 상관관계 차트 생성 완료:`, data.correlations.length, '개 쌍');
            }}
            
            // 페이지 로드 후 모든 차트 생성
            window.addEventListener('load', function() {{
                console.log('개선된 상관관계 차트 테스트 시작');
                ['DCNV2', 'CUSTOM_FOCAL_DL', 'RF'].forEach(modelType => {{
                    createImprovedCorrelationChart(modelType);
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"correlation_test_{timestamp}.html"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ 개선된 상관관계 테스트 파일이 '{filename}'에 저장되었습니다!")
    return filename

if __name__ == "__main__":
    # 상관관계 차트 개선 테스트
    test_file = create_correlation_test_html()
    
    print("\n🎉 개선된 상관관계 차트 분석 완료!")
    print("📋 개선사항:")
    print("  ✅ 파라미터 쌍 이름을 '파라미터1 ↔ 파라미터2' 형식으로 명확하게 표시")
    print("  ✅ 상관관계 강도별로 정렬 (강한 상관관계부터)")
    print("  ✅ 상관관계 강도에 따른 색상 구분")
    print("  ✅ 디버그 정보로 분석 결과 요약 표시")
    print("  ✅ 차트 여백과 폰트 크기 최적화")
    print(f"\n📂 테스트 파일: {test_file}")
    print("🌐 웹 브라우저에서 열어서 개선된 상관관계 차트를 확인하세요!") 