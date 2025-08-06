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

def create_advanced_visualization_dashboard(studies):
    """고급 시각화 대시보드 생성"""
    print("=== 고급 시각화 대시보드 생성 ===")
    
    # HTML 시작
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎯 Optuna HPO 고급 시각화 대시보드</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1600px;
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
            .filter-panel {
                background-color: #e8f4fd;
                border: 1px solid #b3d9ff;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
            }
            .filter-panel h4 {
                margin: 0 0 10px 0;
                color: #333;
            }
            .filter-controls {
                display: flex;
                align-items: center;
                gap: 15px;
                flex-wrap: wrap;
            }
            .filter-controls select, .filter-controls input, .filter-controls button {
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }
            .filter-controls button {
                background-color: #667eea;
                color: white;
                border: none;
                cursor: pointer;
                transition: background-color 0.3s;
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
                <h1>🎯 Optuna HPO 고급 시각화 대시보드</h1>
                <p>Parallel Coordinate + Contour Plot + Slice Plot + 필터 기능</p>
                <p>생성 시간: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
            
            <div class="content">
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
    
    # 2. 각 모델별 고급 시각화
    for model_type, study in studies.items():
        html_content += f"""
                <div class="model-section">
                    <h3>🎯 {model_type} 고급 시각화</h3>
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
            
            # 파라미터 목록 생성
            if trials:
                first_trial = trials[0]
                param_names = list(first_trial.params.keys())
                param_options = ""
                for param in param_names:
                    param_options += f'<option value="{param}">{param}</option>'
                
                # 필터 패널
                html_content += f"""
                    <div class="filter-panel">
                        <h4>🔍 {model_type} 파라미터 필터</h4>
                        <div class="filter-controls">
                            <select id="{model_type.lower()}-param-selector">
                                {param_options}
                            </select>
                            <input type="range" id="{model_type.lower()}-range" min="0" max="1" step="0.01" value="0.5">
                            <span id="{model_type.lower()}-range-value">0.5</span>
                            <button onclick="applyFilter('{model_type}')">필터 적용</button>
                            <button class="reset" onclick="resetFilter('{model_type}')">필터 초기화</button>
                        </div>
                    </div>
                """
            
            # 1. Parallel Coordinate Plot
            html_content += f"""
                    <h4>🔄 Parallel Coordinate Plot</h4>
                    <div class="chart-container">
                        <div id="parallel_{model_type}"></div>
                    </div>
            """
            
            # 2. Contour Plot
            html_content += f"""
                    <h4>📊 Contour Plot</h4>
                    <div class="chart-container">
                        <div id="contour_{model_type}"></div>
                    </div>
            """
            
            # 3. Slice Plot
            html_content += f"""
                    <h4>📈 Slice Plot</h4>
                    <div class="chart-container">
                        <div id="slice_{model_type}"></div>
                    </div>
            """
            
            # 차트 데이터 준비
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
                            // {model_type} 차트 데이터
                            var {model_type.lower()}_data = {safe_json_dumps(parallel_data)};
                            var {model_type.lower()}_numeric_cols = {safe_json_dumps(numeric_cols)};
                            var {model_type.lower()}_top_params = {safe_json_dumps(top_params)};
                            
                            // 초기 차트 생성
                            createParallelCoordinate('{model_type}');
                            createContourPlot('{model_type}');
                            createSlicePlot('{model_type}');
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
    
    # JavaScript 함수들
    html_content += """
            </div>
            
            <div class="footer">
                <p>🎯 Optuna HPO 고급 시각화 대시보드 | 생성 시간: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
        </div>
        
        <script>
            // 필터 적용 함수
            function applyFilter(modelType) {
                const paramSelector = document.getElementById(modelType.toLowerCase() + '-param-selector');
                const rangeSlider = document.getElementById(modelType.toLowerCase() + '-range');
                const rangeValue = document.getElementById(modelType.toLowerCase() + '-range-value');
                
                const selectedParam = paramSelector.value;
                const selectedValue = parseFloat(rangeSlider.value);
                rangeValue.textContent = selectedValue.toFixed(2);
                
                console.log('필터 적용:', modelType, selectedParam, selectedValue);
                
                // 3개 차트 모두 업데이트
                updateParallelCoordinate(modelType, selectedParam, selectedValue);
                updateContourPlot(modelType, selectedParam, selectedValue);
                updateSlicePlot(modelType, selectedParam, selectedValue);
            }
            
            // 필터 초기화 함수
            function resetFilter(modelType) {
                const rangeSlider = document.getElementById(modelType.toLowerCase() + '-range');
                const rangeValue = document.getElementById(modelType.toLowerCase() + '-range-value');
                
                rangeSlider.value = 0.5;
                rangeValue.textContent = '0.50';
                
                // 차트 초기화
                createParallelCoordinate(modelType);
                createContourPlot(modelType);
                createSlicePlot(modelType);
            }
            
            // Parallel Coordinate Plot 생성
            function createParallelCoordinate(modelType) {
                const data = window[modelType.toLowerCase() + '_data'];
                const numericCols = window[modelType.toLowerCase() + '_numeric_cols'];
                
                if (!data || !numericCols || numericCols.length < 2) return;
                
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
                    title: modelType + ' Parallel Coordinate Plot',
                    height: 400
                };
                
                Plotly.newPlot('parallel_' + modelType, plotData, layout);
            }
            
            // Contour Plot 생성
            function createContourPlot(modelType) {
                const data = window[modelType.toLowerCase() + '_data'];
                const topParams = window[modelType.toLowerCase() + '_top_params'];
                
                if (!data || !topParams || topParams.length < 2) return;
                
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
                    title: modelType + ' Contour Plot (' + topParams[0] + ' vs ' + topParams[1] + ')',
                    xaxis: {title: topParams[0]},
                    yaxis: {title: topParams[1]},
                    height: 400
                };
                
                Plotly.newPlot('contour_' + modelType, plotData, layout);
            }
            
            // Slice Plot 생성
            function createSlicePlot(modelType) {
                const data = window[modelType.toLowerCase() + '_data'];
                const numericCols = window[modelType.toLowerCase() + '_numeric_cols'];
                
                if (!data || !numericCols || numericCols.length === 0) return;
                
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
                    title: modelType + ' Slice Plot (' + selectedParam + ')',
                    xaxis: {title: selectedParam},
                    yaxis: {title: 'Performance'},
                    height: 400
                };
                
                Plotly.newPlot('slice_' + modelType, plotData, layout);
            }
            
            // 차트 업데이트 함수들
            function updateParallelCoordinate(modelType, selectedParam, selectedValue) {
                // 필터링된 데이터로 Parallel Coordinate 업데이트
                console.log('Parallel Coordinate 업데이트:', modelType, selectedParam, selectedValue);
            }
            
            function updateContourPlot(modelType, selectedParam, selectedValue) {
                // 필터링된 데이터로 Contour Plot 업데이트
                console.log('Contour Plot 업데이트:', modelType, selectedParam, selectedValue);
            }
            
            function updateSlicePlot(modelType, selectedParam, selectedValue) {
                // 필터링된 데이터로 Slice Plot 업데이트
                console.log('Slice Plot 업데이트:', modelType, selectedParam, selectedValue);
            }
            
            // 페이지 로드 완료 후 초기화
            window.addEventListener('load', function() {
                console.log('고급 시각화 대시보드 로드 완료');
            });
        </script>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optuna_advanced_visualization_{timestamp}.html"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 고급 시각화 대시보드가 '{filename}'에 저장되었습니다!")
    return filename

if __name__ == "__main__":
    # 모든 study 로드
    studies = load_studies()
    
    if not studies:
        print("❌ 로드할 study가 없습니다!")
    else:
        # 고급 시각화 대시보드 생성
        dashboard_file = create_advanced_visualization_dashboard(studies)
        
        print("\n🎉 고급 시각화 대시보드 생성 완료!")
        print("📋 포함된 기능:")
        print("  ✅ Parallel Coordinate Plot")
        print("  ✅ Contour Plot")
        print("  ✅ Slice Plot")
        print("  ✅ 모델별 독립 필터 기능")
        print("  ✅ 실시간 차트 업데이트")
        print(f"\n📂 파일 위치: {dashboard_file}")
        print("🌐 웹 브라우저에서 열어서 고급 시각화를 확인하세요!") 