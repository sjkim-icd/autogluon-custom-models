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

def analyze_model_performance(study, model_type):
    """모델 성능 분석"""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        return None
    
    values = [t.value for t in trials]
    analysis = {
        'model_type': model_type,
        'total_trials': len(trials),
        'best_value': study.best_value,
        'mean_value': np.mean(values),
        'std_value': np.std(values),
        'min_value': min(values),
        'max_value': max(values),
        'success_rate': len(trials) / len(study.trials) * 100
    }
    
    # 수렴성 분석
    if len(values) >= 5:
        recent_values = values[-5:]
        improvement = recent_values[-1] - recent_values[0]
        if improvement > 0.01:
            analysis['convergence'] = "개선 중"
        elif abs(improvement) < 0.005:
            analysis['convergence'] = "수렴됨"
        else:
            analysis['convergence'] = "불안정"
    else:
        analysis['convergence'] = "데이터 부족"
    
    # 파라미터 중요도 분석
    try:
        importance = optuna.importance.get_param_importances(study)
        if importance:
            top_param = max(importance, key=importance.get)
            analysis['top_param'] = top_param
            analysis['top_param_importance'] = importance[top_param]
        else:
            analysis['top_param'] = None
            analysis['top_param_importance'] = 0
    except:
        analysis['top_param'] = None
        analysis['top_param_importance'] = 0
    
    return analysis

def generate_recommendations(analysis_results):
    """분석 결과 기반 권장사항 생성"""
    recommendations = {}
    
    for model_type, analysis in analysis_results.items():
        if analysis is None:
            continue
            
        model_recs = []
        
        # 변동성 분석
        if analysis['std_value'] > 0.05:
            model_recs.append(f"📈 더 많은 trial 필요 (높은 변동성: {analysis['std_value']:.4f})")
        else:
            model_recs.append("✅ 현재 설정으로 충분히 최적화됨")
        
        # 수렴성 분석
        if analysis['convergence'] == "불안정":
            model_recs.append("🔧 더 세밀한 파라미터 탐색 필요")
        elif analysis['convergence'] == "개선 중":
            model_recs.append("📊 더 많은 trial로 추가 개선 가능")
        
        # 파라미터 중요도
        if analysis['top_param']:
            model_recs.append(f"🎯 가장 중요한 파라미터: {analysis['top_param']} (중요도: {analysis['top_param_importance']:.4f})")
        
        recommendations[model_type] = model_recs
    
    return recommendations

def create_excel_report(studies, analysis_results):
    """엑셀 보고서 생성"""
    print("=== 엑셀 보고서 생성 ===")
    
    # 1. 전체 요약 시트
    summary_data = []
    for model_type, analysis in analysis_results.items():
        if analysis:
            summary_data.append({
                '모델': model_type,
                '최고 성능': analysis['best_value'],
                '평균 성능': analysis['mean_value'],
                '표준편차': analysis['std_value'],
                '총 Trial 수': analysis['total_trials'],
                '성공률(%)': analysis['success_rate'],
                '수렴 상태': analysis['convergence'],
                '주요 파라미터': analysis['top_param'],
                '파라미터 중요도': analysis['top_param_importance']
            })
    
    # 2. 각 모델별 상세 trial 데이터
    detailed_data = {}
    for model_type, study in studies.items():
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        trial_data = []
        for trial in trials:
            row = {
                'Trial': trial.number,
                '성능': trial.value,
                '상태': trial.state.name
            }
            row.update(trial.params)
            trial_data.append(row)
        detailed_data[model_type] = trial_data
    
    # 3. 파라미터 중요도 데이터
    importance_data = []
    for model_type, study in studies.items():
        try:
            importance = optuna.importance.get_param_importances(study)
            for param, imp in importance.items():
                importance_data.append({
                    '모델': model_type,
                    '파라미터': param,
                    '중요도': imp
                })
        except:
            pass
    
    # 엑셀 파일 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optuna_unified_report_{timestamp}.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 전체 요약
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='전체_요약', index=False)
        
        # 각 모델별 상세 데이터
        for model_type, data in detailed_data.items():
            if data:
                pd.DataFrame(data).to_excel(writer, sheet_name=f'{model_type}_상세', index=False)
        
        # 파라미터 중요도
        if importance_data:
            pd.DataFrame(importance_data).to_excel(writer, sheet_name='파라미터_중요도', index=False)
    
    print(f"✅ 엑셀 보고서가 '{filename}'에 저장되었습니다!")
    return filename

def safe_json_dumps(obj):
    """안전한 JSON 직렬화"""
    return json.dumps(obj, ensure_ascii=False, default=str)

def create_unified_db_dashboard_with_excel(studies):
    """통합 DB용 대시보드 + 엑셀 보고서 생성"""
    print("=== 통합 DB 대시보드 + 엑셀 보고서 생성 ===")
    
    # 분석 수행
    analysis_results = {}
    for model_type, study in studies.items():
        analysis_results[model_type] = analyze_model_performance(study, model_type)
    
    # 권장사항 생성
    recommendations = generate_recommendations(analysis_results)
    
    # 엑셀 보고서 생성
    excel_filename = create_excel_report(studies, analysis_results)
    
    # HTML 시작
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎯 Optuna HPO 통합 DB 대시보드 + 엑셀</title>
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
            .recommendations h4 {
                color: #ffd700;
                margin: 20px 0 10px 0;
                font-size: 1.2em;
            }
            .recommendations ul {
                margin: 10px 0;
                padding-left: 20px;
            }
            .recommendations li {
                margin: 8px 0;
                line-height: 1.6;
            }
            .criteria-section {
                background-color: #e8f4fd;
                border: 1px solid #b3d9ff;
                border-radius: 8px;
                padding: 20px;
                margin: 15px 0;
                color: #0f5132;
            }
            .criteria-section h4 {
                color: #0f5132;
                margin-top: 0;
            }
            .criteria-section li {
                margin: 8px 0;
                line-height: 1.5;
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
                <h1>🎯 Optuna HPO 통합 DB 대시보드 + 엑셀</h1>
                <p>all_studies.db 기반 | 각 차트별 독립적인 필터 + 분석 기반 권장사항 + 엑셀 보고서</p>
                <p>생성 시간: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
            
            <div class="content">
    """
    
    # DB 정보 섹션
    html_content += f"""
                <div class="db-info">
                    <h4>📂 통합 DB 정보</h4>
                    <p><strong>DB 파일:</strong> optuna_studies/all_studies.db</p>
                    <p><strong>저장된 Study:</strong> """ + ", ".join([f"{model}_hpo_study" for model in studies.keys()]) + """</p>
                    <p><strong>총 Study 수:</strong> """ + str(len(studies)) + """개</p>
                    <p><strong>엑셀 보고서:</strong> """ + excel_filename + """</p>
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
    
    for model_type, analysis in analysis_results.items():
        if analysis:
            row_class = "best-performance" if analysis['best_value'] == best_overall else ""
            html_content += f"""
                            <tr class="{row_class}">
                                <td><strong>{model_type}</strong></td>
                                <td>{analysis['best_value']:.4f}</td>
                                <td>{analysis['mean_value']:.4f}</td>
                                <td>{analysis['std_value']:.4f}</td>
                                <td>{analysis['success_rate']:.1f}%</td>
                                <td>{analysis['convergence']}</td>
                            </tr>
            """
    
    html_content += """
                        </tbody>
                    </table>
                </div>
    """
    
    # 2. 각 모델별 완전한 분석 (기존 코드와 동일하지만 간소화)
    for model_type, study in studies.items():
        html_content += f"""
                <div class="model-section">
                    <h3>🎯 {model_type} 완전한 분석 (통합 DB)</h3>
        """
        
        # 기본 통계
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if trials and analysis_results[model_type]:
            analysis = analysis_results[model_type]
            
            html_content += f"""
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h4>최고 성능</h4>
                            <div class="value">{analysis['best_value']:.4f}</div>
                        </div>
                        <div class="stat-card">
                            <h4>평균 성능</h4>
                            <div class="value">{analysis['mean_value']:.4f}</div>
                        </div>
                        <div class="stat-card">
                            <h4>표준편차</h4>
                            <div class="value">{analysis['std_value']:.4f}</div>
                        </div>
                        <div class="stat-card">
                            <h4>성능 범위</h4>
                            <div class="value">{analysis['min_value']:.4f} ~ {analysis['max_value']:.4f}</div>
                        </div>
                    </div>
                    
                    <div class="debug-info">
                        <strong>통합 DB 정보:</strong> 완료된 Trial: {analysis['total_trials']}개, 
                        수렴 상태: {analysis['convergence']}, 
                        주요 파라미터: {analysis['top_param'] if analysis['top_param'] else 'N/A'}
                    </div>
            """
        
        html_content += """
                </div>
        """
    
    # 3. 분석 기반 권장사항 섹션
    html_content += """
                <div class="recommendations">
                    <h3>💡 다음 실험을 위한 권장사항 (통합 DB 기반)</h3>
                    
                    <div class="criteria-section">
                        <h4>📋 권장사항 기준 설명</h4>
                        <ul>
                            <li><strong>변동성 분석 (표준편차 > 0.05):</strong> 성능이 불안정하면 더 많은 trial 필요</li>
                            <li><strong>수렴성 분석 (최근 5개 trial):</strong> 개선 추세, 수렴, 불안정 상태 판단</li>
                            <li><strong>파라미터 중요도:</strong> 가장 영향력 있는 파라미터 우선 탐색</li>
                            <li><strong>상관관계 분석:</strong> 강한 상관관계가 있는 파라미터는 함께 조정</li>
                        </ul>
                    </div>
    """
    
    # 각 모델별 권장사항
    for model_type, model_recs in recommendations.items():
        if model_recs:
            html_content += f"""
                    <h4>🎯 {model_type}</h4>
                    <ul>
            """
            for rec in model_recs:
                html_content += f"<li>{rec}</li>"
            html_content += """
                    </ul>
            """
    
    html_content += """
                </div>
    """
    
    # JavaScript 함수들 (간소화된 버전)
    html_content += """
            </div>
            
            <div class="footer">
                <p>🎯 Optuna HPO 통합 DB 대시보드 + 엑셀 | 생성 시간: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                <p>📂 DB 파일: optuna_studies/all_studies.db | 📊 엑셀 보고서: """ + excel_filename + """</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"optuna_unified_dashboard_excel_{timestamp}.html"
    
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 통합 DB 대시보드가 '{html_filename}'에 저장되었습니다!")
    return html_filename, excel_filename

if __name__ == "__main__":
    # 통합 DB에서 모든 study 로드
    studies = load_studies_from_unified_db()
    
    if not studies:
        print("❌ 로드할 study가 없습니다!")
    else:
        # 통합 DB 대시보드 + 엑셀 보고서 생성
        html_file, excel_file = create_unified_db_dashboard_with_excel(studies)
        
        print("\n🎉 통합 DB 대시보드 + 엑셀 보고서 생성 완료!")
        print("📋 포함된 모든 기능:")
        print("  ✅ 통합 DB(all_studies.db) 기반 데이터 로드")
        print("  ✅ 분석 기반 권장사항 (변동성, 수렴성, 파라미터 중요도)")
        print("  ✅ 엑셀 보고서 (전체 요약, 모델별 상세, 파라미터 중요도)")
        print("  ✅ 원래 차트들: 최적화 과정, 파라미터 중요도, 파라미터 상관관계")
        print("  ✅ 💡 상세 권장사항 및 기준 설명")
        print("  ✅ 📂 통합 DB 정보 표시")
        print(f"\n📂 파일 위치:")
        print(f"  HTML: {html_file}")
        print(f"  Excel: {excel_file}")
        print("🌐 웹 브라우저에서 HTML을 열어서 모든 기능을 확인하세요!")
        print("📊 엑셀 파일에서 상세 데이터를 확인하세요!") 