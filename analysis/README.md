# 📊 Analysis Tools

이 폴더는 Optuna HPO 결과를 분석하고 보고서를 생성하는 도구들을 포함합니다.

## 🚀 주요 도구

### `create_final_unified_dashboard_excel_fixed.py`
**최종 통합 분석 및 보고서 생성 도구**

#### 기능:
- ✅ 통합 DB (`optuna_studies/all_studies.db`)에서 데이터 로드
- ✅ HTML 대시보드 생성 (6가지 차트 + 개별 필터)
- ✅ Excel 보고서 생성 (고급 분석 + 조건부 서식)
- ✅ 사용자 지정 권장사항 생성
- ✅ 상관관계 차트 개선 (명확한 변수명 + 강도별 색상)

#### 사용법:
```bash
# 1단계: HPO 실행 (experiments 폴더에서)
python experiments/optuna_single_stage_hpo_unified_db.py

# 2단계: 분석 및 보고서 생성 (analysis 폴더에서)
python analysis/create_final_unified_dashboard_excel_fixed.py
```

#### 생성되는 파일:
- `optuna_unified_dashboard_correlation_fixed_YYYYMMDD_HHMMSS.html` (HTML 대시보드)
- `optuna_advanced_report_unified_db_fixed_YYYYMMDD_HHMMSS.xlsx` (Excel 보고서)

## 📋 워크플로우

1. **HPO 실행**: `experiments/optuna_single_stage_hpo_unified_db.py`
2. **분석 실행**: `analysis/create_final_unified_dashboard_excel_fixed.py`
3. **결과 확인**: HTML 대시보드 + Excel 보고서

## 🎯 특징

- **통합 DB 지원**: 모든 Optuna study를 하나의 SQLite DB에서 관리
- **실시간 상호작용**: HTML 대시보드의 필터 기능
- **과학적 분석**: 파라미터 중요도, 상관관계, 수렴성 분석
- **맞춤형 권장사항**: 분석 결과 기반의 다음 실험 제안 