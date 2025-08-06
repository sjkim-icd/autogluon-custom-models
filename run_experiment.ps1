# 실험 파이프라인 실행 스크립트
param(
    [string]$ExperimentName = "titanic_5models_hpo_v1"
)

Write-Host "🚀 실험 파이프라인 시작..." -ForegroundColor Green
Write-Host ""

Write-Host "🎯 실험 이름: $ExperimentName" -ForegroundColor Yellow
Write-Host ""

# 1단계: HPO 실행
Write-Host "📊 1단계: HPO 실행 시작..." -ForegroundColor Cyan
try {
    python experiments/optuna_single_stage_hpo_unified_db.py $ExperimentName
    if ($LASTEXITCODE -ne 0) {
        throw "HPO 실행 실패"
    }
    Write-Host "✅ HPO 실행 완료!" -ForegroundColor Green
} catch {
    Write-Host "❌ HPO 실행 실패: $_" -ForegroundColor Red
    Read-Host "계속하려면 Enter를 누르세요"
    exit 1
}

Write-Host ""

# 2단계: 분석 대시보드 생성
Write-Host "📊 2단계: 분석 대시보드 생성 시작..." -ForegroundColor Cyan
try {
    python analysis/create_final_unified_dashboard_excel_fixed.py $ExperimentName
    if ($LASTEXITCODE -ne 0) {
        throw "분석 실행 실패"
    }
    Write-Host "✅ 분석 실행 완료!" -ForegroundColor Green
} catch {
    Write-Host "❌ 분석 실행 실패: $_" -ForegroundColor Red
    Read-Host "계속하려면 Enter를 누르세요"
    exit 1
}

Write-Host ""
Write-Host "🎉 실험 파이프라인 완료!" -ForegroundColor Green
Write-Host "📂 결과 파일들은 results/ 폴더에 저장되었습니다." -ForegroundColor Yellow
Read-Host "계속하려면 Enter를 누르세요" 