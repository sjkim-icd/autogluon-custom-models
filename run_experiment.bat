@echo off
echo 🚀 실험 파이프라인 시작...
echo.

REM 첫 번째 인수가 실험 이름, 없으면 기본값 사용
set EXPERIMENT_NAME=%1
if "%EXPERIMENT_NAME%"=="" set EXPERIMENT_NAME=titanic_5models_hpo_v1

echo 🎯 실험 이름: %EXPERIMENT_NAME%
echo.

echo 📊 1단계: HPO 실행 시작...
python experiments/optuna_single_stage_hpo_unified_db.py "%EXPERIMENT_NAME%"

if %ERRORLEVEL% NEQ 0 (
    echo ❌ HPO 실행 실패!
    pause
    exit /b 1
)

echo.
echo 📊 2단계: 분석 대시보드 생성 시작...
python analysis/create_final_unified_dashboard_excel_fixed.py "%EXPERIMENT_NAME%"

if %ERRORLEVEL% NEQ 0 (
    echo ❌ 분석 실행 실패!
    pause
    exit /b 1
)

echo.
echo 🎉 실험 파이프라인 완료!
echo 📂 결과 파일들은 results/ 폴더에 저장되었습니다.
pause 