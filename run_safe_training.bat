@echo off
echo ==================================
echo 🚀 안전 모드 학습 (Optuna 제외)
echo ==================================
echo 시작 시간: %date% %time%
echo.
echo 설정:
echo   - 바이오마커: 6개
echo   - 모델: TabNet + Stacking Ensemble
echo   - Optuna 최적화: 비활성화 (안정성)
echo   - 예상 시간: 1~2시간
echo.

cd src

python -c "from TABNET_ENHANCED_MODEL import main; import sys; import time; start_time = time.time(); print('='*80); print('🧠 TabNet + Stacking 학습 시작 (Optuna 없이)'); print('='*80); print(); results, summary = main(use_tabnet_stacking=True, use_optuna=False, optuna_trials=1); elapsed_time = time.time() - start_time; hours = int(elapsed_time // 3600); minutes = int((elapsed_time %% 3600) // 60); print(); print('='*80); print('✅ 학습 완료!'); print('='*80); print(f'총 소요 시간: {hours}시간 {minutes}분'); print(); print('📊 최종 결과:'); print(summary.to_string(index=False)); print()"

echo.
echo ==================================
echo ✅ 완료
echo ==================================
echo 종료 시간: %date% %time%
pause
