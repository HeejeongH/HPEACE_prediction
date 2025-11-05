@echo off
chcp 65001 > nul
echo ==================================
echo GPU Training Start
echo ==================================
echo Start time: %date% %time%
echo.
echo Settings:
echo   - Biomarkers: 6
echo   - Model: TabNet + Stacking Ensemble
echo   - Device: GPU (CUDA)
echo   - Optuna: Disabled (stability)
echo   - Estimated time: 30min-1hour (GPU)
echo.

cd src

python -c "from TABNET_ENHANCED_MODEL import main; import time; start = time.time(); print('='*80); print('GPU Training Started'); print('='*80); print(); results, summary = main(use_tabnet_stacking=True, use_optuna=False, optuna_trials=1); elapsed = time.time() - start; hours = int(elapsed // 3600); minutes = int((elapsed %% 3600) // 60); print(); print('='*80); print('Training Complete!'); print('='*80); print(f'Total time: {hours}h {minutes}m'); print(); print('Results:'); print(summary.to_string(index=False))"

echo.
echo ==================================
echo Complete
echo ==================================
echo End time: %date% %time%
pause
