@echo off
echo ==================================
echo 🚀 GPU 가속 학습
echo ==================================
echo 시작 시간: %date% %time%
echo.

:: CUDA 확인
echo 🔍 GPU 확인 중...
python -c "import torch; print('GPU 사용 가능!' if torch.cuda.is_available() else 'GPU 없음 - CPU 모드로 실행'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '')"

if errorlevel 1 (
    echo.
    echo ❌ GPU를 사용할 수 없습니다.
    echo 💡 install_gpu.bat를 먼저 실행하세요.
    pause
    exit /b 1
)

echo.
echo 설정:
echo   - 바이오마커: 6개
echo   - 모델: TabNet + Stacking Ensemble
echo   - 가속: GPU (CUDA)
echo   - Optuna 최적화: 비활성화 (안정성)
echo   - 예상 시간: 30분~1시간 (GPU 가속)
echo.

cd src

python -c "import torch; import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0' if torch.cuda.is_available() else ''; from TABNET_ENHANCED_MODEL import main; import time; start = time.time(); print('='*80); print('🧠 GPU 가속 TabNet + Stacking 학습 시작'); print('='*80); print(); results, summary = main(use_tabnet_stacking=True, use_optuna=False, optuna_trials=1); elapsed = time.time() - start; hours = int(elapsed // 3600); minutes = int((elapsed %% 3600) // 60); print(); print('='*80); print('✅ 학습 완료!'); print('='*80); print(f'총 소요 시간: {hours}시간 {minutes}분'); print(); print('📊 최종 결과:'); print(summary.to_string(index=False))"

echo.
echo ==================================
echo ✅ 완료
echo ==================================
echo 종료 시간: %date% %time%
pause
