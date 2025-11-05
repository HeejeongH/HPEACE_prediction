@echo off
echo ========================================
echo 🚀 GPU 버전 PyTorch 설치
echo ========================================
echo.

:: CUDA 버전 선택
echo CUDA 버전을 선택하세요:
echo 1. CUDA 11.8 (권장 - 안정적)
echo 2. CUDA 12.1 (최신)
echo 3. CPU 버전만 (GPU 없음)
echo.
set /p choice="선택 (1-3): "

if "%choice%"=="1" (
    echo.
    echo ✅ CUDA 11.8 버전 설치 중...
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if "%choice%"=="2" (
    echo.
    echo ✅ CUDA 12.1 버전 설치 중...
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else if "%choice%"=="3" (
    echo.
    echo ✅ CPU 버전 설치 중...
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio
) else (
    echo ❌ 잘못된 선택입니다.
    pause
    exit /b 1
)

echo.
echo ========================================
echo 📦 기타 라이브러리 설치 중...
echo ========================================
pip install -r requirements.txt

echo.
echo ========================================
echo ✅ 설치 완료!
echo ========================================
echo.
echo 🔍 CUDA 확인:
python check_cuda.py

echo.
pause
