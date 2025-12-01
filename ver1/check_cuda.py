"""
CUDA 및 GPU 사용 가능 여부 확인 스크립트
"""
import sys

print("="*80)
print("🔍 CUDA 및 GPU 환경 확인")
print("="*80)
print()

# 1. PyTorch 설치 여부 확인
print("1️⃣ PyTorch 확인 중...")
try:
    import torch
    print(f"   ✅ PyTorch 버전: {torch.__version__}")
except ImportError:
    print("   ❌ PyTorch가 설치되지 않았습니다.")
    print("   설치 명령어:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

print()

# 2. CUDA 사용 가능 여부 확인
print("2️⃣ CUDA 사용 가능 여부...")
if torch.cuda.is_available():
    print("   ✅ CUDA 사용 가능!")
    print(f"   GPU 개수: {torch.cuda.device_count()}개")
    
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
    print(f"   CUDA 버전: {torch.version.cuda}")
    print(f"   cuDNN 버전: {torch.backends.cudnn.version()}")
    
    # 메모리 정보
    print()
    print("3️⃣ GPU 메모리 정보...")
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"   GPU {i} 메모리: {total_memory:.2f} GB")
    
    print()
    print("="*80)
    print("🎉 GPU 가속 사용 가능!")
    print("="*80)
    print()
    print("💡 추천: GPU 모드로 학습하면 2-3배 빠릅니다!")
    print()
    
else:
    print("   ❌ CUDA를 사용할 수 없습니다.")
    print()
    print("📋 가능한 원인:")
    print("   1. NVIDIA GPU가 없음")
    print("   2. NVIDIA 드라이버가 설치되지 않음")
    print("   3. PyTorch CPU 버전이 설치됨")
    print()
    
    # NVIDIA GPU 확인 방법 안내
    print("🔍 NVIDIA GPU 확인 방법:")
    print("   CMD에서 실행: nvidia-smi")
    print()
    
    # CPU 모드로 진행 가능
    print("💡 CPU 모드로도 실행 가능합니다 (느리지만 작동함)")
    print()
    
    # PyTorch 재설치 안내
    print("🔧 GPU 버전 PyTorch 설치 방법:")
    print()
    print("   1. 현재 PyTorch 제거:")
    print("      pip uninstall torch torchvision torchaudio")
    print()
    print("   2. CUDA 11.8 버전 설치 (권장):")
    print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("   3. 또는 CUDA 12.1 버전:")
    print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()

print()
print("="*80)
print("4️⃣ 기타 주요 라이브러리 확인")
print("="*80)
print()

# 주요 라이브러리 확인
libraries = [
    'pytorch_tabnet',
    'optuna',
    'sklearn',
    'pandas',
    'numpy',
    'xgboost',
    'lightgbm',
    'catboost',
    'shap',
    'openpyxl'
]

missing = []
for lib in libraries:
    try:
        if lib == 'sklearn':
            import sklearn
        elif lib == 'pytorch_tabnet':
            import pytorch_tabnet
        elif lib == 'optuna':
            import optuna
        elif lib == 'pandas':
            import pandas
        elif lib == 'numpy':
            import numpy
        elif lib == 'xgboost':
            import xgboost
        elif lib == 'lightgbm':
            import lightgbm
        elif lib == 'catboost':
            import catboost
        elif lib == 'shap':
            import shap
        elif lib == 'openpyxl':
            import openpyxl
        print(f"   ✅ {lib}")
    except ImportError:
        print(f"   ❌ {lib}")
        missing.append(lib)

if missing:
    print()
    print(f"⚠️ 누락된 라이브러리: {', '.join(missing)}")
    print()
    print("설치 명령어:")
    print("pip install -r requirements.txt")
else:
    print()
    print("✅ 모든 필수 라이브러리가 설치되어 있습니다!")

print()
print("="*80)
print("5️⃣ 실행 준비 완료!")
print("="*80)
print()

if torch.cuda.is_available():
    print("🚀 GPU 모드로 실행:")
    print("   run_gpu_training.bat")
else:
    print("🚀 CPU 모드로 실행:")
    print("   run_safe_training.bat")

print()
print("💡 빠른 테스트:")
print("   cd src")
print("   python ultra_quick_demo.py")
print()
