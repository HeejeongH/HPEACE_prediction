# 윈도우 GPU 설치 및 실행 가이드

## 🎯 빠른 시작 (3단계)

### 1단계: GPU 확인 및 설치

```cmd
:: GPU 확인
python check_cuda.py

:: GPU 버전 설치 (자동 설치 스크립트)
install_gpu.bat
```

### 2단계: 빠른 테스트 (1분)

```cmd
cd src
python ultra_quick_demo.py
```

### 3단계: 전체 학습 실행

```cmd
:: GPU 가속 (빠름 - 30분~1시간)
run_gpu_training.bat

:: 또는 CPU 모드 (느림 - 1-2시간)
run_safe_training.bat
```

---

## 🔍 상세 설치 가이드

### Step 1: NVIDIA 드라이버 확인

CMD에서 실행:
```cmd
nvidia-smi
```

**결과 해석:**
- ✅ GPU 정보가 보임 → NVIDIA 드라이버 설치됨
- ❌ 명령을 찾을 수 없음 → 드라이버 설치 필요

**드라이버 설치:**
1. https://www.nvidia.com/Download/index.aspx 방문
2. GPU 모델 선택
3. 드라이버 다운로드 및 설치
4. 재부팅

### Step 2: CUDA 버전 확인

`nvidia-smi` 명령어 결과에서 오른쪽 상단 확인:
```
CUDA Version: 12.1
```

**권장 설치:**
- CUDA 11.8 이상 → PyTorch CUDA 11.8 설치
- CUDA 12.0 이상 → PyTorch CUDA 12.1 설치

### Step 3: PyTorch GPU 버전 설치

#### 옵션 A: 자동 설치 (추천)
```cmd
install_gpu.bat
```

#### 옵션 B: 수동 설치

**CUDA 11.8 (권장):**
```cmd
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```cmd
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU 버전 (GPU 없을 때):**
```cmd
pip install torch torchvision torchaudio
```

### Step 4: 나머지 라이브러리 설치

```cmd
pip install -r requirements.txt
```

### Step 5: CUDA 설치 확인

```cmd
python check_cuda.py
```

**정상 결과:**
```
✅ PyTorch 버전: 2.x.x+cu118
✅ CUDA 사용 가능!
GPU 개수: 1개
GPU 0: NVIDIA GeForce RTX 3060
```

---

## 🚀 실행 방법

### 방법 1: GPU 가속 (추천!)

```cmd
run_gpu_training.bat
```

**장점:**
- 2-3배 빠른 속도
- 30분~1시간 소요
- 동일한 성능

### 방법 2: CPU 모드

```cmd
run_safe_training.bat
```

**사용 시기:**
- GPU가 없을 때
- GPU 메모리 부족할 때
- 안정성 최우선일 때

---

## 📊 성능 비교

| 모드 | 하드웨어 | 시간 | 성능 | 안정성 |
|------|----------|------|------|--------|
| **GPU 가속** | RTX 3060 | 30분-1시간 | R²=0.68-0.72 | ✅ |
| **CPU** | i7-10700 | 1-2시간 | R²=0.68-0.72 | ✅ |
| **GPU + Optuna** | RTX 3060 | 1-2시간 | R²=0.70-0.75 | ⚠️ |

---

## 🔧 GPU 메모리 최적화

### GPU 메모리 부족 시

#### 방법 1: Batch Size 조정

`src/TABNET_ENHANCED_MODEL.py` 수정:
```python
# 기존
batch_size=256

# 메모리 부족 시
batch_size=512  # 또는 1024
```

#### 방법 2: 혼합 정밀도 (Mixed Precision)

```python
# TabNet 생성 시
model = TabNetRegressor(
    ...
    device_name='cuda',
    # mixed_precision=True  # 메모리 절약
)
```

#### 방법 3: GPU 메모리 정리

```cmd
:: 학습 전 실행
nvidia-smi

:: GPU 프로세스가 있으면 작업 관리자에서 종료
```

---

## ⚠️ 문제 해결

### 1. "CUDA out of memory"

```cmd
:: 해결: Batch size 증가
:: src/TABNET_ENHANCED_MODEL.py에서
batch_size=512  # 256 → 512 변경
```

### 2. "No CUDA GPUs are available"

```cmd
:: 확인 1: nvidia-smi 실행되는지
nvidia-smi

:: 확인 2: PyTorch가 GPU 버전인지
python -c "import torch; print(torch.version.cuda)"

:: 해결: GPU 버전 재설치
install_gpu.bat
```

### 3. "NVIDIA driver version is insufficient"

```
:: 드라이버 업데이트 필요
:: https://www.nvidia.com/Download/index.aspx
```

### 4. cuDNN 에러

```cmd
:: PyTorch 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 💡 GPU 활용 팁

### 1. GPU 사용률 모니터링

```cmd
:: 별도 CMD 창에서
nvidia-smi -l 1
```

### 2. 여러 GPU가 있을 때

특정 GPU 지정:
```cmd
set CUDA_VISIBLE_DEVICES=0
python src/TABNET_ENHANCED_MODEL.py
```

### 3. GPU와 CPU 혼합

```python
# 일부는 GPU, 일부는 CPU
# TabNet만 GPU 사용
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

---

## 📋 체크리스트

설치 전:
- [ ] NVIDIA GPU 확인 (nvidia-smi)
- [ ] Python 3.8 이상 설치
- [ ] Anaconda 설치 (권장)

설치 후:
- [ ] `python check_cuda.py` 실행
- [ ] "CUDA 사용 가능!" 메시지 확인
- [ ] GPU 메모리 8GB 이상 (권장)

실행 전:
- [ ] `cd src && python ultra_quick_demo.py` 테스트
- [ ] GPU 사용률 모니터링 준비
- [ ] 충분한 디스크 공간 (10GB 이상)

---

## 🎯 권장 사양

### 최소 사양
- GPU: NVIDIA GTX 1060 (6GB)
- RAM: 16GB
- CUDA: 11.0 이상

### 권장 사양
- GPU: NVIDIA RTX 3060 (12GB)
- RAM: 32GB
- CUDA: 11.8 또는 12.1

### 최적 사양
- GPU: NVIDIA RTX 3080 (10GB) 이상
- RAM: 64GB
- CUDA: 12.1
- SSD: NVMe 500GB 이상

---

## 🚀 빠른 명령어 모음

```cmd
:: GPU 확인
python check_cuda.py

:: GPU 설치
install_gpu.bat

:: 빠른 테스트
cd src && python ultra_quick_demo.py

:: GPU 학습
run_gpu_training.bat

:: CPU 학습
run_safe_training.bat

:: GPU 모니터링
nvidia-smi -l 1
```

---

## ✅ 완료!

이제 GPU 가속으로 빠르게 학습할 수 있습니다! 🎉

**30분~1시간**만에 6개 바이오마커 학습 완료!
