# 📁 파일 정리 가이드

## ✅ 필수 파일 (절대 삭제 금지)

### Python 코드
- `src/TABNET_ENHANCED_MODEL.py` - 메인 모델
- `src/ultra_quick_demo.py` - 빠른 검증

### 설정 파일
- `requirements.txt` - 라이브러리 목록
- `.gitignore` - Git 제외 목록

### 데이터
- `data/` 폴더 전체
- `result/` 폴더 전체

---

## 🎯 실행 파일 (용도별 1개만 선택)

### Windows 실행 파일 (4개 중 1개만 필요)
- ✅ **`train.bat`** ← 이것만 있으면 됨! (가장 간단)
- ⚠️ `run_gpu_training.bat` (한글 문제 있음 - 삭제 가능)
- 🔄 `run_gpu_training_fixed.bat` (영어 버전 - 선택)
- 🔄 `run_safe_training.bat` (안전 모드 - 선택)

### Linux/Mac 실행 파일 (윈도우에서는 불필요)
- ⚠️ `run_full_training.sh` (리눅스/맥 전용 - 윈도우면 삭제 가능)
- ⚠️ `run_safe_training.sh` (리눅스/맥 전용 - 윈도우면 삭제 가능)

### 유틸리티
- ✅ `check_cuda.py` (GPU 확인용 - 유용)
- ✅ `install_gpu.bat` (GPU 설치용 - 유용)

---

## 📚 문서 파일 (선택)

### 메인 문서
- ✅ **`README.md`** ← 필수! (전체 가이드)

### 추가 문서 (3개 - 선택적)
- 🔄 `WINDOWS_GUIDE.md` (윈도우 가이드)
- 🔄 `WINDOWS_GPU_SETUP.md` (GPU 설정 가이드)
- 🔄 `SEGFAULT_FIX.md` (문제 해결 가이드)

**판단:**
- README.md만 잘 정리되어 있으면 나머지는 삭제 가능
- 또는 `docs/` 폴더 만들어서 이동

---

## 🗑️ 삭제 추천 목록

### 윈도우 사용자라면:
```
run_full_training.sh         (리눅스/맥 전용)
run_safe_training.sh         (리눅스/맥 전용)
run_gpu_training.bat         (한글 문제 - 작동 안함)
```

### 문서 정리:
```
선택 1: 모두 유지 (안전)
선택 2: docs/ 폴더 만들어서 이동
선택 3: README.md만 남기고 삭제
```

---

## 💡 권장 정리 방안

### 방안 A: 최소한 정리 (추천)
```
삭제:
- run_gpu_training.bat (작동 안하는 버전)

유지:
- train.bat (실행)
- check_cuda.py (GPU 확인)
- install_gpu.bat (GPU 설치)
- README.md (메인 문서)
- 나머지 문서 3개
```

### 방안 B: 깔끔하게 정리
```
삭제:
- run_gpu_training.bat
- run_full_training.sh
- run_safe_training.sh
- WINDOWS_GUIDE.md
- WINDOWS_GPU_SETUP.md
- SEGFAULT_FIX.md

유지:
- train.bat
- check_cuda.py
- install_gpu.bat
- README.md (모든 내용 통합)
```

### 방안 C: 문서 폴더 정리
```
생성: docs/ 폴더
이동:
- WINDOWS_GUIDE.md → docs/
- WINDOWS_GPU_SETUP.md → docs/
- SEGFAULT_FIX.md → docs/

루트에는:
- README.md만 유지
```

---

## 📊 최종 추천 구조

```
프로젝트/
├── src/                    (Python 코드)
│   ├── TABNET_ENHANCED_MODEL.py
│   └── ultra_quick_demo.py
├── data/                   (데이터)
├── result/                 (결과)
├── docs/                   (문서들)
│   ├── WINDOWS_GUIDE.md
│   ├── WINDOWS_GPU_SETUP.md
│   └── SEGFAULT_FIX.md
├── README.md               (메인 문서)
├── requirements.txt        (라이브러리)
├── .gitignore
├── train.bat              (윈도우 실행)
├── check_cuda.py          (GPU 확인)
└── install_gpu.bat        (GPU 설치)
```

---

## 🎯 결론

**윈도우 사용자라면:**
1. `run_gpu_training.bat` 삭제 (작동 안함)
2. `.sh` 파일들 삭제 (리눅스/맥 전용)
3. 문서들을 `docs/` 폴더로 이동
4. `train.bat` 하나로 실행

**가장 간단하게:**
```cmd
train.bat  (실행)
check_cuda.py (GPU 확인)
README.md (문서)
```
이 3개만 있어도 충분합니다!
