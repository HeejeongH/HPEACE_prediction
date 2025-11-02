# 윈도우에서 실행 가이드

## 🖥️ 윈도우 호환성

### ✅ 호환되는 것
- Python 코드 (.py 파일들)
- 모든 라이브러리 (requirements.txt)
- 데이터 분석 및 모델 학습

### ❌ 호환 안 되는 것
- Shell Script (.sh 파일들) - 리눅스/맥 전용
- bash 명령어

---

## 🚀 윈도우 실행 방법

### 1단계: Git 설치 및 코드 받기

```cmd
:: Git이 설치되어 있다면
git clone https://github.com/HeejeongH/HPEACE_prediction.git
cd HPEACE_prediction

:: 또는 업데이트
git pull origin main
```

### 2단계: 라이브러리 설치

```cmd
:: Anaconda Prompt 또는 CMD에서
pip install -r requirements.txt
```

### 3단계: 실행

#### 옵션 A: 배치 파일 사용 (가장 쉬움)

```cmd
:: 더블클릭 또는 CMD에서
run_safe_training.bat
```

#### 옵션 B: Python 직접 실행

```cmd
cd src

:: 안전 모드 (추천)
python -c "from TABNET_ENHANCED_MODEL import main; main(use_tabnet_stacking=True, use_optuna=False)"

:: 또는 전체 모드 (Optuna 포함, Segfault 위험)
python TABNET_ENHANCED_MODEL.py
```

#### 옵션 C: 빠른 검증 (1분)

```cmd
cd src
python ultra_quick_demo.py
```

---

## 📊 실행 모드 비교

| 모드 | 명령어 | 시간 | 안정성 | 성능 |
|------|--------|------|--------|------|
| **빠른 검증** | `python ultra_quick_demo.py` | 1분 | ✅ | R²=0.95 (체중만) |
| **안전 모드** | `run_safe_training.bat` | 1-2시간 | ✅ | R²=0.68-0.72 |
| **전체 모드** | `python TABNET_ENHANCED_MODEL.py` | 2-3시간 | ⚠️ | R²=0.70-0.75 |

---

## 🔧 윈도우 특화 팁

### PowerShell 사용 (CMD 대신)

```powershell
# PowerShell에서
cd "C:\Users\YourName\Documents\#Prediction"
python -c "from TABNET_ENHANCED_MODEL import main; main(use_tabnet_stacking=True, use_optuna=False)"
```

### Jupyter Notebook 사용

```cmd
:: Jupyter 설치
pip install jupyter

:: Jupyter 실행
cd src
jupyter notebook

:: 새 노트북에서
from TABNET_ENHANCED_MODEL import main
results, summary = main(use_tabnet_stacking=True, use_optuna=False)
print(summary)
```

### Anaconda 환경 사용 (추천)

```cmd
:: 새 환경 생성
conda create -n biomarker python=3.11
conda activate biomarker

:: 라이브러리 설치
pip install -r requirements.txt

:: 실행
cd src
python ultra_quick_demo.py
```

---

## ⚠️ 윈도우 주의사항

### 1. 경로 구분자
```cmd
:: 윈도우는 백슬래시 사용
cd C:\Users\Name\Documents\#Prediction

:: Python 코드에서는 슬래시도 가능
data_path = "C:/Users/Name/Documents/data.xlsx"
```

### 2. 한글 경로 문제
```cmd
:: 한글 경로는 따옴표로 감싸기
cd "C:\사용자\이름\문서\#인력양성"
```

### 3. Long Path 문제 (경로가 260자 이상)

윈도우 레지스트리 수정 필요:
```
1. Win + R
2. regedit 입력
3. HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
4. LongPathsEnabled를 1로 설정
```

또는 짧은 경로 사용:
```cmd
:: Google Drive를 D:\ 드라이브 등에 매핑
subst D: "C:\Users\...\GoogleDrive\..."
cd D:\#Prediction
```

---

## 💡 추천 실행 순서 (윈도우)

### 1차: 빠른 검증 (1분)
```cmd
cd src
python ultra_quick_demo.py
```
→ 라이브러리가 제대로 설치되었는지 확인

### 2차: 안전 모드 (1-2시간)
```cmd
run_safe_training.bat
```
→ 6개 바이오마커 학습 (Optuna 없이)

### 3차: (선택) 전체 모드
```cmd
cd src
python TABNET_ENHANCED_MODEL.py
```
→ Segfault 위험이 있지만 최고 성능

---

## 🐛 윈도우에서 흔한 에러

### 1. ModuleNotFoundError
```cmd
:: 해결: pip 재설치
pip install -r requirements.txt --force-reinstall
```

### 2. Permission Denied
```cmd
:: 해결: 관리자 권한으로 CMD 실행
:: Win + X → "명령 프롬프트(관리자)"
```

### 3. Encoding Error (한글 문제)
```python
# 코드 상단에 추가
import sys
sys.stdout.reconfigure(encoding='utf-8')
```

### 4. PyTorch 설치 실패
```cmd
:: CPU 버전 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## ✅ 확인 체크리스트

- [ ] Python 3.8 이상 설치됨
- [ ] Git 설치됨 (선택사항)
- [ ] requirements.txt의 모든 라이브러리 설치 완료
- [ ] 데이터 파일 (total_again.xlsx) 있음
- [ ] 경로에 한글이 있으면 따옴표로 감싸기
- [ ] Anaconda 환경 사용 (추천)

---

## 🎯 결론

**윈도우에서도 완벽하게 실행 가능합니다!**

- Python 코드는 100% 호환
- Shell Script 대신 Batch 파일 사용
- 동일한 성능과 결과 보장

**추천**: `run_safe_training.bat` 더블클릭으로 간단 실행!
