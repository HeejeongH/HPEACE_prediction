# Python으로 직접 실행하기 가이드

여러 가지 방법으로 Python을 사용하여 모델을 실행할 수 있습니다.

---

## 🚀 방법 1: 실행 스크립트 사용 (가장 편리함, 추천!)

### 대화형 메뉴로 실행
```bash
python run_training.py
```

메뉴가 나타나면 원하는 모드를 선택하세요:
- `1` - 안전 모드 (추천)
- `2` - 전체 최적화 모드
- `3` - TabNet만 사용
- `4` - 빠른 테스트 모드

### 명령줄로 바로 실행
```bash
# 안전 모드 (추천)
python run_training.py safe
# 또는
python run_training.py 1

# 전체 최적화 모드
python run_training.py full
# 또는
python run_training.py 2

# TabNet만 사용
python run_training.py tabnet
# 또는
python run_training.py 3

# 빠른 테스트 모드
python run_training.py quick
# 또는
python run_training.py 4
```

---

## 📝 방법 2: src 폴더에서 직접 실행

```bash
cd src
python TABNET_ENHANCED_MODEL.py
```

기본 설정으로 실행됩니다.

---

## 🎯 방법 3: Python -c 명령으로 실행

### 안전 모드 (Optuna 없음, 추천)
```bash
python -c "import sys; sys.path.insert(0, 'src'); from TABNET_ENHANCED_MODEL import main; main(use_tabnet_stacking=True, use_optuna=False)"
```

### 전체 최적화 모드 (Optuna 포함)
```bash
python -c "import sys; sys.path.insert(0, 'src'); from TABNET_ENHANCED_MODEL import main; main(use_tabnet_stacking=True, use_optuna=True)"
```

### TabNet만 사용
```bash
python -c "import sys; sys.path.insert(0, 'src'); from TABNET_ENHANCED_MODEL import main; main(use_tabnet_stacking=False, use_optuna=False)"
```

### 빠른 테스트 (Optuna 5회)
```bash
python -c "import sys; sys.path.insert(0, 'src'); from TABNET_ENHANCED_MODEL import main; main(use_tabnet_stacking=True, use_optuna=True, optuna_trials=5)"
```

---

## 💻 방법 4: Python 인터랙티브 모드

```python
# Python 실행
python

# 다음 코드 입력
import sys
sys.path.insert(0, 'src')
from TABNET_ENHANCED_MODEL import main

# 실행 (원하는 옵션 선택)
main(use_tabnet_stacking=True, use_optuna=False)  # 안전 모드
```

---

## ⚙️ 실행 모드 상세 설명

### 1. 안전 모드 (추천) ✅
```python
main(use_tabnet_stacking=True, use_optuna=False)
```
- **TabNet + Stacking Ensemble**: 사용
- **Optuna 최적화**: 미사용
- **장점**: 빠르고 안정적, segfault 문제 없음
- **예상 성능**: 체중 R²≈0.95, BMI R²≈0.90
- **실행 시간**: 약 20-30분

### 2. 전체 최적화 모드 ⚡
```python
main(use_tabnet_stacking=True, use_optuna=True, optuna_trials=20)
```
- **TabNet + Stacking Ensemble**: 사용
- **Optuna 최적화**: 사용 (20회)
- **장점**: 최고 성능
- **단점**: Optuna segfault 발생 가능
- **예상 성능**: 체중 R²≈0.96+, BMI R²≈0.91+
- **실행 시간**: 약 60-90분

### 3. TabNet 전용 모드 🧠
```python
main(use_tabnet_stacking=False, use_optuna=False)
```
- **TabNet 딥러닝 모델만**: 사용
- **Stacking Ensemble**: 미사용
- **Optuna 최적화**: 미사용
- **장점**: 순수 딥러닝 모델 성능 확인
- **예상 성능**: 체중 R²≈0.93, BMI R²≈0.88
- **실행 시간**: 약 15-20분

### 4. 빠른 테스트 모드 🏃
```python
main(use_tabnet_stacking=True, use_optuna=True, optuna_trials=5)
```
- **TabNet + Stacking Ensemble**: 사용
- **Optuna 최적화**: 사용 (5회만)
- **장점**: 빠른 최적화
- **단점**: 성능은 전체 최적화보다 약간 낮음
- **예상 성능**: 체중 R²≈0.95+, BMI R²≈0.90+
- **실행 시간**: 약 30-40분

---

## 📊 성능 비교

| 모드 | R² (체중) | R² (BMI) | 실행 시간 | 안정성 |
|------|-----------|----------|-----------|--------|
| 안전 모드 | ~0.95 | ~0.90 | 20-30분 | ⭐⭐⭐⭐⭐ |
| 전체 최적화 | ~0.96+ | ~0.91+ | 60-90분 | ⭐⭐⭐ |
| TabNet 전용 | ~0.93 | ~0.88 | 15-20분 | ⭐⭐⭐⭐⭐ |
| 빠른 테스트 | ~0.95+ | ~0.90+ | 30-40분 | ⭐⭐⭐⭐ |

---

## 🔧 main() 함수 파라미터

```python
def main(use_tabnet_stacking=True, use_optuna=True, optuna_trials=20):
    """
    Args:
        use_tabnet_stacking (bool): TabNet + Stacking Ensemble 사용 여부
        use_optuna (bool): Optuna 하이퍼파라미터 최적화 사용 여부
        optuna_trials (int): Optuna 최적화 시도 횟수 (기본값: 20)
    """
```

### 파라미터 설명:
- **use_tabnet_stacking**: 
  - `True`: TabNet + XGBoost/LightGBM/CatBoost 등을 Stacking으로 결합 (최고 성능)
  - `False`: TabNet 딥러닝 모델만 사용
  
- **use_optuna**: 
  - `True`: Optuna로 하이퍼파라미터 자동 최적화 (최고 성능, segfault 위험)
  - `False`: 기본 하이퍼파라미터 사용 (안정적, 추천)
  
- **optuna_trials**: 
  - Optuna 최적화 시도 횟수
  - 기본값 20: 최고 성능, 오래 걸림
  - 5-10: 빠른 최적화, 성능 약간 낮음
  - 50+: 매우 오래 걸림, 성능 향상 미미

---

## ⚠️ 주의사항

### Optuna Segmentation Fault 문제
- **문제**: Optuna를 사용하면 `Segmentation fault (core dumped)` 오류가 발생할 수 있습니다
- **원인**: PyTorch TabNet과 Optuna의 메모리 관리 충돌
- **해결책**: 
  1. **안전 모드 사용** (추천): `use_optuna=False`
  2. 전체 최적화 필요 시: 여러 번 시도 (운에 따라 성공할 수 있음)
  3. 자세한 내용: `docs/SEGFAULT_FIX.md` 참조

### GPU 메모리 부족
- **증상**: `CUDA out of memory` 오류
- **해결책**: 
  ```python
  # TABNET_ENHANCED_MODEL.py에서 batch_size 줄이기
  batch_size = 256  # 기본값
  # → 128 또는 64로 줄이기
  ```

---

## 🎯 추천 실행 방법

### 첫 실행 (안정성 중시)
```bash
python run_training.py safe
```
또는
```bash
python run_training.py 1
```

### 최고 성능 필요 (시간 많음)
```bash
python run_training.py full
```

### 빠른 결과 확인
```bash
python run_training.py tabnet
```

---

## 📁 출력 결과

모든 결과는 `result/` 폴더에 저장됩니다:
- `tabnet_enhanced_results.csv` - 전체 결과 데이터
- `*_feature_importance_*.png` - 특성 중요도 그래프
- `*_predictions_*.png` - 예측 vs 실제값 그래프
- `*_shap_*.png` - SHAP 분석 그래프

---

## ❓ FAQ

**Q: 가장 추천하는 실행 방법은?**
A: `python run_training.py safe` (안전 모드)

**Q: 최고 성능을 원해요**
A: `python run_training.py full` (전체 최적화), 하지만 segfault 위험

**Q: Optuna segfault 오류가 계속 나요**
A: 안전 모드 사용 (`use_optuna=False`), 성능 차이는 크지 않음 (0.01 R² 정도)

**Q: GPU가 없어요**
A: 괜찮습니다. CPU로도 실행 가능 (조금 느림)

**Q: 실행 시간을 줄이고 싶어요**
A: `python run_training.py tabnet` (TabNet 전용, 15-20분)

---

## 🆘 문제 해결

### ImportError 발생
```bash
pip install -r requirements.txt
```

### GPU 관련 오류
```bash
python check_cuda.py  # GPU 상태 확인
python install_gpu.bat  # GPU PyTorch 재설치
```

### 기타 문제
- Windows 가이드: `docs/WINDOWS_GUIDE.md`
- GPU 설정: `docs/WINDOWS_GPU_SETUP.md`
- Segfault 해결: `docs/SEGFAULT_FIX.md`
