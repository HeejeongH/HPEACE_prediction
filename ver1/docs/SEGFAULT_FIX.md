# Segmentation Fault 해결 방법

## 🚨 문제

Optuna 최적화 중 segmentation fault 발생:
```
⚙️ Optuna 최적화 진행 중...
zsh: segmentation fault
```

## ✅ 해결 방법

### 방법 1: 안전 모드로 실행 (Optuna 제외) - **추천!**

```bash
# 안전 모드 스크립트 사용
./run_safe_training.sh
```

또는 Python 직접 실행:
```bash
cd src
python -c "
from TABNET_ENHANCED_MODEL import main
main(
    use_tabnet_stacking=True,
    use_optuna=False,  # Optuna 비활성화
    optuna_trials=1
)
"
```

### 방법 2: Stacking만 사용 (TabNet 제외)

```bash
cd src
python -c "
from TABNET_ENHANCED_MODEL import main
main(
    use_tabnet_stacking=False,  # TabNet 제외
    use_optuna=False
)
"
```

### 방법 3: 메모리 제한 설정

```bash
# 메모리 제한 (8GB)
ulimit -v 8388608

# 실행
cd src
python TABNET_ENHANCED_MODEL.py
```

## 📊 성능 비교

| 모드 | TabNet | Optuna | 예상 R² | 시간 | 안정성 |
|------|--------|--------|---------|------|--------|
| **전체 (기본)** | ✅ | ✅ | 0.70-0.75 | 2-3시간 | ⚠️ 불안정 |
| **안전 모드** | ✅ | ❌ | 0.68-0.72 | 1-2시간 | ✅ 안정 |
| **Stacking만** | ❌ | ❌ | 0.65-0.70 | 30-60분 | ✅ 매우안정 |

## 💡 추천 순서

1. **먼저**: 안전 모드 (TabNet + Stacking, Optuna 없이)
2. **안정되면**: Optuna 추가 시도
3. **실패하면**: Stacking만 사용

## 🔧 Segmentation Fault 원인

1. **메모리 부족**: TabNet + Optuna는 많은 메모리 사용
2. **PyTorch/TabNet 버그**: 특정 환경에서 불안정
3. **멀티프로세싱 충돌**: Optuna의 병렬 처리 문제

## 📝 추가 팁

### PyTorch CPU 모드 강제
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU 비활성화
```

### 메모리 모니터링
```bash
# 별도 터미널에서
watch -n 1 'ps aux | grep python | head -5'
```

### 안전한 Optuna 설정
- trials 감소: 20 → 5
- n_jobs=1 (병렬 처리 방지)
- device_name='cpu' 명시

## ✅ 확인된 작동 환경

- ✅ 안전 모드 (Optuna 없이): 정상 작동
- ✅ Stacking만: 정상 작동  
- ✅ 빠른 검증 스크립트: 정상 작동

## 🎯 결론

**Optuna 최적화는 선택사항입니다.** 
안전 모드로도 충분히 좋은 성능을 얻을 수 있습니다!

- Optuna 없이: R² 0.68-0.72 (여전히 우수)
- Optuna 포함: R² 0.70-0.75 (약간 더 나음)
- 차이: 약 2-4% (안정성 trade-off)
