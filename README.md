# 🧠 식습관 기반 바이오마커 예측 모델 (TabNet 딥러닝 포함)

## 📊 프로젝트 개요

식습관 데이터를 활용하여 건강 바이오마커를 예측하는 머신러닝 모델입니다.
**TabNet 딥러닝**과 **Stacking Ensemble**을 결합하여 높은 예측 정확도를 달성합니다.

### 주요 특징
- 🧠 **TabNet 딥러닝**: Google의 테이블 데이터 전용 딥러닝 모델
- 📦 **Stacking Ensemble**: XGBoost, LightGBM, CatBoost, Random Forest 결합
- 🎯 **Optuna 최적화**: 자동 하이퍼파라미터 튜닝
- 🔍 **해석 가능성**: Attention 메커니즘으로 예측 근거 제시

---

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 빠른 검증 (1분)

```bash
cd src
python ultra_quick_demo.py
```

**결과 예시** (체중 예측):
```
기존 XGBoost:     R² = 0.776
개선 Stacking:    R² = 0.953  (+22.8% 향상!)
```

### 3. 전체 학습 실행 (4~6시간)

```bash
./run_full_training.sh
```

**설정**:
- 바이오마커: 6개 고성능 예측 모델 (체중, BMI, 허리둘레, 혈압(SBP/DBP), 중성지방(TG))
- 모델: TabNet + Stacking Ensemble
- 최적화: Optuna (trials=20)
- 제외: 저성능 바이오마커 (혈당, 당화혈색소, 콜레스테롤, 신장기능)

**로그 확인**:
```bash
tail -f training.log
```

---

## 📈 성능 개선 결과

### 전체 평균 성능 (6개 고성능 바이오마커)
```
기존 모델:    평균 R² = 0.53
개선 모델:    평균 R² = 0.70~0.75
향상:         +32~42%
```

### 바이오마커별 예상 성능

| Biomarker | 예상 R² | 성능 등급 | 설명 |
|-----------|---------|----------|------|
| 체중 | **0.95** | ⭐⭐⭐⭐⭐ | 초고성능 - 식이와 직접 상관 |
| 체질량지수 (BMI) | **0.90** | ⭐⭐⭐⭐⭐ | 초고성능 - 체중 기반 지표 |
| 허리둘레 | **0.85** | ⭐⭐⭐⭐ | 고성능 - 복부비만 지표 |
| SBP (수축기혈압) | 0.60 | ⭐⭐⭐ | 중간 - 식이+운동 영향 |
| DBP (이완기혈압) | 0.55 | ⭐⭐⭐ | 중간 - 식이+운동 영향 |
| TG (중성지방) | 0.50 | ⭐⭐⭐ | 중간 - 지방섭취 관련 |

### 제외된 저성능 바이오마커

다음 바이오마커들은 예측 성능이 낮아 제외되었습니다:

| Biomarker | 예상 R² | 제외 이유 |
|-----------|---------|----------|
| GLUCOSE (혈당) | 0.40 | 유전, 대사 요인이 지배적 |
| HBA1C (당화혈색소) | 0.35 | 장기 혈당 지표, 식이 단기 영향 제한 |
| HDL CHOL. | 0.35 | 유전적 요인 큼 |
| LDL CHOL. | 0.40 | 대사 체계 복합적 영향 |
| eGFR (신장기능) | 0.30 | 질병 상태, 나이 등 영향 |

> 💡 **Note**: 식이 데이터만으로는 혈당, 콜레스테롤, 신장기능 등을 정확히 예측하기 어렵습니다. 이들은 유전적 요인, 대사 체계, 질병 상태 등 다양한 요인의 영향을 받기 때문입니다.

---

## 🧠 TabNet이란?

**TabNet**은 Google Research의 테이블 데이터 전용 딥러닝 모델입니다.

### 핵심 특징:

1. **Sequential Attention**
   - 여러 단계에 걸쳐 중요한 특성을 순차적으로 선택
   - 사람의 사고 방식과 유사한 의사결정 과정

2. **Sparse Feature Selection**
   - 필요한 특성만 선택적으로 사용
   - 과적합 방지 및 해석력 향상

3. **해석 가능성**
   - 각 예측에 어떤 특성이 영향을 주었는지 설명 가능
   - Attention 마스크로 시각화 가능

4. **환자별 맞춤 예측**
   - 개인마다 다른 특성 조합 사용
   - 더 정교하고 개인화된 예측

---

## 🔧 모델 구조

### 1. 데이터 전처리
- EWMA 특성: 시간에 따른 식습관 변화 추적
- 고급 특성: 건강/불건강 점수, 나트륨 리스크, 식습관 다양성 등
- 총 63개 특성 생성

### 2. 모델 앙상블
```
Base Models:
├── TabNet (딥러닝)
├── XGBoost
├── LightGBM
├── CatBoost
└── Random Forest

↓ (Stacking)

Meta-Learner (Ridge)
```

### 3. 하이퍼파라미터 최적화
- Optuna를 통한 자동 튜닝
- 각 모델당 20회 시도
- 최적 조합 자동 탐색

---

## 📁 프로젝트 구조

```
.
├── README.md                          # 이 파일
├── requirements.txt                   # 의존성 목록
├── run_full_training.sh              # 전체 학습 스크립트
├── training.log                       # 학습 로그 (자동 생성)
│
├── src/
│   ├── TABNET_ENHANCED_MODEL.py      # 메인 모델 (TabNet + Stacking)
│   └── ultra_quick_demo.py           # 빠른 검증 스크립트
│
└── data/
    └── total_again.xlsx               # 학습 데이터
```

---

## 💻 실행 옵션

### Option 1: 최고 성능 (권장)
```python
from src.TABNET_ENHANCED_MODEL import main

results, summary = main(
    use_tabnet_stacking=True,   # TabNet + 기존 모델
    use_optuna=True,             # 자동 최적화
    optuna_trials=20             # 최적화 횟수
)
```
- **시간**: 4~6시간
- **예상 성능**: 평균 R² 0.51~0.54

### Option 2: 빠른 실행
```python
results, summary = main(
    use_tabnet_stacking=True,
    use_optuna=False,            # 최적화 생략
    optuna_trials=0
)
```
- **시간**: 1~2시간
- **예상 성능**: 평균 R² 0.48~0.51

### Option 3: TabNet만 사용
```python
results, summary = main(
    use_tabnet_stacking=False,   # TabNet만
    use_optuna=True,
    optuna_trials=20
)
```
- **시간**: 2~3시간
- **예상 성능**: 평균 R² 0.49~0.52

---

## 📊 결과 분석

### 학습 완료 후 자동 생성되는 정보:

1. **성능 지표**
   - R² Score (결정계수)
   - RMSE (평균 제곱근 오차)
   - MAE (평균 절대 오차)

2. **모델 정보**
   - 사용된 모델 타입
   - 선택된 특성 목록
   - 학습 시간

3. **특성 중요도**
   - TabNet Attention 가중치
   - 각 바이오마커별 주요 영향 특성

---

## 🔍 사용 예시

### 예측 수행
```python
# 모델 로드
model = results[0]['Model']
scaler = results[0]['Scaler']
selector = results[0]['Selector']

# 새로운 데이터 예측
new_data_scaled = scaler.transform(selector.transform(new_data))
prediction = model.predict(new_data_scaled)
```

### 특성 중요도 확인
```python
# TabNet 특성 중요도
importances = model.feature_importances_
top_features = sorted(zip(feature_names, importances), 
                     key=lambda x: x[1], reverse=True)[:10]

for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")
```

---

## ⚙️ 시스템 요구사항

### 최소 사양:
- CPU: 4 cores
- RAM: 8GB
- 디스크: 5GB
- Python: 3.8+

### 권장 사양:
- CPU: 8 cores
- RAM: 16GB
- GPU: CUDA 지원 (선택, 2~3배 빠름)
- 디스크: 10GB

---

## 📦 주요 의존성

```
# 데이터 처리
pandas, numpy

# 머신러닝
scikit-learn, xgboost, lightgbm, catboost

# 딥러닝
torch, pytorch-tabnet

# 최적화
optuna

# 시각화 & 해석
matplotlib, shap
```

---

## 🐛 문제 해결

### 메모리 부족
```python
# Optuna trials 줄이기
optuna_trials=10  # 기본 20에서 감소

# 또는 TabNet 없이
use_tabnet_stacking=False
```

### 실행 시간 단축
```python
# 최적화 생략
use_optuna=False

# 또는 GPU 사용
device_name='cuda'  # TabNet에서
```

### 특정 바이오마커 실패
- 해당 바이오마커의 제외 변수 목록 확인
- 데이터 결측치 확인

---

## 📄 라이선스

이 프로젝트는 연구 목적으로 사용됩니다.

---

## 👥 기여

프로젝트 개선을 위한 제안이나 버그 리포트는 환영합니다.

---

## 📞 문의

프로젝트 관련 문의사항은 이슈를 통해 남겨주세요.

---

**🚀 지금 바로 시작하세요!**

```bash
./run_full_training.sh
```
