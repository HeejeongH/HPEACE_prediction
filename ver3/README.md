# Ver3: 식습관 변화 기반 건강지표 및 MetS 예측 모델

## 📋 프로젝트 개요

**Ver3**는 두 번 연속 방문자 데이터를 활용하여 식습관 변화가 건강지표 변화 및 대사증후군(MetS) 발생/개선에 미치는 영향을 예측하는 AI 모델입니다.

### 🔄 버전 비교

| 항목 | Ver1 | Ver2 | **Ver3 (신규)** |
|------|------|------|----------------|
| **분석 방법** | 횡단면 (Cross-sectional) | 종단 변화 (Longitudinal) | **통합 접근** |
| **입력** | 식습관 (1회 방문) | 식습관 변화만 | **Baseline + 변화** |
| **출력** | 건강지표 | 건강지표 변화 | **건강지표 변화 + MetS** |
| **특징** | 현재 상태 추정 | 변화량만 예측 | **통합 예측 + 분류** |
| **성능** | R² 0.90 | R² < 0.05 (실패) | **목표 R² > 0.5** |
| **상태** | ✅ 완료 | ❌ 실패 | 🚀 **개발 완료** |

### 🎯 Ver3의 핵심 개선점

1. **Baseline 정보 활용**: 첫 방문 시점의 건강 상태를 특성으로 포함
2. **이중 목표**: 건강지표 변화 예측 (Regression) + MetS 발생/개선 예측 (Classification)
3. **고급 특성 엔지니어링**: 
   - 식습관 개선 점수
   - 건강/위험 식습관 점수
   - 월별 변화 강도
   - 기준선 위험도
4. **앙상블 모델**: TabNet + XGBoost + LightGBM + CatBoost + Stacking

---

## 📁 프로젝트 구조

```
ver3/
├── src/
│   ├── data_preprocessing.py        # 데이터 전처리 및 paired visits 생성
│   ├── health_prediction_model.py   # 건강지표 변화 예측 (Regression)
│   └── mets_prediction_model.py     # MetS 발생/개선 예측 (Classification)
│
├── models/                          # 학습된 모델 저장
│   ├── health_predictor/
│   └── mets_predictor/
│
├── results/                         # 결과 저장
│   ├── figures/                     # 시각화 결과
│   ├── reports/                     # 분석 보고서
│   └── paired_data_*.csv           # 전처리된 데이터
│
├── notebooks/                       # Jupyter 노트북
│
├── run_ver3_pipeline.py            # 전체 파이프라인 실행
└── README.md                        # 이 파일
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Ver3 디렉토리로 이동
cd ver3

# 필요한 패키지 설치 (상위 디렉토리의 requirements.txt 사용)
pip install -r ../requirements.txt
```

### 2. 전체 파이프라인 실행

**기본 실행 (모든 단계 자동)**:
```bash
python run_ver3_pipeline.py --data ../data/total_again.xlsx
```

**커스텀 옵션**:
```bash
python run_ver3_pipeline.py \
    --data ../data/total_again.xlsx \
    --output ./results \
    --min-gap 90 \
    --max-gap 365 \
    --no-ensemble  # TabNet만 사용 (빠른 실행)
```

**예상 소요 시간**:
- 전처리: 1-2분
- 건강지표 예측 (6개 타겟): 20-30분 (앙상블), 10-15분 (TabNet만)
- MetS 예측: 5-10분
- **총 소요 시간**: 약 30-40분 (GPU 사용 시 더 빠름)

### 3. 개별 단계 실행

**데이터 전처리만**:
```python
from src.data_preprocessing import PairedVisitPreprocessor

preprocessor = PairedVisitPreprocessor(
    min_time_gap=90,   # 최소 3개월
    max_time_gap=365   # 최대 1년
)

processed_df, info = preprocessor.preprocess('../data/total_again.xlsx')
processed_df.to_csv('./results/paired_data.csv', index=False)
```

**건강지표 예측만**:
```python
from src.health_prediction_model import HealthIndicatorPredictor
import pandas as pd

df = pd.read_csv('./results/paired_data.csv')
predictor = HealthIndicatorPredictor(random_state=42)
results = predictor.train_all_targets(df, use_ensemble=True)
predictor.save_models('./models/health_predictor')
```

**MetS 예측만**:
```python
from src.mets_prediction_model import MetSPredictor
import pandas as pd

df = pd.read_csv('./results/paired_data.csv')
predictor = MetSPredictor(random_state=42)
result = predictor.train(df, use_ensemble=True)
predictor.save_model('./models/mets_predictor')
```

---

## 📊 데이터 구조

### 입력 데이터

원본 데이터 (`total_again.xlsx`):
- **R-ID**: 참여자 ID (중복 가능, 여러 방문)
- **수진일**: 방문 날짜
- **식습관 변수**: 간식빈도, 고지방 육류, 단맛, 단백질류, 곡류, 과일, 유제품, 음료류, 인스턴트 가공식품, 짠 간, 짠 식습관, 채소, 튀김 (13개)
- **건강지표**: 체중, BMI, 허리둘레, SBP, DBP, TG, HDL, glucose, HbA1c (9개)
- **인구통계**: 성별, 나이

### Paired Visits 데이터

전처리 후 생성되는 데이터 구조:
- **각 행**: 한 참여자의 연속된 두 번 방문
- **Baseline 변수**: 첫 번째 방문 시점의 값 (예: `체중_baseline`)
- **Change 변수**: 두 번째 방문에서의 변화량 (예: `체중_change`)
- **Change % 변수**: 변화율 (예: `체중_change_pct`)
- **MetS 정보**: Baseline 및 Follow-up MetS 진단, 변화 패턴

**주요 특성**:
- `time_gap_days`: 방문 간격 (90-365일)
- `mets_transition`: MetS 변화 패턴 (stable_no_mets, new_onset, remission, persistent)
- `healthy_score_baseline/change`: 건강한 식습관 점수
- `unhealthy_score_baseline/change`: 위험한 식습관 점수
- `diet_improvement_score`: 종합 식습관 개선 점수
- `baseline_risk`: 기준선 건강 위험도

---

## 🎯 예측 타겟

### 1. 건강지표 변화 예측 (Regression)

6개 주요 건강지표의 변화량을 예측:

| 타겟 | 설명 | 임상적 의의 |
|------|------|------------|
| `체중_change` | 체중 변화 (kg) | 체중 관리 효과 |
| `체질량지수_change` | BMI 변화 | 비만도 변화 |
| `허리둘레(WAIST)_change` | 복부 비만 변화 (cm) | 내장지방 변화 |
| `SBP_change` | 수축기 혈압 변화 (mmHg) | 심혈관 위험도 |
| `DBP_change` | 이완기 혈압 변화 (mmHg) | 고혈압 위험도 |
| `TG_change` | 중성지방 변화 (mg/dL) | 지질대사 개선 |

**평가 지표**:
- **R² Score**: 모델의 설명력 (0~1, 높을수록 좋음)
- **RMSE**: 평균 제곱근 오차 (낮을수록 좋음)
- **MAE**: 평균 절대 오차 (낮을수록 좋음)

### 2. MetS 발생/개선 예측 (Classification)

4가지 MetS 변화 패턴을 예측:

| 클래스 | 설명 | 임상적 의의 |
|--------|------|------------|
| `stable_no_mets` | MetS 없음 유지 | 건강 상태 유지 |
| `new_onset` | MetS 발생 | 위험군, 적극 개입 필요 |
| `remission` | MetS 개선 | 식습관 개선 효과 확인 |
| `persistent` | MetS 지속 | 추가 관리 필요 |

**평가 지표**:
- **Accuracy**: 전체 정확도
- **F1 Score**: 정밀도와 재현율의 조화 평균
- **Confusion Matrix**: 클래스별 예측 정확도
- **Classification Report**: 클래스별 상세 성능

---

## 🧠 모델 아키텍처

### Regression Models (건강지표 예측)

```
입력: Baseline + 식습관 변화
  ↓
━━━━━━━━━━━━━━━━━━━━━━
  Base Models (병렬)
━━━━━━━━━━━━━━━━━━━━━━
  │
  ├─ TabNet (딥러닝)
  ├─ XGBoost
  ├─ LightGBM
  ├─ CatBoost
  └─ RandomForest
  ↓
━━━━━━━━━━━━━━━━━━━━━━
  Stacking Ensemble
  (Meta-learner: Ridge)
━━━━━━━━━━━━━━━━━━━━━━
  ↓
출력: 건강지표 변화 예측값
```

### Classification Model (MetS 예측)

```
입력: Baseline MetS + 식습관 변화
  ↓
━━━━━━━━━━━━━━━━━━━━━━
  Base Models (병렬)
━━━━━━━━━━━━━━━━━━━━━━
  │
  ├─ TabNet Classifier
  ├─ XGBoost Classifier
  ├─ LightGBM Classifier
  └─ CatBoost Classifier
  ↓
━━━━━━━━━━━━━━━━━━━━━━
  Voting Ensemble
  (평균 확률)
━━━━━━━━━━━━━━━━━━━━━━
  ↓
출력: MetS 변화 패턴 (4클래스)
```

---

## 📈 예상 성능

### 건강지표 변화 예측

Ver2의 실패 원인(Baseline 정보 미사용, SNR 낮음)을 개선하여 다음 성능 목표:

| 타겟 | 목표 R² | 예상 RMSE |
|------|---------|----------|
| 체중 | 0.60-0.70 | ±1.5 kg |
| BMI | 0.60-0.70 | ±0.5 kg/m² |
| 허리둘레 | 0.50-0.60 | ±3 cm |
| SBP | 0.40-0.50 | ±10 mmHg |
| DBP | 0.40-0.50 | ±7 mmHg |
| TG | 0.30-0.40 | ±40 mg/dL |

**평균 목표**: R² > 0.5

### MetS 발생/개선 예측

| 지표 | 목표 값 |
|------|---------|
| Accuracy | > 0.70 |
| F1 Score | > 0.65 |
| AUC-ROC | > 0.75 |

---

## 🔍 주요 특성

### 식습관 특성 (19개)

**Baseline (13개)**:
- 채소, 과일, 단백질류, 유제품, 곡류 (건강한 식습관)
- 튀김, 인스턴트 가공식품, 고지방 육류, 음료류, 단맛, 짠 식습관, 간식빈도 (위험 식습관)

**Change (13개)**:
- 각 식습관의 변화량

**고급 특성 (7개)**:
- `healthy_score_baseline/change`: 건강 식습관 점수
- `unhealthy_score_baseline/change`: 위험 식습관 점수
- `diet_improvement_score`: 종합 개선 점수
- `diet_diversity_baseline`: 식습관 다양성
- `baseline_risk`: 기준선 건강 위험도

### 건강지표 특성

**Baseline (9개)**:
- 체중, BMI, 허리둘레, SBP, DBP, TG, HDL, glucose, HbA1c

**MetS 정보 (5개)**:
- `mets_diagnosis_baseline`: Baseline MetS 진단 (0/1)
- `mets_count_baseline`: 충족된 MetS 기준 개수 (0-5)
- 5개 MetS 기준별 충족 여부

---

## 💻 사용 예제

### 1. 새로운 데이터 예측 (건강지표)

```python
import pandas as pd
from src.health_prediction_model import HealthIndicatorPredictor

# 모델 로드
predictor = HealthIndicatorPredictor()
# ... 모델 로드 코드 ...

# 새로운 paired visit 데이터
new_data = pd.DataFrame({
    'sex': ['M'],
    'age_baseline': [45],
    'time_gap_days': [180],
    '체중_baseline': [75.0],
    '채소_baseline': [3],
    '채소_change': [1],  # 채소 섭취 증가
    '튀김_baseline': [2],
    '튀김_change': [-1],  # 튀김 섭취 감소
    # ... 기타 특성 ...
})

# 예측
weight_change_pred = predictor.predict(new_data, '체중_change')
print(f"예상 체중 변화: {weight_change_pred[0]:.2f} kg")
```

### 2. MetS 발생/개선 예측

```python
from src.mets_prediction_model import MetSPredictor

# 모델 로드
predictor = MetSPredictor()
# ... 모델 로드 코드 ...

# 예측 (확률 포함)
result_df = predictor.predict_with_labels(new_data)
print(result_df)

# 출력 예:
#   predicted_class  prob_stable_no_mets  prob_new_onset  prob_remission  prob_persistent
# 0      remission              0.15            0.10          0.65           0.10
```

---

## 📊 결과 해석

### 건강지표 변화 예측

**양수(+)**: 증가 예측
- 체중, BMI, 허리둘레: 증가 (주의 필요)
- 혈압, TG: 증가 (위험 신호)

**음수(-)**: 감소 예측
- 체중, BMI, 허리둘레: 감소 (개선)
- 혈압, TG: 감소 (건강 개선)

### MetS 변화 예측

| 예측 클래스 | 의미 | 권장 사항 |
|------------|------|----------|
| `stable_no_mets` | 건강 유지 | 현재 식습관 유지 |
| `new_onset` | MetS 발생 위험 | 적극적 식습관 개선 필요 |
| `remission` | MetS 개선 가능 | 긍정적 신호, 지속적 관리 |
| `persistent` | MetS 지속 | 추가 의학적 개입 고려 |

---

## 🔧 하이퍼파라미터

### TabNet

```python
TabNetRegressor/Classifier(
    n_d=64,              # Decision layer 크기
    n_a=64,              # Attention layer 크기
    n_steps=5,           # Feature selection 단계
    gamma=1.5,           # Sparsity 강도
    lambda_sparse=1e-4,  # Sparsity regularization
    optimizer_fn=torch.optim.Adam,
    lr=2e-2,             # Learning rate
    max_epochs=100,
    patience=20          # Early stopping
)
```

### XGBoost / LightGBM / CatBoost

```python
n_estimators=300      # 트리 개수
max_depth=6           # 최대 깊이
learning_rate=0.05    # 학습률
subsample=0.8         # 샘플 비율
colsample_bytree=0.8  # 특성 비율
```

---

## ⚠️ 주의사항

### Data Leakage 방지

1. **타겟의 Baseline 제외**: `체중_change` 예측 시 `체중_baseline` 제외하지 않음 (중요한 특성)
2. **Follow-up 정보 제외**: MetS 예측 시 follow-up 건강지표는 사용하지 않음
3. **시간 순서 준수**: Train/Test 분할 시 시간 정보 고려

### 결측치 처리

- 중앙값(median)으로 대체
- 결측치가 30% 이상인 샘플은 제외

### 클래스 불균형

MetS 예측 시 클래스 불균형 가능:
- Stratified split 사용
- 필요 시 SMOTE 등 오버샘플링 고려

---

## 📚 참고 문헌

1. **TabNet**: Arik, S. Ö., & Pfister, T. (2019). TabNet: Attentive Interpretable Tabular Learning. arXiv preprint arXiv:1908.07442.

2. **MetS 진단 기준**: Korean Society for the Study of Obesity (2018). Clinical Practice Guidelines for Obesity.

3. **Ver1 분석**: 기존 횡단면 분석 (R² 0.90)

4. **Ver2 실패 분석**: Longitudinal 변화 예측 (R² < 0.05)

---

## 🤝 기여자

- **SNUH Prediction Team**
- 서울대학교병원 푸드테크 연구팀

---

## 📄 라이선스

이 프로젝트는 연구 목적으로만 사용됩니다.

---

## 📞 문의

질문이나 제안사항이 있으시면 이슈를 등록해주세요.

---

**Last Updated**: 2026-01-02
