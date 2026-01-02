# Ver3: 빠른 시작 가이드

## 🎯 개요

Ver3는 **두 번 연속 방문자 데이터**를 활용하여 식습관 변화가 건강에 미치는 영향을 예측합니다.

**핵심 차별점**:
- Ver1 (횡단면): 단일 시점 예측 → R² 0.90
- Ver2 (종단): 변화량만 예측 → **실패** (R² < 0.05)
- **Ver3 (통합)**: Baseline + 변화량 → **목표 R² > 0.5**

---

## ⚡ 5분 안에 시작하기

### 1단계: 환경 준비

```bash
# 프로젝트 클론
git clone https://github.com/HeejeongH/HPEACE_prediction.git
cd HPEACE_prediction

# 의존성 설치
pip install -r requirements.txt
```

### 2단계: Ver3 디렉토리로 이동

```bash
cd ver3
```

### 3단계: 파이프라인 실행

```bash
# 기본 실행 (모든 단계 자동)
python run_ver3_pipeline.py --data ../data/total_again.xlsx
```

**예상 소요 시간**: 30-40분 (GPU 사용 시 더 빠름)

---

## 📊 실행 결과

파이프라인 완료 후 생성되는 파일:

```
ver3/results/
├── paired_data_*.csv              # 전처리된 데이터
├── figures/                       # 시각화 결과
│   ├── health_prediction_performance_*.png
│   ├── mets_confusion_matrix_*.png
│   └── feature_importance_*.png
├── models/                        # 학습된 모델
│   ├── health_predictor_*/
│   └── mets_predictor_*/
└── reports/                       # 분석 보고서
    ├── FINAL_REPORT_*.md
    ├── health_prediction_summary_*.csv
    └── mets_prediction_summary_*.csv
```

---

## 🔧 커스텀 옵션

### 시간 간격 조정

```bash
python run_ver3_pipeline.py \
    --data ../data/total_again.xlsx \
    --min-gap 90 \    # 최소 방문 간격 (일)
    --max-gap 365     # 최대 방문 간격 (일)
```

### 빠른 실행 (TabNet만)

```bash
python run_ver3_pipeline.py \
    --data ../data/total_again.xlsx \
    --no-ensemble     # 앙상블 비활성화
```

**소요 시간**: 15-20분

---

## 📈 성능 기대값

### 건강지표 변화 예측 (Regression)

| 지표 | 목표 R² | 예상 RMSE |
|------|---------|----------|
| 체중 | 0.60-0.70 | ±1.5 kg |
| BMI | 0.60-0.70 | ±0.5 kg/m² |
| 허리둘레 | 0.50-0.60 | ±3 cm |
| SBP | 0.40-0.50 | ±10 mmHg |
| DBP | 0.40-0.50 | ±7 mmHg |
| TG | 0.30-0.40 | ±40 mg/dL |

### MetS 발생/개선 예측 (Classification)

| 지표 | 목표 값 |
|------|---------|
| Accuracy | > 0.70 |
| F1 Score | > 0.65 |

---

## 💡 사용 예제

### 예제 1: 개별 단계 실행

```python
# 전처리만 실행
from src.data_preprocessing import PairedVisitPreprocessor

preprocessor = PairedVisitPreprocessor(min_time_gap=90, max_time_gap=365)
processed_df, info = preprocessor.preprocess('../data/total_again.xlsx')
processed_df.to_csv('./results/paired_data.csv', index=False)
```

### 예제 2: 건강지표 예측

```python
from src.health_prediction_model import HealthIndicatorPredictor
import pandas as pd

# 데이터 로드
df = pd.read_csv('./results/paired_data.csv')

# 모델 학습
predictor = HealthIndicatorPredictor(random_state=42)
results = predictor.train_all_targets(df, use_ensemble=True)

# 모델 저장
predictor.save_models('./models/health_predictor')
```

### 예제 3: MetS 예측

```python
from src.mets_prediction_model import MetSPredictor
import pandas as pd

# 데이터 로드
df = pd.read_csv('./results/paired_data.csv')

# 모델 학습
predictor = MetSPredictor(random_state=42)
result = predictor.train(df, use_ensemble=True)

# 모델 저장
predictor.save_model('./models/mets_predictor')
```

---

## 🐛 문제 해결

### 메모리 부족 오류

```bash
# TabNet만 사용 (메모리 절약)
python run_ver3_pipeline.py --data ../data/total_again.xlsx --no-ensemble
```

### GPU 사용

PyTorch가 자동으로 GPU를 감지합니다. GPU 사용 확인:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 데이터 파일이 없는 경우

`total_again.xlsx` 파일이 `data/` 디렉토리에 있는지 확인하세요.

---

## 📚 추가 문서

- **상세 설명**: `ver3/README.md`
- **API 문서**: 각 모듈의 docstring 참조
- **Ver1 비교**: `../ver1/README.md`

---

## 🤝 도움이 필요하신가요?

GitHub Issues에 질문을 남겨주세요:
https://github.com/HeejeongH/HPEACE_prediction/issues

---

**Last Updated**: 2026-01-02
