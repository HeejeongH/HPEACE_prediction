# Ver3 추가 내용

## 📁 프로젝트 구조 업데이트

```
HPEACE_prediction/
├── ver1/                      # Ver1: Cross-sectional Analysis ✅
├── ver2/                      # Ver2: Longitudinal Analysis ❌
└── ver3/                      # Ver3: Integrated Prediction 🆕
    ├── src/
    │   ├── data_preprocessing.py         # Paired visits 전처리
    │   ├── health_prediction_model.py    # 건강지표 변화 예측
    │   └── mets_prediction_model.py      # MetS 발생/개선 예측
    ├── models/                # 학습된 모델
    ├── results/               # 결과 및 시각화
    ├── run_ver3_pipeline.py   # 전체 파이프라인
    └── README.md              # Ver3 상세 설명
```

## 🔀 버전 비교 (업데이트)

| 항목 | Ver1 (횡단면) | Ver2 (종단) | **Ver3 (통합)** |
|------|---------------|-------------|----------------|
| **분석 방법** | Cross-sectional | Longitudinal | **Integrated** |
| **입력** | 식습관 (1개 시점) | 식습관 변화만 | **Baseline + 변화** |
| **출력** | 건강지표 | 건강지표 변화량 | **변화량 + MetS** |
| **질문** | "이런 식습관의 건강은?" | "바꾸면 얼마나 변할까?" | **"건강 개선/악화 예측"** |
| **관계** | 상관관계 | 인과관계 시도 | **통합 예측** |
| **데이터** | 29,098개 독립 샘플 | ~4,896개 방문 쌍 | **~4,896개 방문 쌍** |
| **성능** | ✅ R² 0.90 | ❌ R² < 0.05 | **🆕 목표 R² > 0.5** |
| **상태** | ✅ 완료 | ❌ 실패 | **🆕 개발 완료** |

## 🚀 Ver3 빠른 시작

```bash
# Ver3 디렉토리로 이동
cd ver3

# 전체 파이프라인 실행
python run_ver3_pipeline.py --data ../data/total_again.xlsx

# 또는 커스텀 옵션
python run_ver3_pipeline.py \
    --data ../data/total_again.xlsx \
    --output ./results \
    --min-gap 90 \
    --max-gap 365
```

## 📊 Ver3 주요 기능

### 1. 건강지표 변화 예측 (Regression)
- 체중, BMI, 허리둘레, SBP, DBP, TG 변화 예측
- TabNet + XGBoost + LightGBM + CatBoost 앙상블
- 목표: R² > 0.5

### 2. MetS 발생/개선 예측 (Classification)
- 4가지 MetS 변화 패턴 예측:
  - stable_no_mets: MetS 없음 유지
  - new_onset: MetS 발생 (위험)
  - remission: MetS 개선 (긍정)
  - persistent: MetS 지속 (관리 필요)
- 목표: Accuracy > 0.70

## 📖 자세한 내용

Ver3의 상세한 설명, 사용 방법, API 문서는 `ver3/README.md`를 참조하세요.
