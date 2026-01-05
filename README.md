# HPEACE Prediction - MetS 예측 모델

대사증후군(Metabolic Syndrome) 변화 예측을 위한 딥러닝 모델

## 📊 프로젝트 개요

### 목표
연속 방문자의 식습관 변화를 활용하여 대사증후군(MetS) 변화를 예측

### 데이터
- **전체**: 29,098건 → **분석 대상**: 11,238명 (2회 이상 방문자)
- **변수**: 65개 (식습관 19 + 인구학 4 + 생활습관 3 + 바이오마커 11 + 기타)
- **예측 타겟**: `mets_transition` (개선/유지/악화)

## 🎯 현재 성능

### 3-Class 모델 (개선/유지/악화)
```
베이스라인: F1 = 0.490 (5개 질병 평균)
최적화 후: F1 = 0.378 (FocalLoss + Optuna)
문제점: 클래스 불균형 심각 (6:1 ~ 13:1) 🔴
```

### 2-Class 모델 (유지 vs 변화) ⭐ 신규
```
목표: F1 > 0.60 (임상 활용 가능)
클래스 불균형: 3:1 ~ 5:1 (개선됨) ✅
예상 성능: F1 = 0.65 ~ 0.75
```

**클래스 불균형 분석:**
- 개선: 6.7-11.5% (매우 적음)
- 유지: 73.4-84.5% (압도적 다수)
- 악화: 8.2-15.1% (적음)

## 🏗️ 모델 아키텍처

### 1. MultiDiseasePredictor (3-Class) - 기존
```
입력 75차원 → 7개 인코더 (각 8차원) → 합성 56차원 → 출력 3-class

문제점: Hidden dim = 8 (너무 작음), 클래스 불균형
```

### 2. BinaryDiseasePredictor (2-Class) ⭐ 신규 추천
```
입력 75차원 → 4층 신경망 (256 → 128 → 64 → 32) → 출력 2-class

특징:
- Hidden dim = 64 (8배 증가)
- 타겟: 유지(0) vs 변화(1)
- 클래스 불균형 완화 (3:1 ~ 5:1)
- 학습 용이성 향상

학습:
- Optimizer: AdamW (lr=1e-4, wd=6e-4)
- Scheduler: ReduceLROnPlateau
- Early Stopping: patience=15
- Loss: CrossEntropy
- SMOTE: 적용

파일:
- binary_prediction_model.py: 모델 정의 및 학습/평가 함수
- run_binary_model.py: 실행 스크립트
```

## 📂 프로젝트 구조

```
webapp/
├── src/
│   ├── data_import.py              # 데이터 전처리
│   ├── feature_engineering.py      # 피처 엔지니어링 (PCA, Interaction)
│   ├── MetS_prediction_model.py    # 딥러닝 모델 (3-Class)
│   ├── binary_prediction_model.py  # 2-Class 예측 모델 ⭐ 신규
│   ├── run_binary_model.py         # 2-Class 모델 실행 스크립트 ⭐ 신규
│   ├── train_eval_function.py      # 학습/평가
│   ├── loss_functions.py           # Loss 함수 (개선 버전 포함)
│   ├── resampling.py               # SMOTE, 불균형 처리
│   ├── diet_recommend.py           # 식이 추천
│   ├── SHAP.py                     # 해석 가능성
│   ├── utils.py                    # 유틸리티
│   └── main.ipynb                  # 메인 실행 노트북
├── data/                           # 데이터 파일 (gitignore)
├── result/                         # 결과 저장 디렉토리
├── requirements.txt                # 패키지 의존성
├── .gitignore                      # Git 무시 파일
└── README.md                       # 이 파일
```

## 🚀 사용 방법

### 1. 환경 설정
```bash
git clone https://github.com/HeejeongH/HPEACE_prediction.git
cd HPEACE_prediction
pip install -r requirements.txt
```

### 2. 실행

#### 방법 1: 3-Class 모델 (기존)
```bash
cd src
jupyter notebook main.ipynb
```

#### 방법 2: 2-Class 모델 (권장) ⭐ 신규
```bash
cd src
python run_binary_model.py
```

**결과 저장 위치**: `../result/binary_model/`
- `detailed_results.csv`: 질병별 상세 성능
- `summary_results.csv`: 평균 성능 요약
- `binary_model_results.pkl`: 전체 결과 (모델 포함)

## 🔬 개선된 기능 (v0.2.0)

### Loss 함수 개선
**loss_functions.py에 추가:**

1. **`calculate_improved_class_weights()`**
   - 3가지 클래스 가중치 계산 방법
   - inverse_freq / effective_num / balanced

2. **`improved_loss_methods_configs()`**
   - 8가지 Loss 함수 자동 설정
   - CrossEntropy, FocalLoss (gamma=1.5/2.0/2.5)
   - WeightedCE (3가지), FocalLoss+Weighted

**사용 예시:**
```python
from loss_functions import improved_loss_methods_configs

# 개선된 Loss 함수 가져오기
loss_configs = improved_loss_methods_configs(
    {disease_name: train_loader}, disease_name, device
)

# 각 Loss로 학습 및 비교
for loss_name, criterion in loss_configs.items():
    # 모델 학습
    result = train_model_with_loss(model, train_loader, val_loader, 
                                    test_loader, disease_name, 'base', 
                                    loss_method=loss_name)
```

## 📈 향후 계획

### Phase 1: Loss 함수 최적화 (진행 중)
- [x] 개선된 클래스 가중치 구현
- [x] 8가지 Loss 함수 설정
- [ ] Loss 함수별 성능 비교 실험

### Phase 2: 데이터 보강
- [ ] 누락 변수 임퓨테이션 (GLUCOSE, HbA1c, HDL)
- [ ] SMOTE + Tomek Links (Hybrid)

### Phase 3: 앙상블 강화
- [ ] Stacking (RF + GB + LR + XGBoost)
- [ ] Soft Voting 개선

### Phase 4: 최적화 확장
- [ ] Optuna 100 trials (현재 30)
- [ ] K-Fold Cross Validation
- [ ] 모델 체크포인트 저장/로드

## ⚠️ 주의사항

### 누락 변수
현재 데이터에 포함되지 않은 변수:
- GLUCOSE (공복혈당)
- HBA1C (당화혈색소)
- HDL CHOL (고밀도 콜레스테롤)

### 데이터 불균형
클래스 분포 불균형으로 SMOTE 적용 중
→ 개선된 클래스 가중치로 추가 보정 진행 중

## 🔗 링크

- **GitHub**: https://github.com/HeejeongH/HPEACE_prediction
- **저자**: 희정 (HeejeongH)

---

**최종 업데이트**: 2026-01-05  
**버전**: v0.3.0  
**상태**: 2-Class 예측 모델 구현 완료 ⭐
