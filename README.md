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

```
베이스라인: F1 = 0.481 (CrossEntropy + SMOTE)
최적화 후: F1 = 0.506 (Optuna 30 trials)
목표 성능: F1 > 0.60
```

**최적 하이퍼파라미터:**
- dropout_rate: 0.357
- l1_lambda: 0.000366
- l2_lambda: 0.000596

## 🏗️ 모델 아키텍처

### MultiDiseasePredictor (PyTorch)
```
입력 75차원 → 7개 인코더 (각 8차원) → 합성 56차원 → 출력 3-class

인코더 그룹:
1. Diet (19) → 2. Demo (4) → 3. Life (3) → 4. Bio (11)
5. Delta (6) → 6. Interaction (6) → 7. PCA (10)

학습:
- Optimizer: AdamW (lr=0.0001, wd=6e-4)
- Scheduler: ReduceLROnPlateau
- Early Stopping: patience=15
- Loss: CrossEntropy / FocalLoss / WeightedCE
```

## 📂 프로젝트 구조

```
webapp/
├── src/
│   ├── data_import.py              # 데이터 전처리
│   ├── feature_engineering.py      # 피처 엔지니어링 (PCA, Interaction)
│   ├── MetS_prediction_model.py    # 딥러닝 모델
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
```bash
cd src
jupyter notebook main.ipynb
```

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

**최종 업데이트**: 2026-01-03  
**버전**: v0.2.0  
**상태**: Loss 함수 개선 완료, 실험 진행 예정
