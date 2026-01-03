# HPEACE Prediction - MetS 예측 모델

대사증후군(Metabolic Syndrome) 변화 예측을 위한 딥러닝 및 머신러닝 모델

## 📊 프로젝트 개요

### 목표
연속 방문자의 식습관 변화를 활용하여 대사증후군(MetS) 및 건강 지표의 변화를 예측

### 데이터
- **전체 데이터**: 29,098건
- **분석 대상**: 11,238명 (2회 이상 방문자)
- **변수 수**: 65개 (피처 엔지니어링 후)
  - 식습관(FFQ): 19개
  - 인구학적: 4개
  - 생활습관: 3개
  - 바이오마커: 11개
  - 변화량(Delta): 가변
  - 상호작용: 6개
  - PCA: 10개

### 예측 타겟
`mets_transition`: 대사증후군 변화 (3-class)
- 0: 개선 (Improvement)
- 1: 유지 (Stable)
- 2: 악화 (Worsening)

## 🎯 현재 성능

### 최신 결과 (2026-01-03)
```
베이스라인 (CrossEntropy + SMOTE):
- F1 Score: 0.481

Optuna 최적화 후:
- F1 Score: 0.506 (+0.025)
- 최적 파라미터:
  * dropout_rate: 0.357
  * l1_lambda: 0.000366
  * l2_lambda: 0.000596
```

### 성능 개선 목표
- **단기**: F1 > 0.60 (Loss 함수 개선)
- **중기**: F1 > 0.65 (앙상블 강화)
- **장기**: F1 > 0.70 (데이터 보강 + 하이퍼파라미터 재최적화)

## 🏗️ 모델 아키텍처

### MultiDiseasePredictor (PyTorch)
```
입력 (75차원) → 7개 인코더 (각 8차원) → 합성 (56차원) → 질병 헤드 (3-class)

인코더 구조:
- 7개 그룹별 독립 인코더
  1. Diet (식습관): 19 → 8
  2. Demo (인구학): 4 → 8
  3. Life (생활습관): 3 → 8
  4. Bio (바이오마커): 11 → 8
  5. Delta (변화량): 6 → 8
  6. Interaction (상호작용): 6 → 8
  7. PCA (주성분): 10 → 8

각 인코더: Linear → BatchNorm → ReLU → Dropout(0.357)
질병 헤드: Linear(56→16) → BatchNorm → ReLU → Dropout → Linear(16→3)
```

### 학습 설정
- **Optimizer**: AdamW (lr=0.0001, weight_decay=6e-4)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Early Stopping**: patience=15, min_delta=0.001
- **정규화**: L1 + L2 regularization
- **최대 Epoch**: 100
- **배치 크기**: 64

## 📂 프로젝트 구조

```
webapp/
├── data/
│   └── total_again.xlsx              # 원본 데이터 (5.7MB)
├── src/
│   ├── data_import.py                # 데이터 전처리 (DataPreprocessor)
│   ├── feature_engineering.py        # 피처 엔지니어링 (PCA, Interaction)
│   ├── MetS_prediction_model.py      # 딥러닝 모델 정의
│   ├── train_eval_function.py        # 학습 및 평가 함수
│   ├── loss_functions.py             # Loss 함수 (CE, Focal, Weighted)
│   ├── resampling.py                 # SMOTE, 불균형 처리
│   ├── diet_recommend.py             # 식이 추천 시스템
│   ├── SHAP.py                       # 해석 가능성 분석
│   ├── utils.py                      # 유틸리티 함수
│   ├── main.ipynb                    # 메인 실행 노트북
│   ├── test_improved_loss.py         # 성능 개선 테스트 스크립트 (NEW!)
│   └── quick_performance_boost.py    # 성능 개선 가이드 (NEW!)
├── result/
│   ├── best_combo/
│   │   └── comprehensive_combo_cache.pkl
│   ├── figures/
│   │   ├── figure1_optimization.png  # 최적화 과정
│   │   └── figure2_roc_pr_confusion.png  # ROC/PR/Confusion
│   └── engineered_analysis/
│       └── feature_importance_analysis.png
├── requirements.txt                  # Python 패키지 의존성
├── .gitignore                        # Git 무시 파일 목록
└── README.md                         # 이 파일
```

## 🚀 사용 방법

### 1. 환경 설정
```bash
# 저장소 클론
git clone https://github.com/HeejeongH/HPEACE_prediction.git
cd HPEACE_prediction

# 필수 패키지 설치
pip install -r requirements.txt
```

### 2. 기본 실행 (Jupyter)
```bash
cd src
jupyter notebook main.ipynb

# 또는 JupyterLab
jupyter lab main.ipynb
```

### 3. 성능 개선 테스트 (NEW!)
```bash
cd src

# 가이드 확인
python quick_performance_boost.py

# 실제 테스트 실행
python test_improved_loss.py
```

## 🔬 성능 개선 전략

### Phase 1: Loss 함수 개선 ✅ (구현 완료)
```python
# loss_functions.py에 새로 추가된 함수들
- calculate_improved_class_weights()  # 개선된 클래스 가중치 계산
- improved_loss_methods_configs()     # 8가지 Loss 함수 설정

테스트할 Loss 함수:
1. CrossEntropy (기준)
2. FocalLoss (gamma=1.5, 2.0, 2.5)
3. WeightedCE (inverse_freq, effective_num, balanced)
4. FocalLoss + Weighted (조합)
```

### Phase 2: 앙상블 강화 (예정)
- Stacking (RF + GB + LR + XGBoost)
- Soft Voting 개선
- 메타 러너 최적화

### Phase 3: 데이터 보강 (예정)
- SMOTE + Tomek Links (Hybrid)
- ADASYN 테스트
- 누락 변수 임퓨테이션 (GLUCOSE, HbA1c, HDL)

### Phase 4: 하이퍼파라미터 재최적화 (예정)
- Optuna trials: 30 → 100
- 탐색 공간 확장
- K-Fold Cross Validation

## 📈 실험 결과 (업데이트 예정)

### Loss 함수별 성능 비교
| Loss 함수 | Accuracy | F1 Score | ROC AUC | 개선폭 |
|-----------|----------|----------|---------|--------|
| CrossEntropy (기준) | - | 0.506 | - | - |
| FocalLoss_gamma2.0 | - | TBD | - | - |
| WeightedCE_inverse | - | TBD | - | - |
| FocalLoss_weighted | - | TBD | - | - |

**TBD**: `test_improved_loss.py` 실행 후 업데이트 예정

## 🔧 개발 현황

### 완료된 작업 ✅
- [x] 데이터 전처리 파이프라인
- [x] 피처 엔지니어링 (PCA, 상호작용)
- [x] MultiDiseasePredictor 모델 구현
- [x] SMOTE 불균형 처리
- [x] Optuna 하이퍼파라미터 최적화
- [x] 기본 Loss 함수 (CE, Focal, Weighted)
- [x] ROC/PR/Confusion Matrix 시각화
- [x] **개선된 Loss 함수 구현** (2026-01-03)
- [x] **성능 테스트 스크립트 작성** (2026-01-03)

### 진행 중 🔄
- [ ] Loss 함수별 성능 비교 실험 실행
- [ ] 결과 분석 및 최적 Loss 선택

### 예정 작업 📋
- [ ] 앙상블 기법 강화 (Stacking)
- [ ] 누락 변수 임퓨테이션
- [ ] 하이퍼파라미터 재최적화 (100 trials)
- [ ] 모델 체크포인트 저장/로드 로직
- [ ] 교차검증 추가
- [ ] 실험 추적 시스템 (MLflow/Wandb)
- [ ] 웹 데모 개발 (Streamlit/Gradio)

## 📝 주의사항

### 누락 변수
다음 변수는 데이터에 포함되어 있지 않아 예측에 사용되지 않음:
- GLUCOSE (공복혈당)
- HBA1C (당화혈색소)
- HDL CHOL (고밀도 콜레스테롤)

→ **Phase 3**에서 임퓨테이션 또는 재수집 예정

### 데이터 불균형
클래스 분포가 불균형하여 SMOTE 적용:
- 개선(0): 소수
- 유지(1): 다수
- 악화(2): 소수

→ 개선된 클래스 가중치로 추가 보정 (Phase 1)

## 🔗 링크

- **GitHub**: https://github.com/HeejeongH/HPEACE_prediction
- **데이터**: `/home/user/webapp/data/total_again.xlsx`
- **결과**: `/home/user/webapp/result/`

## 👥 기여

- 희정 (HeejeongH) - 프로젝트 리드, 모델 개발

## 📄 라이선스

이 프로젝트는 연구 목적으로 개발되었습니다.

---

**최종 업데이트**: 2026-01-03  
**버전**: v0.2.0 (Performance Boost)  
**상태**: 🔄 성능 개선 진행 중
