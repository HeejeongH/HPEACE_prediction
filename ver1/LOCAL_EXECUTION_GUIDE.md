# Ver1 모델 로컬 실행 가이드

## 📋 개요

샌드박스 환경에서는 메모리 제약으로 Ver1 TabNet 모델 학습이 불가능합니다.
로컬 컴퓨터에서 실행하여 모델을 생성하고 결과를 업로드해주세요.

---

## 🔧 사전 준비

### 1. 저장소 클론
```bash
git clone https://github.com/HeejeongH/HPEACE_prediction.git
cd HPEACE_prediction/ver1
```

### 2. 필요한 라이브러리 설치
```bash
pip install pandas numpy scikit-learn
pip install xgboost lightgbm catboost
pip install torch pytorch-tabnet
pip install matplotlib seaborn
pip install openpyxl tqdm
pip install shap  # SHAP 분석용 (선택사항)
```

**또는 requirements.txt 사용:**
```bash
pip install -r requirements.txt
```

### 3. 데이터 확인
- 데이터 위치: `../data/total_again.xlsx`
- 파일 크기: 약 7.7MB
- 샘플 수: 29,098개

---

## 🚀 실행 방법

### Option 1: 안전 모드 (추천) ⭐

**가장 안정적이고 빠른 방법**

```bash
python run_training.py safe
```

또는:
```bash
python run_training.py 1
```

**특징:**
- TabNet + Stacking Ensemble 사용
- Optuna 최적화 없음 (안정성 우선)
- 예상 시간: 30-60분
- 메모리 사용: 2-4GB

---

### Option 2: 전체 최적화 모드

**최고 성능을 원하는 경우**

```bash
python run_training.py full
```

**특징:**
- TabNet + Stacking + Optuna 최적화
- Optuna 20회 시도
- 예상 시간: 1-2시간
- 메모리 사용: 4-8GB
- ⚠️ Optuna segfault 발생 가능

---

### Option 3: 빠른 테스트 모드

**동작 확인용**

```bash
python run_training.py quick
```

**특징:**
- Optuna 5회만 시도
- 예상 시간: 20-30분
- 성능은 안전 모드와 유사

---

## 📊 출력 결과

학습이 완료되면 다음 디렉토리에 결과가 저장됩니다:

```
ver1/result/
├── models/                          # 학습된 모델 파일
│   ├── 체중_model.pkl
│   ├── 체질량지수_model.pkl
│   ├── 허리둘레(WAIST)_model.pkl
│   ├── SBP_model.pkl
│   ├── DBP_model.pkl
│   └── TG_model.pkl
│
├── predictions/                     # 예측 결과
│   ├── 체중_predictions.csv
│   ├── 체질량지수_predictions.csv
│   └── ...
│
├── feature_importance/              # 특성 중요도
│   ├── 체중_feature_importance.csv
│   └── ...
│
└── performance/                     # 성능 지표
    └── overall_performance.csv      # R², RMSE, MAE 등
```

---

## 📤 결과 업로드 방법

### 1. 결과 압축
```bash
cd ver1
tar -czf ver1_results.tar.gz result/
```

### 2. GitHub에 업로드

**방법 A: Git LFS 사용 (대용량 파일)**
```bash
# Git LFS 설치 (처음만)
git lfs install
git lfs track "*.pkl"
git lfs track "*.tar.gz"

# 커밋 및 푸시
git add result/
git commit -m "Add Ver1 trained models and results"
git push origin main
```

**방법 B: 압축 파일을 GitHub Release에 업로드**
1. GitHub 저장소로 이동
2. "Releases" → "Create a new release"
3. `ver1_results.tar.gz` 파일 첨부
4. 릴리즈 생성

**방법 C: Google Drive 링크 공유**
1. `ver1_results.tar.gz`를 Google Drive에 업로드
2. 공유 링크를 이슈에 코멘트로 남기기

---

## 🔍 실행 중 확인사항

### 진행 상황 확인
```python
# 콘솔 출력 예시:
================================================================================
🎯 타겟: 체중
================================================================================
   📊 사용 가능한 샘플: 29,098개
   📊 사용 특성 수: 100개
   📊 이상치 제거 후: 84,723개
   📊 선택된 특성: 50개

   🔧 TabNet 통합 Stacking Ensemble 구성 중...
      🧠 TabNet 딥러닝 모델 학습 중...
      
Early stopping occurred at epoch 124 with best_epoch = 74 and best_val_0_rmse = 0.29674

      ✅ TabNet 학습 완료
      🎯 다른 Base Models 학습 중...
         ▸ XGBoost... 완료
         ▸ LightGBM... 완료
         ▸ CatBoost... 완료
         ▸ RandomForest... 완료
      
      ✅ Stacking Ensemble 완료
   
   📈 성능:
      Train R²: 0.9634
      Test R²: 0.9012
      RMSE: 2.85

================================================================================
✅ 체중 모델 저장: result/models/체중_model.pkl
================================================================================
```

### 예상 로그 타임라인

| 시간 | 단계 | 설명 |
|------|------|------|
| 0-2분 | 데이터 로드 | Excel 파일 읽기 및 전처리 |
| 2-5분 | 특성 생성 | EWMA 및 파생 특성 생성 |
| 5-60분 | 모델 학습 | 6개 지표 × 각 5-10분 |
| 60분+ | 결과 저장 | 모델 및 평가 결과 저장 |

---

## ❌ 문제 해결

### 문제 1: 메모리 부족 (MemoryError)
```bash
# 해결: TabNet만 사용 (Stacking 없음)
python run_training.py tabnet
```

### 문제 2: Optuna segfault
```bash
# 해결: 안전 모드 사용 (Optuna 비활성화)
python run_training.py safe
```

### 문제 3: CUDA/GPU 에러
```python
# src/TABNET_ENHANCED_MODEL.py 에서 수정:
device_name = 'cpu'  # 'cuda' 대신 'cpu' 사용
```

### 문제 4: 라이브러리 설치 실패
```bash
# 최소 요구사항만 설치
pip install pandas numpy scikit-learn xgboost lightgbm torch pytorch-tabnet
```

---

## 📊 예상 성능

안전 모드로 실행 시 예상 성능:

| 건강지표 | R² | RMSE | MAE |
|---------|-----|------|-----|
| 체중 | 0.90+ | 2.5-3.0 | 1.8-2.2 |
| 체질량지수 | 0.92+ | 0.8-1.0 | 0.6-0.8 |
| 허리둘레 | 0.88+ | 4.0-5.0 | 3.0-4.0 |
| SBP | 0.82+ | 8.0-10.0 | 6.0-8.0 |
| DBP | 0.80+ | 5.0-6.0 | 4.0-5.0 |
| TG | 0.78+ | 40-50 | 30-40 |

---

## 🎯 다음 단계 (모델 생성 후)

모델이 생성되면 다음 분석을 진행할 수 있습니다:

### 1. 서브그룹 모델링 (Phase 3)
```bash
python subgroup_modeling.py
```
- 예상 시간: 30-60분
- 출력: `advanced_results/subgroup_models/`

### 2. SHAP 해석성 분석 (Phase 4)
```bash
python shap_analysis.py
```
- 예상 시간: 30-60분
- 출력: `advanced_results/shap_analysis/`

### 3. 논문 보고서 업데이트
```bash
python generate_paper_report.py
```
- 실제 성능 지표로 보고서 업데이트

---

## 💬 문의 및 이슈

문제가 발생하면:
1. 에러 메시지 전체 캡처
2. `python --version` 및 `pip list` 출력
3. GitHub Issues에 보고

---

## ✅ 체크리스트

실행 전 확인사항:

- [ ] Python 3.8+ 설치됨
- [ ] 필요한 라이브러리 모두 설치됨
- [ ] `../data/total_again.xlsx` 파일 존재
- [ ] 최소 4GB 메모리 여유 있음
- [ ] 60분+ 실행 시간 확보

실행 후 확인사항:

- [ ] `result/models/` 폴더에 6개 `.pkl` 파일 생성
- [ ] `result/performance/overall_performance.csv` 존재
- [ ] 콘솔에 "✅ 모든 모델 학습 완료" 메시지 출력
- [ ] 압축 파일 생성 및 업로드 완료

---

**화이팅! 💪**

문제없이 실행되면 약 1시간 후 완료될 것입니다.
결과가 나오면 GitHub 또는 Drive에 업로드해주세요!
