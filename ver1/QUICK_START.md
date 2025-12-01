# ⚡ Ver1 모델 빠른 실행 가이드

## 🎯 목표
로컬 PC에서 Ver1 TabNet 모델을 학습하여 6개 건강지표 예측 모델 생성

---

## 📦 1단계: 준비 (5분)

```bash
# 1. 저장소 클론
git clone https://github.com/HeejeongH/HPEACE_prediction.git
cd HPEACE_prediction/ver1

# 2. 라이브러리 설치
pip install -r requirements.txt

# 또는 최소 설치:
pip install pandas numpy scikit-learn xgboost lightgbm catboost torch pytorch-tabnet matplotlib seaborn openpyxl tqdm
```

---

## 🚀 2단계: 실행 (60분)

### 추천 명령어 (안전 모드):
```bash
python run_training.py safe
```

**또는:**
```bash
python run_training.py 1
```

### 실행 중 화면 예시:
```
================================================================================
🎯 타겟: 체중
================================================================================
   📊 사용 가능한 샘플: 29,098개
   🔧 TabNet 통합 Stacking Ensemble 구성 중...
   
Early stopping occurred at epoch 124 with best_epoch = 74

   ✅ 체중 모델 저장: result/models/체중_model.pkl
   📈 Test R²: 0.9012, RMSE: 2.85

================================================================================
🎯 타겟: 체질량지수
================================================================================
   ...
```

---

## 📤 3단계: 결과 업로드

### 생성된 파일들:
```
ver1/result/
├── models/              # 6개 .pkl 파일 (중요!)
├── performance/         # 성능 지표
├── predictions/         # 예측 결과
└── feature_importance/  # 특성 중요도
```

### 업로드 방법:

**Option A: 압축 후 Google Drive**
```bash
cd ver1
tar -czf ver1_results.tar.gz result/
# ver1_results.tar.gz를 Google Drive에 업로드
```

**Option B: GitHub에 직접 업로드**
```bash
git add result/
git commit -m "Add Ver1 trained models"
git push origin main
```

---

## 🎯 4단계: 결과 공유

다음 중 하나로 공유해주세요:

1. **Google Drive 링크** (추천)
   - 공유 설정: "링크가 있는 모든 사용자"

2. **GitHub Release**
   - https://github.com/HeejeongH/HPEACE_prediction/releases

3. **이메일/메신저**
   - `ver1_results.tar.gz` 파일 전송

---

## ✅ 체크리스트

- [ ] Python 3.8+ 설치
- [ ] 라이브러리 설치 완료
- [ ] `python run_training.py safe` 실행
- [ ] 약 60분 대기
- [ ] `result/models/` 폴더에 6개 파일 확인
- [ ] 압축 및 업로드 완료

---

## ❓ 문제 발생 시

### 메모리 부족:
```bash
python run_training.py tabnet  # Stacking 없이 TabNet만 사용
```

### 시간 단축 (테스트용):
```bash
python run_training.py quick   # Optuna 5회만
```

### 자세한 가이드:
`LOCAL_EXECUTION_GUIDE.md` 참고

---

## 📊 예상 결과

| 지표 | 예상 R² | 실행 순서 |
|------|---------|----------|
| 체중 | 0.90+ | 1번째 (10분) |
| 체질량지수 | 0.92+ | 2번째 (10분) |
| 허리둘레 | 0.88+ | 3번째 (10분) |
| SBP | 0.82+ | 4번째 (10분) |
| DBP | 0.80+ | 5번째 (10분) |
| TG | 0.78+ | 6번째 (10분) |

**총 예상 시간: 60분**

---

## 🔗 관련 링크

- 저장소: https://github.com/HeejeongH/HPEACE_prediction
- 자세한 가이드: `LOCAL_EXECUTION_GUIDE.md`
- 문의: GitHub Issues

---

**실행 성공하시길 바랍니다! 💪**
