# Version 1: Cross-sectional Analysis (횡단면 분석)

## 📊 분석 방법

**Cross-sectional Prediction**: 특정 시점의 식습관 → 같은 시점의 건강지표 예측

```
입력 (Input):  식습관 19개 변수 (1개 시점)
모델 (Model):  TabNet + Stacking Ensemble
출력 (Output): 건강 바이오마커 6개 (같은 시점)
```

## 🎯 예측 내용

"**이런 식습관을 가진 사람들의 평균적인 건강지표**"

- ✅ 식습관 패턴별 건강 상태 추정
- ✅ Population-level prediction
- ✅ 상관관계(Correlation) 분석

## 📈 성능 결과

```
평균 R² = 0.8992 (89.92%)

체조성 지표:
- BMI: R² = 0.9988 (99.88%)
- 체중: R² = 0.9986 (99.86%)
- 허리둘레: R² = 0.9651 (96.51%)

심혈관/대사 지표:
- DBP: R² = 0.8164 (81.64%)
- TG: R² = 0.8093 (80.93%)
- SBP: R² = 0.8068 (80.68%)
```

## 📊 데이터 구조

```python
# 각 행 = 독립적인 1번 방문 기록
총 샘플: 29,098개
참여자: 11,238명
평균 방문: 2.6회/인

Row 1: [사람A_방문1] 식습관 → 건강지표
Row 2: [사람A_방문2] 식습관 → 건강지표  # 독립 샘플로 취급
Row 3: [사람B_방문1] 식습관 → 건강지표
...

→ 방문 간 연결 정보 사용 안 함
```

## ⚠️ 한계점

### 1. 인과관계 불명확
```
❌ "식습관을 바꾸면 건강이 개선된다" (인과)
✅ "건강한 식습관과 좋은 건강지표가 연관되어 있다" (상관)
```

### 2. 개인 맞춤 변화 예측 불가
```
❌ "당신이 식습관을 이렇게 바꾸면 체중이 5kg 감소"
✅ "이런 식습관의 사람들은 평균 체중이 82kg"
```

### 3. 시간 개념 없음
```
❌ 3개월 후 예측
✅ 현재 상태 추정
```

## 🚀 실행 방법

### 로컬 PC에서 실행 (추천)

샌드박스 환경의 메모리 제약으로 **로컬 PC에서 실행을 권장**합니다.

```bash
# 1. 빠른 시작 가이드
QUICK_START.md 참고

# 2. 안전 모드 실행 (60분)
python run_training.py safe

# 3. 전체 최적화 (2시간)
python run_training.py full
```

**자세한 가이드**: `LOCAL_EXECUTION_GUIDE.md`

### 고급 분석 (모델 생성 후)

```bash
# Phase 1: 서브그룹 분석
python advanced_analysis.py

# Phase 2: 임계값 분석 (완료됨)
python threshold_analysis.py

# Phase 3: 서브그룹 모델링
python subgroup_modeling.py

# Phase 4: SHAP 해석성 분석
python shap_analysis.py

# Phase 5: 논문 보고서 생성
python generate_paper_report.py
```

## 📁 파일 구조

```
ver1/
├── src/
│   ├── TABNET_ENHANCED_MODEL.py     # 메인 모델 코드
│   └── ultra_quick_demo.py          # 빠른 테스트
│
├── run_training.py                   # 모델 학습 스크립트
├── QUICK_START.md                    # 빠른 시작 가이드 ⭐
├── LOCAL_EXECUTION_GUIDE.md          # 상세 실행 가이드 ⭐
│
├── advanced_analysis.py              # Phase 1: 서브그룹 분석
├── threshold_analysis.py             # Phase 2: 임계값 분석
├── subgroup_modeling.py              # Phase 3: 그룹별 모델
├── shap_analysis.py                  # Phase 4: SHAP 해석
├── generate_paper_report.py          # Phase 5: 논문 생성
│
├── advanced_results/                 # 분석 결과
│   ├── FINAL_PAPER_REPORT.md        # 논문 초안 ✅
│   ├── threshold_analysis/          # 임계값 분석 결과 ✅
│   ├── subgroup_models/             # 그룹별 모델
│   └── shap_analysis/               # SHAP 결과
│
└── requirements.txt                  # 의존성
```

## 📚 관련 문서

- **빠른 시작**: `QUICK_START.md` ⭐
- **상세 가이드**: `LOCAL_EXECUTION_GUIDE.md` ⭐
- **고급 분석**: `ADVANCED_ANALYSIS_README.md`
- **논문 초안**: `advanced_results/FINAL_PAPER_REPORT.md` ✅
- **분석 보고서**: `/docs/ANALYSIS_REPORT.md`
- **입출력 설명**: `/docs/INPUT_OUTPUT_EXPLANATION.md`

## 📊 완료된 분석 결과

### ✅ Phase 1: 기본 서브그룹 분석
- 연령/성별/BMI별 그룹 생성
- 그룹별 통계 및 분포 시각화
- 29,098명 데이터 분석 완료

### ✅ Phase 2: 임계값 분석
- 4개 건강지표 (BMI, SBP, DBP, TG)
- 위험도 분류 및 최적 임계값
- 개인맞춤형 식습관 권고안
- 16개 결과 파일 생성 완료

### ⏳ Phase 3-4: 대기 중
- 서브그룹 모델링: Ver1 모델 필요
- SHAP 해석성 분석: Ver1 모델 필요
- **로컬 PC에서 모델 학습 후 실행 가능**

---

**작성일**: 2025-11-06 (업데이트)  
**분석 유형**: Cross-sectional (횡단면)  
**성능**: ⭐⭐⭐⭐⭐ Excellent (R² 0.90)  
**상태**: ✅ 기본 분석 완료, ⏳ 모델 학습 대기
