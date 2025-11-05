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

```bash
# 안전 모드 (30분)
python run_training.py safe

# 전체 최적화 (17시간)
python run_training.py full
```

## 📁 파일 구조

```
ver1/
├── src/
│   ├── TABNET_ENHANCED_MODEL.py    # 메인 모델 코드
│   └── ultra_quick_demo.py          # 빠른 테스트
├── run_training.py                   # 실행 스크립트
├── train.bat                         # Windows 실행
├── check_cuda.py                     # GPU 확인
└── requirements.txt                  # 의존성
```

## 📚 관련 문서

- **분석 보고서**: `/docs/ANALYSIS_REPORT.md`
- **입출력 설명**: `/docs/INPUT_OUTPUT_EXPLANATION.md`

## 🔄 Ver2로의 발전

Ver1의 한계를 극복하기 위해 **Ver2 (Longitudinal Analysis)**가 개발 중입니다.

→ 식습관 **변화**가 건강지표 **변화**에 미치는 영향 분석

---

**작성일**: 2025-11-04  
**분석 유형**: Cross-sectional (횡단면)  
**성능**: ⭐⭐⭐⭐⭐ Excellent  
**상태**: ✅ 완료
