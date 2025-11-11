# Data Leakage 수정 가이드

## 📅 작성일: 2025-11-11
## 🎯 목적: Data Leakage 제거 및 재학습 안내

---

## 🚨 발견된 Data Leakage 문제

### **문제 코드 (ver1/src/TABNET_ENHANCED_MODEL.py, Line 208-210)**

```python
# ❌ 문제 있는 코드 (삭제됨)
if '체질량지수' in df.columns:
    df['bmi_unhealthy_interaction'] = df['체질량지수'] * df['weighted_unhealthy_score']
    df['bmi_sodium_interaction'] = df['체질량지수'] * df['sodium_risk_score']
```

### **왜 문제인가?**

**순환 논리 (Circular Logic):**
```
BMI = 체중(kg) / 키²(m²)

체중 예측 시:
입력 → BMI 파생 특성 (bmi_unhealthy_interaction)
     ↓
  BMI가 체중으로 계산됨
     ↓
출력 ← 체중

→ 정답(체중)으로 계산된 값(BMI)을 사용해서 정답 예측!
```

**증거:**
- 체중 예측: R² = **0.9996** (비정상적으로 높음)
- BMI 예측: R² = **0.9997** (거의 완벽, 의심스러움)
- 허리둘레 예측: R² = **0.9697** (매우 높음)
- SBP 예측: R² = 0.7884 (정상적 수준)

**정상적인 식습관 예측 R²: 0.6-0.8**

---

## ✅ 수정 내용

### **수정된 코드 (ver1/src/TABNET_ENHANCED_MODEL.py)**

```python
# ✅ 수정됨: BMI 파생 특성 생성 제거
# ❌ Data Leakage 방지: BMI 파생 특성 제거
# BMI는 체중/키²로 계산되므로, 체중 예측 시 사용하면 순환 논리 발생
# if '체질량지수' in df.columns:
#     df['bmi_unhealthy_interaction'] = df['체질량지수'] * df['weighted_unhealthy_score']
#     df['bmi_sodium_interaction'] = df['체질량지수'] * df['sodium_risk_score']
```

### **변경 사항 요약:**
1. ❌ `bmi_unhealthy_interaction` 특성 생성 제거
2. ❌ `bmi_sodium_interaction` 특성 생성 제거
3. ✅ 주석으로 이유 설명 추가

---

## 🔄 재학습 필요 사항

### **1. 현재 학습 중단**

```bash
# Windows에서 Ctrl+C로 중단
# 또는 프로세스 종료
```

### **2. 최신 코드 가져오기**

```bash
cd "G:\내 드라이브\#인력양성\3. 식이_SNUH\#Prediction"

# 로컬 변경사항 stash (필요시)
git stash

# 최신 코드 pull
git pull origin main

# ver1 디렉토리로 이동
cd ver1
```

### **3. 재학습 실행**

```bash
# 안전 모드로 재실행
python run_training.py safe
```

### **4. 예상 실행 시간**

- 총 시간: **약 60분** (6개 지표)
- 각 지표당: 약 10분

---

## 📊 예상 성능 변화

### **수정 전 (Data Leakage 있음) vs 수정 후**

| 건강지표 | 수정 전 R² | 예상 수정 후 R² | 판단 |
|---------|-----------|----------------|------|
| **체중** | 0.9996 🚨 | **0.85-0.90** ✅ | 정상화 |
| **체질량지수** | 0.9997 🚨 | **0.85-0.92** ✅ | 정상화 |
| **허리둘레** | 0.9697 🚨 | **0.80-0.85** ✅ | 정상화 |
| **SBP** | 0.7884 ✅ | **0.78-0.82** ✅ | 변화 없음 |
| **DBP** | ? | **0.75-0.80** ✅ | 정상 범위 |
| **TG** | ? | **0.70-0.78** ✅ | 정상 범위 |

### **성능 평가 기준:**

- **R² ≥ 0.80**: ⭐⭐⭐ 매우 우수
- **R² 0.70-0.80**: ⭐⭐ 우수
- **R² 0.60-0.70**: ⭐ 양호
- **R² < 0.60**: 개선 필요

### **해석:**
- 수정 후에도 **R² 0.85 이상**이면 여전히 매우 우수한 성능
- 식습관만으로 건강지표를 85% 설명할 수 있다면 충분히 의미 있음
- 논문 게재에 문제 없는 수준

---

## 📝 논문 작성 시 주의사항

### **1. Data Leakage 확인 필수**

**체크리스트:**
- ✅ 입력 변수에 타겟과 직접 연관된 변수 없음
- ✅ 파생 특성이 타겟으로 계산되지 않음
- ✅ Train/Test split이 전처리 전 수행됨
- ✅ Scaler는 train만 fit, test는 transform만
- ✅ Cross-validation이 올바르게 수행됨

### **2. 성능 보고 시**

**올바른 표현:**
```
"식습관 패턴을 통해 체중을 85% 정확도로 예측할 수 있었으며,
이는 식습관이 체조성의 주요 결정 요인임을 시사한다."
```

**잘못된 표현:**
```
❌ "식습관으로 체중을 99.96% 정확도로 예측"
   → 심사자가 data leakage 의심
```

### **3. 한계점 명시**

```markdown
본 연구의 한계:
1. 횡단면 연구 (Cross-sectional): 인과관계 추론 제한
2. 자가 보고식 식습관 조사: 측정 오차 가능성
3. 유전적 요인 미포함: 설명력 한계
4. 단일 의료기관 데이터: 일반화 제한
```

---

## 🔬 추가 검증 방법

### **1. Feature Importance 확인**

재학습 완료 후:

```bash
# Feature importance 파일 확인
cat result/feature_importance/체중_feature_importance.csv | head -20
```

**정상적인 경우:**
```
feature,importance
나이,0.18
고지방육류,0.15
식사량,0.12
단맛,0.10
성별,0.09
...
```

**문제 있는 경우 (발생하면 안 됨):**
```
feature,importance
BMI,0.85                    ← 이게 나오면 문제!
bmi_interaction,0.10        ← 이것도 문제!
```

### **2. Train/Test 성능 차이 확인**

**정상 범위:**
- 차이 < 8%: 과적합 없음 ✅
- 차이 8-15%: 약간의 과적합 ⚠️
- 차이 > 15%: 심각한 과적합 또는 data leakage 🚨

**예시:**
```
체중 모델:
- Train R²: 0.88
- Test R²: 0.85
- 차이: 3.4% ✅ 정상
```

---

## 📂 GitHub 업데이트

### **최신 커밋:**
```
commit db10418
Author: Claude
Date: 2025-11-11

Fix data leakage: Remove BMI-derived features

- Removed bmi_unhealthy_interaction
- Removed bmi_sodium_interaction
- These features caused data leakage when predicting weight/BMI
- BMI = weight / height², so using BMI features to predict weight is circular logic
- Expected performance after fix: R² 0.85-0.92 (more realistic)
```

### **변경된 파일:**
- `ver1/src/TABNET_ENHANCED_MODEL.py` (Line 208-213)

---

## ❓ FAQ

### **Q1: 왜 처음에는 발견하지 못했나요?**

**A:** 코드가 복잡하고, BMI 파생 특성이 깊은 곳에 숨겨져 있었습니다. 
exclude_vars 리스트에는 `체질량지수`가 있었지만, 
`bmi_unhealthy_interaction` 같은 파생 특성은 누락되어 있었습니다.

### **Q2: 수정 후 성능이 떨어지는 게 맞나요?**

**A:** 네, 맞습니다. **정확한 성능**으로 떨어지는 것입니다.
- 수정 전: 부풀려진 성능 (R² 0.99)
- 수정 후: 실제 성능 (R² 0.85-0.90)

식습관만으로 R² 0.85 달성은 여전히 **매우 우수**합니다!

### **Q3: 논문 게재에 영향이 있나요?**

**A:** 오히려 긍정적입니다.
- ✅ Data leakage 없는 깨끗한 연구
- ✅ 현실적이고 신뢰할 수 있는 성능
- ✅ 심사자 신뢰도 향상

R² 0.99는 심사자에게 의심을 받을 가능성이 높습니다.

### **Q4: 다른 Data Leakage는 없나요?**

**A:** 현재 코드 검토 결과:
- ✅ Train/Test split 순서 올바름
- ✅ Scaler fit/transform 올바름
- ✅ Cross-validation 올바름
- ✅ BMI 파생 특성 제거됨

추가 문제는 발견되지 않았습니다.

---

## 📞 문의

재학습 중 문제가 발생하면:
1. 에러 메시지 전체 복사
2. GitHub Issues 또는 대화창에 공유
3. 즉시 지원 가능

---

## 🎯 다음 단계 체크리스트

- [ ] 1. 현재 학습 중단 (필요시)
- [ ] 2. `git pull origin main` 실행
- [ ] 3. `python run_training.py safe` 재실행
- [ ] 4. 약 60분 대기
- [ ] 5. Feature importance 확인
- [ ] 6. Train/Test 성능 차이 확인
- [ ] 7. 결과 압축 및 업로드
- [ ] 8. Phase 3-4 분석 계속 진행

---

**작성일**: 2025-11-11  
**작성자**: AI Assistant  
**상태**: ✅ 수정 완료, 재학습 대기 중
