# Version 2: Longitudinal Change Prediction (종단 변화 예측)

## 🎯 분석 목표

**Longitudinal Analysis**: 식습관 **변화** → 건강지표 **변화** 예측

```
입력 (Input):  방문1 식습관 → 방문2 식습관 (변화량 + 시간차)
모델 (Model):  시계열 딥러닝 (LSTM, Temporal Transformer 등)
출력 (Output): 건강 바이오마커 변화량 (예: 체중 -5kg)
```

## 🆕 Ver1과의 차이

### Ver1 (Cross-sectional)
```
질문: "이런 식습관 사람의 체중은?"
답변: "평균 87kg입니다"
```

### Ver2 (Longitudinal) ⭐ NEW
```
질문: "고지방을 5점→2점으로 줄이면 3개월 후 체중은?"
답변: "현재 87kg에서 약 82kg로 -5kg 감소 예상"
```

## 📊 데이터 구조 (재구성 필요)

### 기존 데이터 (Ver1)
```python
Row 1: [사람A_방문1] 식습관 → 건강지표
Row 2: [사람A_방문2] 식습관 → 건강지표
→ 독립 샘플
```

### 새로운 데이터 (Ver2)
```python
Pair 1: [사람A] 방문1→방문2
  입력: {
    '고지방_변화': 5→2 (변화: -3),
    '채소_변화': 2→5 (변화: +3),
    '시간차': 90일,
    '베이스라인_체중': 87kg,
    '베이스라인_BMI': 28.2
  }
  출력: {
    '체중_변화': -5kg,
    'BMI_변화': -1.8,
    '변화율': -5.7%
  }
→ Paired data (짝지은 데이터)
```

## 🔧 개발 계획

### Phase 1: 데이터 전처리 ⏳ 진행 예정
```python
# 1. 방문 쌍(Paired visits) 생성
paired_data = create_paired_visits(
    min_time_gap=30,    # 최소 30일 간격
    max_time_gap=365    # 최대 1년 간격
)

# 예상 샘플 수: 11,238명 × 평균 1.6쌍 ≈ 18,000 pairs
```

### Phase 2: 특성 공학 ⏳ 계획 중
```python
features = {
    # 식습관 변화량 (19개)
    '고지방_변화': after - before,
    '채소_변화': after - before,
    ...
    
    # 시간 관련
    '시간차_일수': days,
    '시간차_개월': months,
    
    # 베이스라인 (시작점)
    '시작_체중': before_weight,
    '시작_BMI': before_bmi,
    '시작_나이': age_at_visit1,
    
    # 변화 속도
    '고지방_변화율': (after-before) / before,
    
    # 복합 변화
    '총_위험식습관_변화': sum(risk_changes),
    '총_보호식습관_변화': sum(protective_changes)
}
```

### Phase 3: 모델 선정 ⏳ 계획 중

#### Option 1: LSTM (Long Short-Term Memory)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(128, input_shape=(timesteps, features)),
    Dense(64, activation='relu'),
    Dense(6)  # 6개 바이오마커 변화량
])
```

#### Option 2: Temporal Fusion Transformer
```python
# 시계열 + Attention 결합
# 장기 의존성 + 해석 가능성
```

#### Option 3: XGBoost for Change
```python
# 변화량을 직접 학습
# 빠르고 안정적
```

### Phase 4: 학습 및 평가 ⏳ 계획 중
```python
# 평가 지표
- Change R²: 변화 예측 정확도
- MAE: 평균 절대 오차 (kg, mmHg 등)
- Direction Accuracy: 변화 방향 정확도 (증가/감소/유지)
```

## 🎯 기대 효과

### 1. 인과관계 파악
```
✅ "식습관 개선 → 건강 개선" 직접 입증
✅ 개입(Intervention) 효과 정량화
```

### 2. 개인 맞춤 예측
```
✅ "당신이 이렇게 바꾸면 이렇게 됩니다"
✅ Individual-level change prediction
```

### 3. 시간 개념 포함
```
✅ 3개월 후, 6개월 후 예측 가능
✅ 변화 속도 고려
```

### 4. 실용적 활용
```
✅ 맞춤형 식단 계획
✅ 목표 기반 역산 (원하는 건강 → 필요한 식습관)
✅ 진행 상황 모니터링
```

## 📊 예상 성능

### 목표
```
체중 변화 예측:
- R² > 0.70 (변화량 기준)
- MAE < 2kg
- Direction Accuracy > 85%

혈압 변화 예측:
- R² > 0.50
- MAE < 5mmHg
- Direction Accuracy > 80%
```

### Ver1 대비 장점
```
Ver1: "현재 상태 추정" (R² 0.90)
Ver2: "미래 변화 예측" (R² 0.70 목표)

→ R²는 낮아지지만 실용성은 훨씬 높음!
```

## ✅ 현재 상태

```
✅ 구현 완료!

완료: 
- [x] 데이터 재구성 (data_preprocessing.py)
- [x] 특성 공학 (파생 특성 포함)
- [x] XGBoost 모델 구현
- [x] LSTM 모델 구현
- [x] 통합 실행 스크립트 (run_ver2.py)
- [ ] 학습 및 검증 (실행 필요)
- [ ] 결과 분석 (학습 후)

실행 가능: python run_ver2.py
```

## 🚀 실행 방법

### 1. 간편 실행 (추천)
```bash
cd ver2
python run_ver2.py
```

메뉴:
1. 데이터 전처리 (Paired Visits 생성)
2. XGBoost 모델 학습 (Baseline)
3. LSTM 모델 학습 (Advanced)
4. 전체 실행 (1→2→3)
5. 결과 비교 (XGBoost vs LSTM)

### 2. 개별 실행
```bash
# 데이터 전처리
cd ver2/src
python data_preprocessing.py

# XGBoost 학습
python xgboost_model.py

# LSTM 학습
python lstm_model.py
```

## 📋 개발 로드맵

### Week 1-2: 데이터 준비 ✅ 완료
- [x] Paired visits 데이터 생성
- [x] 변화량 계산
- [x] 데이터 탐색 분석(EDA)
- [x] 파생 특성 생성

### Week 3-4: 모델 개발 ✅ 완료
- [x] Baseline 모델 (XGBoost)
- [x] LSTM 모델
- [x] 통합 실행 스크립트
- [x] 모델 비교 기능

### Week 5-6: 평가 및 최적화 ⏳ 실행 필요
- [ ] 실제 데이터로 학습
- [ ] Cross-validation
- [ ] 하이퍼파라미터 튜닝
- [ ] 특성 중요도 분석

### Week 7-8: 문서화 및 배포 ⏳ 계획
- [ ] 분석 보고서 작성
- [ ] 사용자 가이드 업데이트
- [ ] Ver1 vs Ver2 비교 문서

## 🔍 연구 질문

Ver2가 답할 수 있는 질문들:

1. **변화 예측**
   - 식습관을 이렇게 바꾸면 건강이 얼마나 개선될까?

2. **최적 기간**
   - 효과를 보려면 얼마나 지속해야 할까?

3. **역방향 추론**
   - 체중을 5kg 감량하려면 어떤 식습관 변화가 필요할까?

4. **상호작용**
   - 여러 식습관을 동시에 바꾸면 시너지 효과가 있을까?

5. **개인차**
   - 같은 식습관 변화도 사람마다 효과가 다를까?

## 📚 참고 문헌

### Longitudinal Analysis 방법론
- Mixed Effects Models
- Growth Curve Modeling
- Difference-in-Differences
- Propensity Score Matching

### 시계열 딥러닝
- LSTM (Hochreiter & Schmidhuber, 1997)
- Temporal Fusion Transformer (Lim et al., 2021)
- N-BEATS (Oreshkin et al., 2020)

## 💡 아이디어 & 노트

```
1. 베이스라인 중요성
   - 시작점이 다르면 변화량도 다름
   - 예: 체중 100kg vs 60kg 사람의 -5kg는 의미가 다름

2. 비선형 변화
   - 초기에는 빠른 변화, 나중에는 느린 변화
   - Plateau 효과 고려 필요

3. 계절성
   - 여름/겨울 식습관 차이
   - 명절 효과

4. 컴플라이언스
   - 실제로 식습관을 바꿨는지?
   - 자기보고식 데이터의 한계
```

## 🤝 기여 방법

Ver2 개발에 참여하고 싶으시면:
- GitHub Issues: 아이디어 제안
- Pull Requests: 코드 기여
- Discussions: 방법론 토론

---

**작성일**: 2025-11-05  
**분석 유형**: Longitudinal (종단)  
**상태**: 🚧 개발 중  
**예상 완료**: TBD

---

**이전 버전**: [Ver1 - Cross-sectional Analysis](../ver1/)
