## 📊 Ver1 Advanced Analysis - 논문용 심층 분석

**Ver1 결과를 논문 수준으로 발전시키기 위한 고급 분석 패키지**

---

## 🎯 분석 목표

Ver1의 우수한 성능(R² 0.90)을 바탕으로:
1. **세부 그룹 분석**: 나이/성별/BMI별 맞춤형 모델
2. **SHAP 해석력**: 특성 중요도 및 상호작용 분석
3. **임계값 도출**: 건강 위험 기준점 및 권고안
4. **논문 작성**: 통합 보고서 자동 생성

---

## 📁 파일 구조

```
ver1/
├── advanced_analysis.py          # Phase 1: 기초 분석
├── subgroup_modeling.py          # Phase 2: 세부 그룹 모델
├── shap_analysis.py              # Phase 3: SHAP 해석력
├── threshold_analysis.py         # Phase 4: 임계값 분석
├── generate_paper_report.py      # Phase 5: 보고서 생성
├── RUN_ALL_ANALYSIS.py          # 전체 실행 스크립트
└── advanced_results/            # 결과 저장 폴더
    ├── FINAL_PAPER_REPORT.md    # 📄 최종 논문 보고서
    ├── subgroup_models/         # 세부 그룹 모델
    ├── shap_analysis/           # SHAP 결과
    └── threshold_analysis/      # 임계값 분석
```

---

## 🚀 빠른 시작

### Option 1: 전체 자동 실행 (추천)

```bash
cd ver1
python RUN_ALL_ANALYSIS.py
```

**대화형 메뉴**로 각 단계를 선택적으로 실행할 수 있습니다.

### Option 2: 개별 실행

```bash
# Phase 1: 기초 분석 (약 2분)
python advanced_analysis.py

# Phase 2: 임계값 분석 (약 5분)
python threshold_analysis.py

# Phase 3: 세부 그룹 모델 (약 30-60분) ⏰
python subgroup_modeling.py

# Phase 4: SHAP 분석 (약 30-60분) ⏰
# 먼저 SHAP 설치: pip install shap
python shap_analysis.py

# Phase 5: 최종 보고서 생성 (즉시)
python generate_paper_report.py
```

---

## 📊 분석 내용 상세

### 1. 세부 그룹 분석 (Subgroup Analysis)

**목적**: 인구집단별 맞춤형 모델 개발

**그룹 분류**:
- **나이**: 20대, 30대, 40대, 50대, 60대+ (5 groups)
- **성별**: 남성, 여성 (2 groups)
- **BMI**: 저체중, 정상, 과체중, 비만1, 비만2 (5 groups)

**기대 효과**:
- 특정 인구집단에서 더 높은 정확도
- 그룹별 특화된 식습관 패턴 발견
- 개인맞춤형 권고안 근거 제공

**결과 예시**:
```
나이 그룹별 체중 예측 R²:
- 20대: 0.XX (n=1,357)
- 30대: 0.XX (n=4,587)
- 40대: 0.XX (n=10,226)
- 50대: 0.XX (n=9,763)
- 60대+: 0.XX (n=3,165)
```

---

### 2. SHAP 해석력 분석 (Interpretability)

**목적**: 모델의 "블랙박스"를 투명하게

**분석 항목**:
1. **Feature Importance**: 어떤 식습관이 가장 중요한가?
2. **Dependence Plots**: 특성 값에 따른 영향 변화
3. **Feature Interactions**: 식습관 간 상호작용
4. **Individual Explanations**: 개인별 예측 설명

**SHAP 설치**:
```bash
pip install shap
```

**시각화 결과**:
- `{indicator}_shap_summary.png`: 전체 특성 중요도
- `{indicator}_shap_dependence.png`: Top 5 특성 의존성
- `{indicator}_feature_interactions.png`: 상호작용 히트맵

---

### 3. 임계값 분석 (Threshold Analysis)

**목적**: 실천 가능한 건강 위험 기준 도출

**방법론**:
1. **임상적 기준**: WHO/대한의학회 가이드라인
2. **통계적 기준**: ROC curve, Youden's J statistic
3. **Percentile 기준**: 25th, 50th, 75th 백분위

**결과물**:
- 위험도 분류표
- 식습관별 최적 임계값
- 맞춤형 개선 권고안

**예시**:
```
BMI 위험도 분류:
- 정상 (18.5-23): 유지 전략
- 과체중 (23-25): 주의, 간식 빈도 3점 이하
- 비만1 (25-30): 위험, 고지방 육류 2점 이하
```

---

### 4. 논문 보고서 (Paper Report)

**자동 생성 내용**:
- Abstract
- Introduction
- Methods
- Results (표 및 그래프)
- Discussion
- Conclusions
- References
- Appendix

**형식**: Markdown (`.md`)
**위치**: `advanced_results/FINAL_PAPER_REPORT.md`

**다음 단계**:
1. 실제 결과값으로 플레이스홀더(`XX`) 대체
2. 그래프 이미지 삽입
3. 학술지 템플릿으로 변환 (LaTeX, Word 등)

---

## ⏱️ 소요 시간

| Phase | 내용 | 예상 시간 | 권장 |
|-------|------|----------|------|
| 1 | 기초 분석 | 2분 | ✅ 필수 |
| 2 | 임계값 분석 | 5분 | ✅ 필수 |
| 3 | 세부 그룹 모델 | 30-60분 | ⚠️ 선택 |
| 4 | SHAP 분석 | 30-60분 | ⚠️ 선택 |
| 5 | 보고서 생성 | 즉시 | ✅ 필수 |

**전체**: 약 1-2시간 (Phase 3, 4 포함 시)

**빠른 실행** (Phase 1, 2, 5만): 약 10분

---

## 💡 사용 팁

### 1. 단계적 실행 (권장)

```bash
# 먼저 빠른 분석 실행 (10분)
python advanced_analysis.py
python threshold_analysis.py
python generate_paper_report.py

# 결과 검토 후, 시간 있을 때 심화 분석
python subgroup_modeling.py   # 점심시간 or 밤에
python shap_analysis.py        # 점심시간 or 밤에
```

### 2. 선택적 실행

시간이 부족하면:
- **최소 필수**: Phase 1, 2, 5 (약 10분)
- **논문 제출용**: Phase 1, 2, 4, 5 (SHAP만, 약 1시간)
- **완전판**: 전체 실행 (약 2시간)

### 3. 결과 확인

```bash
# 생성된 파일 확인
ls -lh advanced_results/

# 보고서 보기
cat advanced_results/FINAL_PAPER_REPORT.md
```

---

## 📈 기대 결과

### 학술적 기여

1. **방법론**:
   - TabNet + Subgroup modeling
   - SHAP interpretability
   - 임상적 임계값 도출

2. **발견**:
   - 식습관-건강 관계 정량화 (R² 0.90)
   - 인구집단별 차이 규명
   - 우선순위 식습관 식별

3. **실용성**:
   - 개인맞춤형 권고안
   - 위험도 평가 도구
   - 임상 의사결정 지원

### 논문 투고 목표

**국내 학술지**:
- 대한의학회지 (KCI)
- 한국영양학회지
- 예방의학회지

**국제 학술지**:
- Nutrients (IF ~5.0)
- Journal of Nutritional Science (IF ~4.0)
- BMC Public Health (IF ~4.0)

---

## 🐛 문제 해결

### 1. SHAP 설치 오류

```bash
# Python 버전 확인 (3.8+ 필요)
python --version

# SHAP 재설치
pip uninstall shap
pip install shap --no-cache-dir
```

### 2. 메모리 부족

세부 그룹 모델 학습 시:
```python
# subgroup_modeling.py 수정
# 샘플 크기 줄이기
df_subset = df_subset.sample(n=1000, random_state=42)
```

### 3. 시간 단축

```bash
# Phase 3, 4를 백그라운드 실행
nohup python subgroup_modeling.py > subgroup.log 2>&1 &
nohup python shap_analysis.py > shap.log 2>&1 &

# 로그 확인
tail -f subgroup.log
```

---

## 📚 참고 자료

### Ver1 원본 결과

- `README.md`: Ver1 전체 성능 요약
- `docs/ANALYSIS_REPORT.md`: 상세 분석 보고서
- `docs/INPUT_OUTPUT_EXPLANATION.md`: 입출력 설명

### 관련 논문

1. TabNet: Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. *AAAI*.
2. SHAP: Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NIPS*.

---

## 🤝 기여

문의 사항이나 개선 제안:
- GitHub Issues
- Email: [your-email]

---

## 📄 라이선스

연구 목적으로 사용됩니다.

---

**🎓 논문 작성을 응원합니다!**

Ver1의 우수한 성능(R² 0.90)을 자랑스럽게 발표하세요.
이 분석 도구가 여러분의 연구를 한 단계 업그레이드시킬 것입니다!
