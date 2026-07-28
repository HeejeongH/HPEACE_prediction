"""
식습관 기반 대사증후군 위험요인 예측 API (FastAPI)

프론트엔드(설문 폼)가 이 서비스의 POST /predict 로 사용자의 식이+인구통계+생활습관+
최근 검진수치를 보내면, 질병(허리둘레/혈압/혈당/중성지방/HDL) 5개에 대해
개선/유지/악화 확률과, 그 예측에 가장 크게 기여한 식이 요인(SHAP)을 함께 돌려준다.

모델/전처리 산출물(../result/production_models/, train_production_models.py가 생성)을
그대로 불러 쓴다 — 학습 때 쓴 scaler/label_encoders/pca의 feature_names_in_ 속성을
그대로 활용해 컬럼 순서를 맞추므로, 피처 순서를 이 서비스에서 다시 손으로 만들지 않는다.

실행(로컬): uvicorn main:app --reload --port 8000
"""
import json
import os
import sys
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_DIR = os.environ.get(
    'PRODUCTION_MODEL_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'result', 'production_models'),
)
SRC_DIR = os.environ.get(
    'PROJECT_SRC_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'),
)
sys.path.append(SRC_DIR)

from feature_engineering import create_medical_interactions, create_change_features  # noqa: E402

app = FastAPI(title="식습관 기반 대사증후군 위험요인 예측 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=['*'],
    allow_headers=['*'],
)

with open(os.path.join(MODEL_DIR, 'meta.json'), encoding='utf-8') as f:
    META = json.load(f)

PCA = joblib.load(os.path.join(MODEL_DIR, 'pca.joblib'))
SCALER = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
LABEL_ENCODERS = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.joblib'))

MODELS = {}
EXPLAINERS = {}
for disease, art in META['disease_artifacts'].items():
    MODELS[disease] = joblib.load(os.path.join(MODEL_DIR, f"{art['safe_name']}_model.joblib"))
    EXPLAINERS[disease] = joblib.load(os.path.join(MODEL_DIR, f"{art['safe_name']}_shap_explainer.joblib"))

DIET_RAW_NAMES = [c.replace('_T0', '') for c in META['diet_cols']]
BIO_RAW_NAMES = [c.replace('_T0', '') for c in META['bio_cols']]


class SurveyInput(BaseModel):
    # 인구통계
    성별: str = Field(..., description="'M' 또는 'F'")
    나이: float
    신장: float = Field(..., description="cm")

    # 생활습관 (원본 코딩: 흡연 0=비흡연/1=과거흡연/2=현재흡연, 활동량 0~2, 음주 0~2)
    흡연: int
    활동량: int
    음주: int

    # 최근 검진수치 (baseline biomarker) — 모르면 비워도 되지만 정확도가 크게 떨어짐
    bio: Dict[str, float] = Field(
        ..., description=f"키: {BIO_RAW_NAMES}"
    )

    # 식습관 설문 19문항 (1~3 또는 1~4 척도, 문항별 상이)
    diet: Dict[str, int] = Field(
        ..., description=f"키: {DIET_RAW_NAMES}"
    )


class DietFactor(BaseModel):
    feature: str
    shap_contribution: float


class DiseaseResult(BaseModel):
    disease: str
    disease_kr: str
    prob_improve: float
    prob_maintain: float
    prob_worsen: float
    predicted: str
    top_diet_factors: List[DietFactor]


class PredictResponse(BaseModel):
    results: List[DiseaseResult]


DISEASE_KR = {
    'Increased waist circumference': '복부비만(허리둘레)',
    'Elevated blood pressure': '고혈압',
    'Impaired fasting glucose': '공복혈당장애',
    'Elevated triglycerides': '고중성지방혈증',
    'Decreased HDL-C': 'HDL 콜레스테롤 저하',
}


def build_feature_row(survey: SurveyInput) -> pd.DataFrame:
    row = {'days_between': 0.0}

    row['나이_T0'] = survey.나이
    row['신장_T0'] = survey.신장
    row['성별_T0'] = 0 if survey.성별.upper() == 'M' else 1
    row['흡연_T0'] = survey.흡연
    row['활동량_T0'] = survey.활동량
    row['음주_T0'] = survey.음주

    missing_bio = [c for c in BIO_RAW_NAMES if c not in survey.bio]
    if missing_bio:
        raise HTTPException(422, f"검진수치 누락: {missing_bio}")
    for raw_name in BIO_RAW_NAMES:
        row[f'{raw_name}_T0'] = survey.bio[raw_name]

    missing_diet = [c for c in DIET_RAW_NAMES if c not in survey.diet]
    if missing_diet:
        raise HTTPException(422, f"식습관 응답 누락: {missing_diet}")
    for raw_name in DIET_RAW_NAMES:
        row[f'{raw_name}_T0'] = survey.diet[raw_name]

    # 최초 예측(이전 방문 기록 없음) -> 변화(delta) 관련 값은 전부 0으로 둠
    for col in META['delta_cols']:
        row[col] = 0.0

    df = pd.DataFrame([row])

    # 연속형 스케일링: 학습 때 fit된 scaler가 기억하는 컬럼 순서를 그대로 사용
    cont_cols = list(SCALER.feature_names_in_)
    for c in cont_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[cont_cols] = SCALER.transform(df[cont_cols])

    # 순서형 인코딩: 학습 때 쓴 LabelEncoder 재사용, 미학습 값은 최빈값(0번 클래스)로 대체
    for col, le in LABEL_ENCODERS.items():
        if col not in df.columns:
            continue
        val = df.at[0, col]
        if val not in le.classes_:
            val = le.classes_[0]
        df[col] = le.transform([val])

    df = create_medical_interactions(df)
    df = create_change_features(df)

    pca_input_cols = list(PCA.feature_names_in_)
    for c in pca_input_cols:
        if c not in df.columns:
            df[c] = 0.0
    pca_result = PCA.transform(df[pca_input_cols].fillna(0))
    for i in range(pca_result.shape[1]):
        df[f'pca_component_{i}'] = pca_result[:, i]

    final_cols = META['feature_cols']
    for c in final_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[final_cols].fillna(0)


@app.get("/health")
def health():
    return {"status": "ok", "diseases": list(MODELS.keys())}


@app.post("/predict", response_model=PredictResponse)
def predict(survey: SurveyInput):
    X = build_feature_row(survey)
    class_names = META['class_names']  # ['개선', '유지', '악화']

    results = []
    for disease, model in MODELS.items():
        art = META['disease_artifacts'][disease]
        prob = model.predict_proba(X.values)[0]
        shift = np.array(art['prior_shift'])
        log_p = np.log(np.clip(prob, 1e-9, 1.0)) + shift
        shifted_prob = np.exp(log_p)
        shifted_prob = shifted_prob / shifted_prob.sum()
        pred_idx = int(np.argmax(shifted_prob))

        explainer = EXPLAINERS[disease]
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[2][0]  # '악화' 클래스 기준 기여도
        elif np.ndim(shap_values) == 3:
            sv = shap_values[0, :, 2]
        else:
            sv = shap_values[0]

        diet_feature_set = set(META['diet_cols'])
        contrib = sorted(
            [(f, float(v)) for f, v in zip(META['feature_cols'], sv) if f in diet_feature_set],
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
        top_diet_factors = [{'feature': f.replace('_T0', ''), 'shap_contribution': v} for f, v in contrib]

        results.append(DiseaseResult(
            disease=disease,
            disease_kr=DISEASE_KR.get(disease, disease),
            prob_improve=float(shifted_prob[0]),
            prob_maintain=float(shifted_prob[1]),
            prob_worsen=float(shifted_prob[2]),
            predicted=class_names[pred_idx],
            top_diet_factors=top_diet_factors,
        ))

    return PredictResponse(results=results)
