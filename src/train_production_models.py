"""
실서비스용 최종 모델 학습/저장.

run_lightgbm_tuned_shap.py에서 찾은 방식으로 질병별 LightGBM을 재학습하고,
- 학습된 모델 객체(joblib)
- SHAP TreeExplainer
- prior-shift(결정경계) 보정값
- 입력 피처 조립에 필요한 PCA/스케일러/피처 목록
을 ../result/production_models/ 에 저장한다. FastAPI 추론 서비스가 이 산출물을 그대로 로드해서 쓴다.

실행: python train_production_models.py
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import optuna
import shap
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_import import DataPreprocessor
from feature_engineering import apply_to_all_data
from utils import demo_cols, life_cols, bio_cols, diet_cols, mets_cols

SEED = 42
N_TRIALS = 40
OUT_DIR = '../result/production_models'
os.makedirs(OUT_DIR, exist_ok=True)

data_path = '../data/total_again.xlsx'
data_processor = DataPreprocessor(file_path=data_path, seed=SEED, normalize=False)
train_df, val_df, test_df, _ = data_processor.process_all(
    normalize=True, selection_strategy='max_disease_change'
)
train_df, val_df, test_df, _, pca = apply_to_all_data(train_df, val_df, test_df)

pca_cols = sorted([c for c in train_df.columns if 'pca' in c])
interaction_cols_present = [c for c in ['bmi_waist_risk', 'bp_age_risk', 'tg_hdl_ratio',
                                         'unhealthy_diet_score', 'healthy_diet_score',
                                         'diet_change_rate'] if c in train_df.columns]
delta_cols = sorted([c for c in train_df.columns
                      if '_delta' in c and not any(d in c for d in mets_cols)])
diet_feature_cols = [c for c in diet_cols if c in train_df.columns]
demo_feature_cols = [c for c in demo_cols if c in train_df.columns]
life_feature_cols = [c for c in life_cols if c in train_df.columns]
bio_feature_cols = [c for c in bio_cols if c in train_df.columns]

raw_input_cols = diet_feature_cols + demo_feature_cols + life_feature_cols + bio_feature_cols

feature_cols = (
    diet_feature_cols + demo_feature_cols + life_feature_cols + bio_feature_cols
    + delta_cols + interaction_cols_present + pca_cols
)
print(f"전체 피처 {len(feature_cols)}개 (원본 입력 {len(raw_input_cols)}개 + 파생 {len(feature_cols)-len(raw_input_cols)}개)")

full_train = pd.concat([train_df, val_df], ignore_index=True)


def grid_search_prior_shift(oof_probs, y_true, grid=np.arange(-2.0, 2.01, 0.2)):
    log_p = np.log(np.clip(oof_probs, 1e-9, 1.0))
    best_f1, best_shift = -1, (0.0, 0.0)
    for s0 in grid:
        for s2 in grid:
            pred = np.argmax(log_p + np.array([s0, 0.0, s2]), axis=1)
            f1 = f1_score(y_true, pred, average='macro')
            if f1 > best_f1:
                best_f1, best_shift = f1, (s0, s2)
    return best_shift, best_f1


disease_artifacts = {}

for disease in mets_cols:
    target_col = f'{disease}_delta'
    print(f"\n{'='*70}\n{disease}\n{'='*70}")

    X_tr = train_df[feature_cols].fillna(0).values
    y_tr = train_df[target_col].values
    X_va = val_df[feature_cols].fillna(0).values
    y_va = val_df[target_col].values

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'num_leaves': trial.suggest_int('num_leaves', 7, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        model = lgb.LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbosity=-1,
                                    class_weight='balanced')
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        return f1_score(y_va, pred, average='macro')

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    print(f"  최적 파라미터: {best_params}")

    X_tv = full_train[feature_cols].fillna(0).values
    y_tv = full_train[target_col].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X_tv), 3))
    for tr_idx, va_idx in skf.split(X_tv, y_tv):
        m = lgb.LGBMClassifier(**best_params, random_state=SEED, n_jobs=-1, verbosity=-1,
                                class_weight='balanced')
        m.fit(X_tv[tr_idx], y_tv[tr_idx])
        oof[va_idx] = m.predict_proba(X_tv[va_idx])
    shift, shifted_f1 = grid_search_prior_shift(oof, y_tv)
    print(f"  prior-shift: {shift}, OOF F1(shift 적용): {shifted_f1:.3f}")

    final_model = lgb.LGBMClassifier(**best_params, random_state=SEED, n_jobs=-1, verbosity=-1,
                                      class_weight='balanced')
    final_model.fit(X_tv, y_tv)

    X_te = test_df[feature_cols].fillna(0).values
    y_te = test_df[target_col].values
    test_prob = final_model.predict_proba(X_te)
    log_test = np.log(np.clip(test_prob, 1e-9, 1.0))
    test_pred = np.argmax(log_test + np.array([shift[0], 0.0, shift[1]]), axis=1)
    test_f1 = f1_score(y_te, test_pred, average='macro')
    print(f"  최종 Test F1(prior-shift 적용): {test_f1:.3f}")

    explainer = shap.TreeExplainer(final_model)

    safe_name = disease.replace(' ', '_').replace('/', '_')
    joblib.dump(final_model, f'{OUT_DIR}/{safe_name}_model.joblib')
    joblib.dump(explainer, f'{OUT_DIR}/{safe_name}_shap_explainer.joblib')

    disease_artifacts[disease] = {
        'safe_name': safe_name,
        'best_params': best_params,
        'prior_shift': [float(shift[0]), 0.0, float(shift[1])],
        'test_f1_macro': float(test_f1),
    }

meta = {
    'feature_cols': feature_cols,
    'raw_input_cols': raw_input_cols,
    'diet_cols': diet_feature_cols,
    'demo_cols': demo_feature_cols,
    'life_cols': life_feature_cols,
    'bio_cols': bio_feature_cols,
    'delta_cols': delta_cols,          # 최초 방문(과거 기록 없음)에는 전부 0으로 채움
    'interaction_cols': interaction_cols_present,
    'pca_cols': pca_cols,
    'mets_cols': mets_cols,
    'disease_artifacts': disease_artifacts,
    'class_names': ['개선', '유지', '악화'],
}
with open(f'{OUT_DIR}/meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

joblib.dump(pca, f'{OUT_DIR}/pca.joblib')
joblib.dump(data_processor.scaler, f'{OUT_DIR}/scaler.joblib')
joblib.dump(data_processor.label_encoders, f'{OUT_DIR}/label_encoders.joblib')

print(f"\n저장 완료: {OUT_DIR}/")
print(json.dumps({d: a['test_f1_macro'] for d, a in disease_artifacts.items()}, ensure_ascii=False, indent=2))
