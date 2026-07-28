"""
Priority (사용자 추가 요청 ①②): LightGBM Optuna 튜닝 + SHAP 해석

run_advanced_models.py에서 LightGBM(F1-macro 0.602, 튜닝 전)이 기존 커스텀 딥러닝
baseline(0.491)을 크게 앞선 것을 확인했다. 여기서는
1) Optuna로 질병별 LightGBM 하이퍼파라미터를 튜닝하고 (val_df로 검증, 환자 단위 분리 유지)
2) 최종 모델에 SHAP TreeExplainer를 붙여 원본(비-PCA) 피처 기준 식이 요인 중요도를 뽑는다.

실행: python run_lightgbm_tuned_shap.py
결과:
  ../result/lightgbm_tuned_comparison.csv       (튜닝 전/후 F1 비교)
  ../result/lightgbm_shap_importance.csv        (질병별 전체 피처 중요도)
  ../result/lightgbm_shap_diet_top15.csv        (질병별 식이 변수만 top15)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import optuna
import shap
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_import import DataPreprocessor
from feature_engineering import apply_to_all_data
from utils import demo_cols, life_cols, bio_cols, diet_cols, mets_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

data_path = '../data/total_again.xlsx'
data_processor = DataPreprocessor(file_path=data_path, seed=42, normalize=False)
train_df, val_df, test_df, _ = data_processor.process_all(
    normalize=True, selection_strategy='max_disease_change'
)
train_df, val_df, test_df, _, _pca = apply_to_all_data(train_df, val_df, test_df)

pca_cols = sorted([c for c in train_df.columns if 'pca' in c])
interaction_cols_present = [c for c in ['bmi_waist_risk', 'bp_age_risk', 'tg_hdl_ratio',
                                         'unhealthy_diet_score', 'healthy_diet_score',
                                         'diet_change_rate'] if c in train_df.columns]
delta_cols = sorted([c for c in train_df.columns
                      if '_delta' in c and not any(d in c for d in mets_cols)])
diet_feature_cols = [c for c in diet_cols if c in train_df.columns]

feature_cols = (
    diet_feature_cols
    + [c for c in demo_cols if c in train_df.columns]
    + [c for c in life_cols if c in train_df.columns]
    + [c for c in bio_cols if c in train_df.columns]
    + delta_cols
    + interaction_cols_present
    + pca_cols
)
print(f"피처 수: {len(feature_cols)} (식이 변수 {len(diet_feature_cols)}개)")

full_train = pd.concat([train_df, val_df], ignore_index=True)

comparison_rows = []
shap_importance_rows = []
shap_diet_top15_rows = []

N_TRIALS = 40

for disease in mets_cols:
    target_col = f'{disease}_delta'
    print(f"\n{'='*70}\n{disease}\n{'='*70}")

    X_tr = train_df[feature_cols].fillna(0).values
    y_tr = train_df[target_col].values
    X_va = val_df[feature_cols].fillna(0).values
    y_va = val_df[target_col].values
    X_te = test_df[feature_cols].fillna(0).values
    y_te = test_df[target_col].values

    sw_tr = compute_sample_weight('balanced', y_tr)

    # ---- baseline (튜닝 전, run_advanced_models.py와 동일 설정) ----
    base_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        random_state=42, n_jobs=-1, verbosity=-1, class_weight='balanced'
    )
    base_model.fit(np.vstack([X_tr, X_va]),
                    np.concatenate([y_tr, y_va]))
    base_pred = base_model.predict(X_te)
    base_f1 = f1_score(y_te, base_pred, average='macro')

    # ---- Optuna 튜닝 (train으로 학습, val로 평가) ----
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
        model = lgb.LGBMClassifier(
            **params, random_state=42, n_jobs=-1, verbosity=-1, class_weight='balanced'
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        return f1_score(y_va, pred, average='macro')

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    print(f"  최적 파라미터: {best_params}")
    print(f"  Val F1(탐색중 best): {study.best_value:.3f}")

    # ---- 최종 모델: train+val로 재학습, test로 평가 ----
    final_model = lgb.LGBMClassifier(
        **best_params, random_state=42, n_jobs=-1, verbosity=-1, class_weight='balanced'
    )
    final_model.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    final_pred = final_model.predict(X_te)
    final_prob = final_model.predict_proba(X_te)

    tuned_f1 = f1_score(y_te, final_pred, average='macro')
    tuned_acc = accuracy_score(y_te, final_pred)
    try:
        tuned_roc = roc_auc_score(y_te, final_prob, multi_class='ovr', average='macro')
    except ValueError:
        tuned_roc = float('nan')

    print(f"  튜닝 전 Test F1: {base_f1:.3f} -> 튜닝 후 Test F1: {tuned_f1:.3f} "
          f"({(tuned_f1 - base_f1) / base_f1 * 100:+.1f}%)")
    print(f"  튜닝 후 Test Acc: {tuned_acc:.3f}, ROC-AUC: {tuned_roc:.3f}")

    comparison_rows.append({
        'Disease': disease, 'F1_Before_Tuning': base_f1, 'F1_After_Tuning': tuned_f1,
        'Accuracy_Tuned': tuned_acc, 'ROC_AUC_Tuned': tuned_roc,
        'Best_Params': str(best_params),
    })

    # ---- SHAP (원본 피처 기준, PCA 압축 없이 직접 해석 가능) ----
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_te)
    # 다중클래스면 shap_values가 (n_class, n_sample, n_feature) 또는 리스트 형태일 수 있음 -> 절대값 클래스 평균
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.shape[0] == len(X_te) \
            else np.abs(shap_values).mean(axis=(1, 2)) if shap_values.shape[1] == len(X_te) else np.abs(shap_values).mean(axis=0).mean(axis=-1)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    imp_df = pd.DataFrame({'Feature': feature_cols, 'Mean_Abs_SHAP': mean_abs_shap}) \
        .sort_values('Mean_Abs_SHAP', ascending=False)
    imp_df['Disease'] = disease
    shap_importance_rows.append(imp_df)

    diet_only = imp_df[imp_df['Feature'].isin(diet_feature_cols)].head(15)
    shap_diet_top15_rows.append(diet_only)

    print(f"  Top5 전체 피처: {imp_df['Feature'].head(5).tolist()}")
    print(f"  Top5 식이 피처: {diet_only['Feature'].head(5).tolist()}")

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv('../result/lightgbm_tuned_comparison.csv', index=False)

all_shap = pd.concat(shap_importance_rows, ignore_index=True)
all_shap.to_csv('../result/lightgbm_shap_importance.csv', index=False)

diet_shap = pd.concat(shap_diet_top15_rows, ignore_index=True)
diet_shap.to_csv('../result/lightgbm_shap_diet_top15.csv', index=False)

print("\n" + "=" * 70)
print("최종 요약: 튜닝 전/후 비교")
print("=" * 70)
print(comparison_df[['Disease', 'F1_Before_Tuning', 'F1_After_Tuning']].to_string(index=False))
print(f"\n평균 F1: 튜닝 전 {comparison_df['F1_Before_Tuning'].mean():.4f} -> "
      f"튜닝 후 {comparison_df['F1_After_Tuning'].mean():.4f}")

print("\n저장 완료:")
print("  ../result/lightgbm_tuned_comparison.csv")
print("  ../result/lightgbm_shap_importance.csv")
print("  ../result/lightgbm_shap_diet_top15.csv")
