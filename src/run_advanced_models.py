"""
Priority (사용자 추가 요청): 모델 고도화 비교 — TabPFN(foundation model) vs GBM(XGBoost/LightGBM/CatBoost)
vs 기존 baseline(커스텀 멀티브랜치 신경망, F1-macro 0.491~0.508)

기존 파이프라인(DataPreprocessor + apply_to_all_data)으로 만든 동일한 75차원 피처
(diet+demo+life+bio+delta+interaction+pca)와 3-class 타겟(개선/유지/악화)을 그대로 사용해서
공정하게 비교한다. SMOTE 리샘플링은 쓰지 않고 class_weight/sample_weight로 불균형을 보정한다
(TabPFN은 in-context learning이라 SMOTE로 합성 샘플을 늘리는 것이 원래 취지에 맞지 않음).

실행: python run_advanced_models.py
결과: ../result/advanced_models_comparison.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

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

# 기존 신경망 파이프라인(SingleDietImpactDataset)과 동일한 컬럼 순서로 flat 피처 구성
pca_cols = sorted([c for c in train_df.columns if 'pca' in c])
interaction_cols_present = [c for c in ['bmi_waist_risk', 'bp_age_risk', 'tg_hdl_ratio',
                                         'unhealthy_diet_score', 'healthy_diet_score',
                                         'diet_change_rate'] if c in train_df.columns]
delta_cols = sorted([c for c in train_df.columns
                      if '_delta' in c and not any(d in c for d in mets_cols)])

feature_cols = (
    [c for c in diet_cols if c in train_df.columns]
    + [c for c in demo_cols if c in train_df.columns]
    + [c for c in life_cols if c in train_df.columns]
    + [c for c in bio_cols if c in train_df.columns]
    + delta_cols
    + interaction_cols_present
    + pca_cols
)
print(f"피처 수: {len(feature_cols)}")

# TabPFN은 학습 없이 in-context로 예측하므로 train+val을 합쳐 컨텍스트로 사용
full_train = pd.concat([train_df, val_df], ignore_index=True)

TABPFN_MAX_ROWS = 8000  # TabPFN 실용 상한(속도/메모리) 근처로 서브샘플링

results = []


def evaluate(y_true, y_pred, y_prob, model_name, disease, n_train, elapsed):
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    try:
        roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        roc = float('nan')
    print(f"  [{model_name}] {disease}: F1={f1:.3f} Acc={acc:.3f} ROC-AUC={roc:.3f} "
          f"(n_train={n_train}, {elapsed:.1f}s)")
    results.append({'Model': model_name, 'Disease': disease, 'F1_Macro': f1,
                     'Accuracy': acc, 'ROC_AUC': roc, 'N_Train': n_train,
                     'Elapsed_sec': elapsed})


for disease in mets_cols:
    target_col = f'{disease}_delta'
    print(f"\n{'='*70}\n{disease}\n{'='*70}")

    X_train_full = full_train[feature_cols].fillna(0).values
    y_train_full = full_train[target_col].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df[target_col].values

    sw_train_full = compute_sample_weight('balanced', y_train_full)

    # ---- XGBoost ----
    import xgboost as xgb
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        objective='multi:softprob', num_class=3, random_state=42,
        n_jobs=-1, eval_metric='mlogloss'
    )
    xgb_model.fit(X_train_full, y_train_full, sample_weight=sw_train_full)
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)
    evaluate(y_test, y_pred, y_prob, 'XGBoost', disease, len(X_train_full), time.time() - t0)

    # ---- LightGBM ----
    try:
        import lightgbm as lgb
        t0 = time.time()
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=-1, class_weight='balanced'
        )
        lgb_model.fit(X_train_full, y_train_full)
        y_pred = lgb_model.predict(X_test)
        y_prob = lgb_model.predict_proba(X_test)
        evaluate(y_test, y_pred, y_prob, 'LightGBM', disease, len(X_train_full), time.time() - t0)
    except ImportError:
        print("  lightgbm 미설치, 건너뜀")

    # ---- CatBoost ----
    try:
        from catboost import CatBoostClassifier
        t0 = time.time()
        cat_model = CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.05, random_state=42,
            verbose=False, auto_class_weights='Balanced'
        )
        cat_model.fit(X_train_full, y_train_full)
        y_pred = cat_model.predict(X_test).ravel()
        y_prob = cat_model.predict_proba(X_test)
        evaluate(y_test, y_pred, y_prob, 'CatBoost', disease, len(X_train_full), time.time() - t0)
    except ImportError:
        print("  catboost 미설치, 건너뜀")

    # ---- TabPFN ----
    try:
        from tabpfn import TabPFNClassifier
        t0 = time.time()

        if len(X_train_full) > TABPFN_MAX_ROWS:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train_full), TABPFN_MAX_ROWS, replace=False)
            X_tp, y_tp = X_train_full[idx], y_train_full[idx]
        else:
            X_tp, y_tp = X_train_full, y_train_full

        tabpfn_model = TabPFNClassifier(device=str(device), random_state=42)
        tabpfn_model.fit(X_tp, y_tp)
        y_pred = tabpfn_model.predict(X_test)
        y_prob = tabpfn_model.predict_proba(X_test)
        evaluate(y_test, y_pred, y_prob, 'TabPFN', disease, len(X_tp), time.time() - t0)
    except Exception as e:
        print(f"  TabPFN 실패: {e}")

results_df = pd.DataFrame(results)
out_path = '../result/advanced_models_comparison.csv'
results_df.to_csv(out_path, index=False)
print(f"\n저장: {out_path}")

print("\n" + "=" * 70)
print("모델별 평균 (5개 질병)")
print("=" * 70)
print(results_df.groupby('Model')[['F1_Macro', 'Accuracy', 'ROC_AUC']].mean().round(4)
      .sort_values('F1_Macro', ascending=False))
