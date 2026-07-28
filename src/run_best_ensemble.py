"""
"모든 방법 다 해서 가장 좋은 모델" — 종합 고도화 파이프라인

1) LightGBM / XGBoost / CatBoost: 5-fold StratifiedKFold + Optuna(각 20 trials)로
   단일 train/val split보다 견고하게 재탐색.
2) TabPFN: predict_proba는 지원하되 sample_weight는 없으므로, 컨텍스트 자체를
   클래스 균형 언더샘플링해서 넣어 불균형 편향을 직접 보정.
3) 4개 모델의 5-fold Out-Of-Fold 확률을 모아 멀티노미얼 로지스틱 회귀 메타러너로
   스태킹 앙상블 학습 (단순 평균 앙상블과도 비교).
4) 최종 후보(튜닝 LightGBM 단독 / 평균 앙상블 / 스태킹 앙상블) 중 OOF F1-macro가
   가장 높은 것을 골라, 그 OOF 확률로 클래스별 prior-shift(결정경계)를 그리드서치로
   추가 최적화한 뒤 test set에 적용.
5) 모든 후보의 test 성능을 한 표로 비교하고, 최종 승자를 joblib으로 저장.

실행: python run_best_ensemble.py
결과: ../result/best_ensemble_comparison.csv, ../result/best_ensemble_models/*.joblib
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
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

N_SPLITS = 5
N_TRIALS = 20
TABPFN_CAP = 6000
SEED = 42

os.makedirs('../result/best_ensemble_models', exist_ok=True)

data_path = '../data/total_again.xlsx'
data_processor = DataPreprocessor(file_path=data_path, seed=SEED, normalize=False)
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
print(f"피처 수: {len(feature_cols)}")

# 각 delta row = 서로 다른 환자 1명이므로 train+val을 합쳐도 환자 중복/누수 없음
trainval_df = pd.concat([train_df, val_df], ignore_index=True)


def balanced_subsample_idx(y, cap, rng):
    """TabPFN 컨텍스트용: 클래스당 최대 cap/n_class개로 언더샘플링해 균형 맞춤."""
    classes = np.unique(y)
    per_class_cap = max(1, cap // len(classes))
    idx_list = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) > per_class_cap:
            c_idx = rng.choice(c_idx, per_class_cap, replace=False)
        idx_list.append(c_idx)
    idx = np.concatenate(idx_list)
    rng.shuffle(idx)
    return idx


def tune_gbm_cv(model_name, X, y, n_trials, seed):
    """K-fold CV 기반 Optuna 튜닝, best_params 반환."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    def objective(trial):
        if model_name == 'lightgbm':
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
        elif model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
        else:  # catboost
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
            }

        fold_f1s = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            sw = compute_sample_weight('balanced', y_tr)

            if model_name == 'lightgbm':
                m = lgb.LGBMClassifier(**params, random_state=seed, n_jobs=-1, verbosity=-1)
                m.fit(X_tr, y_tr, sample_weight=sw)
            elif model_name == 'xgboost':
                m = xgb.XGBClassifier(**params, objective='multi:softprob', num_class=3,
                                       random_state=seed, n_jobs=-1, eval_metric='mlogloss')
                m.fit(X_tr, y_tr, sample_weight=sw)
            else:
                m = CatBoostClassifier(**params, random_state=seed, verbose=False,
                                        auto_class_weights='Balanced')
                m.fit(X_tr, y_tr)

            pred = m.predict(X_va)
            fold_f1s.append(f1_score(y_va, np.ravel(pred), average='macro'))

        return float(np.mean(fold_f1s))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def make_model(model_name, params, seed):
    if model_name == 'lightgbm':
        return lgb.LGBMClassifier(**params, random_state=seed, n_jobs=-1, verbosity=-1)
    elif model_name == 'xgboost':
        return xgb.XGBClassifier(**params, objective='multi:softprob', num_class=3,
                                  random_state=seed, n_jobs=-1, eval_metric='mlogloss')
    else:
        return CatBoostClassifier(**params, random_state=seed, verbose=False,
                                   auto_class_weights='Balanced')


def oof_probs_gbm(model_name, params, X, y, seed):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros((len(X), 3))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        m = make_model(model_name, params, seed)
        if model_name == 'catboost':
            m.fit(X_tr, y_tr)
        else:
            sw = compute_sample_weight('balanced', y_tr)
            m.fit(X_tr, y_tr, sample_weight=sw)
        oof[va_idx] = m.predict_proba(X[va_idx])
    return oof


def oof_probs_tabpfn(X, y, seed):
    from tabpfn import TabPFNClassifier
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros((len(X), 3))
    rng = np.random.RandomState(seed)
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        ctx_idx = balanced_subsample_idx(y_tr, TABPFN_CAP, rng)
        m = TabPFNClassifier(device=str(device), random_state=seed)
        m.fit(X_tr[ctx_idx], y_tr[ctx_idx])
        oof[va_idx] = m.predict_proba(X[va_idx])
    return oof


def fit_full_tabpfn(X, y, seed):
    from tabpfn import TabPFNClassifier
    rng = np.random.RandomState(seed)
    ctx_idx = balanced_subsample_idx(y, TABPFN_CAP, rng)
    m = TabPFNClassifier(device=str(device), random_state=seed)
    m.fit(X[ctx_idx], y[ctx_idx])
    return m


def grid_search_prior_shift(oof_probs, y_true, grid=np.arange(-2.0, 2.01, 0.2)):
    """3-class용 클래스별 log-prior shift 그리드서치로 F1-macro 최대화 (class1 기준 고정)."""
    log_p = np.log(np.clip(oof_probs, 1e-9, 1.0))
    best_f1, best_shift = -1, (0.0, 0.0)
    for s0 in grid:
        for s2 in grid:
            shift = np.array([s0, 0.0, s2])
            pred = np.argmax(log_p + shift, axis=1)
            f1 = f1_score(y_true, pred, average='macro')
            if f1 > best_f1:
                best_f1, best_shift = f1, (s0, s2)
    return best_shift, best_f1


all_comparison_rows = []

for disease in mets_cols:
    target_col = f'{disease}_delta'
    print(f"\n{'='*80}\n{disease}\n{'='*80}")
    t_disease = time.time()

    X_tv = trainval_df[feature_cols].fillna(0).values
    y_tv = trainval_df[target_col].values
    X_te = test_df[feature_cols].fillna(0).values
    y_te = test_df[target_col].values

    best_params = {}
    oof_probs = {}
    test_probs = {}

    for model_name in ['lightgbm', 'xgboost', 'catboost']:
        t0 = time.time()
        params, cv_f1 = tune_gbm_cv(model_name, X_tv, y_tv, N_TRIALS, SEED)
        best_params[model_name] = params
        print(f"  [{model_name}] CV F1-macro(탐색): {cv_f1:.3f} ({time.time()-t0:.0f}s) params={params}")

        oof_probs[model_name] = oof_probs_gbm(model_name, params, X_tv, y_tv, SEED)

        final_model = make_model(model_name, params, SEED)
        if model_name == 'catboost':
            final_model.fit(X_tv, y_tv)
        else:
            sw = compute_sample_weight('balanced', y_tv)
            final_model.fit(X_tv, y_tv, sample_weight=sw)
        test_probs[model_name] = final_model.predict_proba(X_te)

    print("  [tabpfn] OOF/최종 컨텍스트 학습 중...")
    t0 = time.time()
    oof_probs['tabpfn'] = oof_probs_tabpfn(X_tv, y_tv, SEED)
    tabpfn_final = fit_full_tabpfn(X_tv, y_tv, SEED)
    test_probs['tabpfn'] = tabpfn_final.predict_proba(X_te)
    print(f"  [tabpfn] 완료 ({time.time()-t0:.0f}s)")

    model_names = ['lightgbm', 'xgboost', 'catboost', 'tabpfn']

    # ---- 후보 1: 각 모델 단독 (OOF 기준 최고 성능 단일 모델) ----
    for name in model_names:
        f1 = f1_score(y_tv, np.argmax(oof_probs[name], axis=1), average='macro')
        print(f"  단독 {name} OOF F1-macro: {f1:.3f}")

    # ---- 후보 2: 단순 평균 앙상블 ----
    oof_avg = np.mean([oof_probs[n] for n in model_names], axis=0)
    test_avg = np.mean([test_probs[n] for n in model_names], axis=0)
    avg_oof_f1 = f1_score(y_tv, np.argmax(oof_avg, axis=1), average='macro')
    print(f"  평균 앙상블 OOF F1-macro: {avg_oof_f1:.3f}")

    # ---- 후보 3: 스태킹(메타러너) 앙상블 ----
    oof_stack_X = np.hstack([oof_probs[n] for n in model_names])
    test_stack_X = np.hstack([test_probs[n] for n in model_names])
    meta = LogisticRegression(max_iter=2000, multi_class='multinomial')
    meta.fit(oof_stack_X, y_tv)
    oof_meta_probs = meta.predict_proba(oof_stack_X)
    stack_oof_f1 = f1_score(y_tv, np.argmax(oof_meta_probs, axis=1), average='macro')
    print(f"  스태킹 앙상블 OOF F1-macro: {stack_oof_f1:.3f}")

    # ---- 최고 OOF 성능 후보 선택 ----
    candidates_oof = {
        **{f'single_{n}': (oof_probs[n], test_probs[n]) for n in model_names},
        'avg_ensemble': (oof_avg, test_avg),
        'stacked_ensemble': (oof_meta_probs, meta.predict_proba(test_stack_X)),
    }
    best_name = max(candidates_oof, key=lambda k: f1_score(y_tv, np.argmax(candidates_oof[k][0], axis=1), average='macro'))
    best_oof_probs, best_test_probs = candidates_oof[best_name]
    print(f"  >> 최고 후보: {best_name}")

    # ---- prior-shift(결정경계) 추가 최적화 ----
    shift, shifted_oof_f1 = grid_search_prior_shift(best_oof_probs, y_tv)
    print(f"  prior-shift 최적화: {best_name} OOF F1 {f1_score(y_tv, np.argmax(best_oof_probs,axis=1), average='macro'):.3f} -> {shifted_oof_f1:.3f} (shift={shift})")

    log_test = np.log(np.clip(best_test_probs, 1e-9, 1.0))
    final_test_pred = np.argmax(log_test + np.array([shift[0], 0.0, shift[1]]), axis=1)

    final_f1 = f1_score(y_te, final_test_pred, average='macro')
    final_acc = accuracy_score(y_te, final_test_pred)
    try:
        final_roc = roc_auc_score(y_te, best_test_probs, multi_class='ovr', average='macro')
    except ValueError:
        final_roc = float('nan')

    # 비교용: 각 방법의 test 성능도 전부 계산
    for cand_name, (c_oof, c_test) in candidates_oof.items():
        pred = np.argmax(c_test, axis=1)
        f1 = f1_score(y_te, pred, average='macro')
        acc = accuracy_score(y_te, pred)
        all_comparison_rows.append({
            'Disease': disease, 'Method': cand_name, 'Test_F1_Macro': f1, 'Test_Accuracy': acc,
        })
    all_comparison_rows.append({
        'Disease': disease, 'Method': f'{best_name}+prior_shift', 'Test_F1_Macro': final_f1,
        'Test_Accuracy': final_acc,
    })

    print(f"  ★ 최종(선택+prior-shift) Test F1-macro: {final_f1:.3f}, Acc: {final_acc:.3f}, ROC-AUC: {final_roc:.3f}")
    print(f"  질병 처리 시간: {time.time()-t_disease:.0f}s")

    joblib.dump({
        'best_params': best_params, 'best_candidate': best_name, 'shift': shift,
        'model_names': model_names, 'feature_cols': feature_cols,
    }, f'../result/best_ensemble_models/{disease.replace(" ", "_").replace("/", "_")}_config.joblib')

comparison_df = pd.DataFrame(all_comparison_rows)
comparison_df.to_csv('../result/best_ensemble_comparison.csv', index=False)

print("\n" + "=" * 80)
print("전체 요약: 방법별 평균 Test F1-macro (5개 질병)")
print("=" * 80)
print(comparison_df.groupby('Method')['Test_F1_Macro'].mean().sort_values(ascending=False).round(4))
print(f"\n저장: ../result/best_ensemble_comparison.csv")
print(f"저장: ../result/best_ensemble_models/*.joblib")
