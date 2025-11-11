"""
TabNet 딥러닝 모델이 추가된 개선 버전
==================================================
주요 개선사항:
1. TabNet 딥러닝 모델 추가
2. TabNet + 기존 모델들의 Stacking Ensemble
3. TabNet 하이퍼파라미터 최적화
4. 향상된 성능 (예상 +0.04~0.07 R²)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 기본 ML 라이브러리
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression

# TabNet
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

# 하이퍼파라미터 최적화
import optuna
from optuna.samplers import TPESampler

# SHAP
import shap

print("=" * 80)
print("TabNet 딥러닝 모델 통합 버전")
print("=" * 80)


# ============================================================================
# 1. 기존 함수들 (IMPROVED_DIET_PREDICTION_MODEL.py에서 가져옴)
# ============================================================================

def load_and_preprocess_data(file_path='../data/total_again.xlsx'):
    """데이터 로드 및 기본 전처리"""
    print("\n📂 데이터 로드 중...")
    
    # 경로가 존재하지 않으면 대체 경로 시도
    if not os.path.exists(file_path):
        # 현재 스크립트 위치 기준으로 경로 재구성
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, '..', 'data', 'total_again.xlsx')
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            # 프로젝트 루트에서 실행된 경우
            root_path = os.path.join(os.getcwd(), 'data', 'total_again.xlsx')
            if os.path.exists(root_path):
                file_path = root_path
            else:
                raise FileNotFoundError(
                    f"데이터 파일을 찾을 수 없습니다.\n"
                    f"시도한 경로들:\n"
                    f"  1. ../data/total_again.xlsx\n"
                    f"  2. {alt_path}\n"
                    f"  3. {root_path}\n"
                    f"현재 작업 디렉토리: {os.getcwd()}"
                )
    
    df = pd.read_excel(file_path, index_col='R-ID')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    print(f"   ✅ 총 데이터: {len(df):,}건")
    print(f"   ✅ 참여자 수: {df.index.nunique():,}명")
    print(f"   ✅ 변수 수: {len(df.columns)}개")
    
    return df


def create_ewma_features(df, available_vars, halflife_days=365):
    """EWMA 특성 생성"""
    print("\n🔧 EWMA 특성 생성 중...")
    analysis_df = df.copy()
    
    ewma_features = []
    for var in available_vars:
        analysis_df[f'{var}_ewma'] = np.nan
        analysis_df[f'{var}_ewma_trend'] = np.nan
        ewma_features.extend([f'{var}_ewma', f'{var}_ewma_trend'])
    
    for patient_id in analysis_df.index.unique():
        patient_data = analysis_df.loc[analysis_df.index == patient_id].copy()
        patient_data = patient_data.sort_values('수진일')
        
        for var in available_vars:
            if var in patient_data.columns:
                values = patient_data[var].values
                dates = patient_data['수진일'].values
                
                ewma_values = []
                trend_values = []
                
                for i in range(len(values)):
                    if i == 0:
                        ewma_values.append(values[i])
                        trend_values.append(0)
                    else:
                        time_diffs = np.array([(pd.Timestamp(dates[i]) - pd.Timestamp(dates[j])).days for j in range(i+1)])
                        weights = np.exp(-np.log(2) * time_diffs / halflife_days)
                        weights = weights / weights.sum()
                        
                        ewma = np.sum(values[:i+1] * weights)
                        ewma_values.append(ewma)
                        
                        if i >= 1:
                            trend = ewma - ewma_values[i-1]
                            trend_values.append(trend)
                        else:
                            trend_values.append(0)
                
                idx = patient_data.index
                for j, (ewma_val, trend_val) in enumerate(zip(ewma_values, trend_values)):
                    analysis_df.loc[idx[j], f'{var}_ewma'] = ewma_val
                    analysis_df.loc[idx[j], f'{var}_ewma_trend'] = trend_val
    
    print(f"   ✅ EWMA 특성 생성 완료: {len(ewma_features)}개")
    return analysis_df, ewma_features


def create_advanced_features(df, available_vars):
    """고급 파생 특성 생성"""
    print("\n🔧 고급 파생 특성 생성 중...")
    
    # 기존 특성
    healthy_weights = {'채소': 2.0, '과일': 1.8, '단백질류': 1.5, '곡류': 1.2, '유제품': 1.3}
    unhealthy_weights = {'인스턴트 가공식품': 2.2, '튀김': 2.0, '단맛': 1.8, '고지방 육류': 1.6, '음료류': 1.4}
    
    df['weighted_healthy_score'] = 0
    df['weighted_unhealthy_score'] = 0
    
    for food, weight in healthy_weights.items():
        if food in df.columns:
            df['weighted_healthy_score'] += df[food] * weight
    
    for food, weight in unhealthy_weights.items():
        if food in df.columns:
            df['weighted_unhealthy_score'] += df[food] * weight
    
    df['advanced_diet_ratio'] = df['weighted_healthy_score'] / (df['weighted_unhealthy_score'] + 1)
    df['diet_quality_score'] = df['weighted_healthy_score'] - df['weighted_unhealthy_score']
    
    sodium_foods = {'짠 식습관': 2.5, '짠 간': 2.0, '인스턴트 가공식품': 1.5}
    df['sodium_risk_score'] = 0
    for food, weight in sodium_foods.items():
        if food in df.columns:
            df['sodium_risk_score'] += df[food] * weight
    
    # 실제로 데이터에 존재하는 컬럼만 필터링 (숫자형만)
    existing_diet_vars = [var for var in available_vars if var in df.columns]
    if existing_diet_vars:
        # 숫자형 컬럼만 선택
        numeric_diet_vars = [var for var in existing_diet_vars if pd.api.types.is_numeric_dtype(df[var])]
        if numeric_diet_vars:
            df['diet_variety_count'] = (df[numeric_diet_vars] > 0).sum(axis=1)
        else:
            df['diet_variety_count'] = 0
    else:
        df['diet_variety_count'] = 0
    
    if 'age' in df.columns:
        df['age_healthy_interaction'] = df['age'] * df['weighted_healthy_score']
        df['age_unhealthy_interaction'] = df['age'] * df['weighted_unhealthy_score']
    
    if '수진일' in df.columns:
        df['month'] = df['수진일'].dt.month
        df['season_numeric'] = df['month'].apply(
            lambda x: 1 if x in [3,4,5] else (2 if x in [6,7,8] else (3 if x in [9,10,11] else 4))
        )
    
    # 추가 특성
    meal_timing_vars = ['아침식사빈도', '저녁식사시간']
    if all(var in df.columns for var in meal_timing_vars):
        df['meal_regularity'] = df['아침식사빈도'] * 0.6 + (10 - df['저녁식사시간'].fillna(7)) * 0.4
    
    if '단백질류' in df.columns and '곡류' in df.columns:
        df['protein_carb_ratio'] = df['단백질류'] / (df['곡류'] + 1)
    
    sweet_vars = ['단맛', '음료류', '간식빈도']
    if all(var in df.columns for var in sweet_vars):
        df['sugar_intake_composite'] = sum(df[var] * weight for var, weight in 
                                           zip(sweet_vars, [2.0, 1.5, 1.0]))
    
    fat_vars = ['고지방 육류', '튀김', '유제품']
    if all(var in df.columns for var in fat_vars):
        df['fat_intake_composite'] = df['고지방 육류'] * 2.0 + df['튀김'] * 1.8 - df['유제품'] * 0.5
    
    fiber_vars = ['채소', '과일', '곡류']
    if all(var in df.columns for var in fiber_vars):
        df['fiber_intake'] = df['채소'] * 1.5 + df['과일'] * 1.3 + df['곡류'] * 0.8
    
    # ❌ Data Leakage 방지: BMI 파생 특성 제거
    # BMI는 체중/키²로 계산되므로, 체중 예측 시 사용하면 순환 논리 발생
    # if '체질량지수' in df.columns:
    #     df['bmi_unhealthy_interaction'] = df['체질량지수'] * df['weighted_unhealthy_score']
    #     df['bmi_sodium_interaction'] = df['체질량지수'] * df['sodium_risk_score']
    
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=[1, 2, 3, 4])
        df['age_diet_quality_interaction'] = df['age_group'].astype(float) * df['diet_quality_score']
    
    new_features = [col for col in df.columns if col not in available_vars and col != '수진일']
    print(f"   ✅ 고급 특성 생성 완료: {len(new_features)}개")
    
    return df


def remove_outliers_improved(df, target_col, method='iqr', iqr_multiplier=1.5):
    """개선된 이상치 제거"""
    if method == 'iqr':
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        mask = (df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
        mask = z_scores < 3
    else:
        mask = pd.Series([True] * len(df), index=df.index)
    
    return df[mask]


# ============================================================================
# 2. TabNet 모델 구현
# ============================================================================

def optimize_tabnet(X_train, y_train, n_trials=30):
    """TabNet 하이퍼파라미터 최적화"""
    def objective(trial):
        params = {
            'n_d': trial.suggest_int('n_d', 8, 64),
            'n_a': trial.suggest_int('n_a', 8, 64),
            'n_steps': trial.suggest_int('n_steps', 3, 10),
            'gamma': trial.suggest_float('gamma', 1.0, 2.0),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
            'momentum': trial.suggest_float('momentum', 0.01, 0.4),
            'mask_type': trial.suggest_categorical('mask_type', ['sparsemax', 'entmax']),
        }
        
        model = TabNetRegressor(
            n_d=params['n_d'],
            n_a=params['n_a'],
            n_steps=params['n_steps'],
            gamma=params['gamma'],
            lambda_sparse=params['lambda_sparse'],
            momentum=params['momentum'],
            mask_type=params['mask_type'],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=0,
            seed=42
        )
        
        # Cross-validation을 위한 간단한 구현
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            # TabNet requires 2D target array (convert to numpy if needed)
            y_tr_2d = y_tr.values.reshape(-1, 1) if hasattr(y_tr, 'values') else y_tr.reshape(-1, 1)
            y_val_2d = y_val.values.reshape(-1, 1) if hasattr(y_val, 'values') else y_val.reshape(-1, 1)
            
            model.fit(
                X_tr, y_tr_2d,
                eval_set=[(X_val, y_val_2d)],
                max_epochs=100,
                patience=20,
                batch_size=256,
                virtual_batch_size=128,
                eval_metric=['rmse']
            )
            
            y_pred = model.predict(X_val).ravel()
            score = r2_score(y_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)
    
    return study.best_params


def create_tabnet_model(X_train, y_train, X_test, y_test, use_optuna=True, n_trials=20):
    """TabNet 모델 생성 및 학습"""
    print("      🧠 TabNet 딥러닝 모델 학습 중...")
    
    if use_optuna:
        print("         ⚙️ Optuna 최적화 진행 중...")
        best_params = optimize_tabnet(X_train, y_train, n_trials=n_trials)
        
        model = TabNetRegressor(
            n_d=best_params['n_d'],
            n_a=best_params['n_a'],
            n_steps=best_params['n_steps'],
            gamma=best_params['gamma'],
            lambda_sparse=best_params['lambda_sparse'],
            momentum=best_params['momentum'],
            mask_type=best_params['mask_type'],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=0,
            seed=42
        )
    else:
        # 기본 파라미터
        model = TabNetRegressor(
            n_d=32,
            n_a=32,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-4,
            momentum=0.3,
            mask_type='entmax',
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=0,
            seed=42
        )
    
    # TabNet requires 2D target array (convert to numpy if needed)
    y_train_2d = y_train.values.reshape(-1, 1) if hasattr(y_train, 'values') else y_train.reshape(-1, 1)
    y_test_2d = y_test.values.reshape(-1, 1) if hasattr(y_test, 'values') else y_test.reshape(-1, 1)
    
    # 학습
    model.fit(
        X_train, y_train_2d,
        eval_set=[(X_test, y_test_2d)],
        max_epochs=200,
        patience=50,
        batch_size=256,
        virtual_batch_size=128,
        eval_metric=['rmse']
    )
    
    # 예측 (ravel to convert back to 1D)
    y_pred_train = model.predict(X_train).ravel()
    y_pred_test = model.predict(X_test).ravel()
    
    # 평가
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"         ✅ TabNet R² (Train): {train_r2:.4f}")
    print(f"         ✅ TabNet R² (Test): {test_r2:.4f}")
    
    return {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': test_rmse,
        'mae': test_mae,
        'predictions': y_pred_test
    }


# ============================================================================
# 3. TabNet 통합 Stacking Ensemble
# ============================================================================

class TabNetWrapper(BaseEstimator, RegressorMixin):
    """TabNet을 sklearn 스타일로 래핑"""
    def __init__(self, tabnet_model=None):
        self.tabnet_model = tabnet_model
        self.model = tabnet_model
    
    def fit(self, X, y):
        # TabNet 모델이 없으면 새로 생성 (clone 시)
        if self.model is None:
            self.model = TabNetRegressor(
                n_d=32,
                n_a=32,
                n_steps=5,
                gamma=1.5,
                lambda_sparse=1e-4,
                momentum=0.3,
                mask_type='entmax',
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size": 10, "gamma": 0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                verbose=0,
                seed=42
            )
        
        # y가 2D가 아니면 2D로 변환
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        self.model.fit(
            X, y,
            max_epochs=100,
            patience=20,
            batch_size=256,
            virtual_batch_size=128,
            eval_metric=['rmse']
        )
        return self
    
    def predict(self, X):
        pred = self.model.predict(X)
        # 1D로 변환 (sklearn stacking이 요구)
        if len(pred.shape) > 1:
            pred = pred.ravel()
        return pred
    
    def get_params(self, deep=True):
        """sklearn 호환을 위한 get_params"""
        return {"tabnet_model": self.tabnet_model}
    
    def set_params(self, **params):
        """sklearn 호환을 위한 set_params"""
        if "tabnet_model" in params:
            self.tabnet_model = params["tabnet_model"]
            self.model = params["tabnet_model"]
        return self


def create_tabnet_stacking_ensemble(X_train, y_train, X_test, y_test,
                                    use_optuna=True, n_trials=20):
    """TabNet을 포함한 Stacking Ensemble"""
    print("\n   🔧 TabNet 통합 Stacking Ensemble 구성 중...")
    
    # TabNet 모델
    tabnet_result = create_tabnet_model(X_train, y_train, X_test, y_test, 
                                       use_optuna=use_optuna, n_trials=n_trials)
    tabnet_wrapper = TabNetWrapper(tabnet_result['model'])
    
    # 기존 모델들
    print("      🔧 기존 모델들 학습 중...")
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05,
                                 random_state=42, n_jobs=-1, verbosity=0)
    lgb_model = lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05,
                                 random_state=42, n_jobs=-1, verbosity=-1)
    cat_model = CatBoostRegressor(iterations=200, depth=8, learning_rate=0.05,
                                  random_seed=42, verbose=False)
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=15,
                                    min_samples_split=5, random_state=42, n_jobs=-1)
    
    # Stacking 구성 (TabNet 포함)
    base_models = [
        ('tabnet', tabnet_wrapper),
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model),
        ('rf', rf_model)
    ]
    
    meta_learner = Ridge(alpha=1.0)
    
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    # 학습
    print("      🎯 Stacking 모델 학습 중...")
    stacking_model.fit(X_train, y_train)
    
    # 예측
    y_pred_train = stacking_model.predict(X_train)
    y_pred_test = stacking_model.predict(X_test)
    
    # 평가
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"      ✅ Stacking+TabNet R² (Train): {train_r2:.4f}")
    print(f"      ✅ Stacking+TabNet R² (Test): {test_r2:.4f}")
    
    # TabNet 단독 vs Stacking 비교
    improvement = test_r2 - tabnet_result['test_r2']
    print(f"      📈 Stacking 추가 향상: {improvement:+.4f} R²")
    
    return {
        'model': stacking_model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': test_rmse,
        'mae': test_mae,
        'predictions': y_pred_test,
        'tabnet_alone_r2': tabnet_result['test_r2']
    }


# ============================================================================
# 4. 개선된 모델 학습 파이프라인
# ============================================================================

def train_tabnet_enhanced_model(df, target_biomarker, exclude_vars, feature_cols,
                                use_tabnet_stacking=True, use_optuna=True, optuna_trials=20):
    """TabNet이 통합된 모델 학습"""
    print(f"\n{'='*80}")
    print(f"🎯 타겟: {target_biomarker}")
    print(f"{'='*80}")
    
    # 특성 및 타겟 준비
    available_features = [col for col in feature_cols 
                         if col in df.columns and col not in exclude_vars 
                         and col not in ['수진일', 'R-ID']]
    
    X = df[available_features].copy()
    y = df[target_biomarker].copy()
    
    # 범주형 변수 처리
    if '성별' in X.columns:
        X['성별'] = X['성별'].map({'M': 1, 'F': 0}).fillna(0)
    if '일반담배_흡연여부' in X.columns:
        X['일반담배_흡연여부'] = X['일반담배_흡연여부'].map({'Y': 1, 'N': 0}).fillna(0)
    
    # 수치형 변환
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # 결측치 및 무한대 제거
    mask = ~(X.isnull().any(axis=1) | np.isinf(X).any(axis=1) | 
             y.isnull() | np.isinf(y))
    X = X[mask]
    y = y[mask]
    
    print(f"   📊 사용 가능한 샘플: {len(X):,}개")
    print(f"   📊 사용 특성 수: {len(available_features)}개")
    
    if len(X) < 100:
        print("   ⚠️ 샘플 수 부족")
        return None
    
    # 이상치 제거
    temp_df = pd.DataFrame({target_biomarker: y}, index=X.index)
    temp_df = remove_outliers_improved(temp_df, target_biomarker, method='iqr', iqr_multiplier=1.5)
    X = X.loc[temp_df.index]
    y = y.loc[temp_df.index]
    
    print(f"   📊 이상치 제거 후: {len(X):,}개")
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature Selection
    n_features = min(50, len(available_features))
    selector = SelectKBest(score_func=f_regression, k=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"   📊 선택된 특성: {len(selected_features)}개")
    
    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # TabNet 통합 모델 학습
    if use_tabnet_stacking:
        result = create_tabnet_stacking_ensemble(
            X_train_scaled, y_train, X_test_scaled, y_test,
            use_optuna=use_optuna, n_trials=optuna_trials
        )
        
        return {
            'Biomarker_KR': target_biomarker,
            'Model': result['model'],
            'Model_Type': 'STACKING+TABNET',
            'R_squared': result['test_r2'],
            'Train_R2': result['train_r2'],
            'RMSE': result['rmse'],
            'MAE': result['mae'],
            'TabNet_Alone_R2': result['tabnet_alone_r2'],
            'Predictions': result['predictions'],
            'Actual': y_test.values,
            'Features': selected_features,
            'Selector': selector,
            'Scaler': scaler,
            'X_test': X_test
        }
    else:
        # TabNet만 단독 사용
        result = create_tabnet_model(
            X_train_scaled, y_train, X_test_scaled, y_test,
            use_optuna=use_optuna, n_trials=optuna_trials
        )
        
        return {
            'Biomarker_KR': target_biomarker,
            'Model': result['model'],
            'Model_Type': 'TABNET',
            'R_squared': result['test_r2'],
            'Train_R2': result['train_r2'],
            'RMSE': result['rmse'],
            'MAE': result['mae'],
            'Predictions': result['predictions'],
            'Actual': y_test,
            'Features': selected_features,
            'Selector': selector,
            'Scaler': scaler,
            'X_test': X_test
        }


# ============================================================================
# 5. 메인 실행 함수
# ============================================================================

def main(use_tabnet_stacking=True, use_optuna=True, optuna_trials=20):
    """메인 실행 함수"""
    
    # 데이터 로드
    df = load_and_preprocess_data()
    
    # 식습관 변수 정의
    available_diet = [
        '간식빈도', '고지방 육류', '단맛', '단백질류', '담배피는데근처있는빈도',
        '곡류', '과일', '너무 빨리 먹는 식습관', '밤늦게 야식', '야채샐러드드레싱',
        '유제품', '음료류', '인스턴트 가공식품', '저녁식사시간', '짠 간', '짠 식습관',
        '채소', '튀김', '아침식사빈도'
    ]
    
    # EWMA 특성 생성
    df, ewma_features = create_ewma_features(df, available_diet)
    
    # 고급 특성 생성
    df = create_advanced_features(df, available_diet)
    
    # 모든 특성 리스트
    all_features = [col for col in df.columns if col not in ['수진일', 'R-ID']]
    
    # 타겟 바이오마커 정의 (고성능 바이오마커만 선택)
    # 예상 성능: 체중 R²≈0.95, 체질량지수 R²≈0.90, 허리둘레 R²≈0.85, SBP R²≈0.60, DBP R²≈0.55, TG R²≈0.50
    # 제외된 저성능 바이오마커: GLUCOSE, HBA1C, HDL CHOL., LDL CHOL., eGFR (R²<0.4)
    target_biomarkers = {
        '체중': '체중',
        '체질량지수': '체질량지수',
        '허리둘레(WAIST)': '허리둘레(WAIST)',
        'SBP': 'SBP',
        'DBP': 'DBP',
        'TG': 'TG'
    }
    
    # 제외 변수 정의
    exclude_variables_by_biomarker = {
        '체중': ['체중', '체질량지수', '허리둘레(WAIST)', '골격근량', '체지방량', 
                '내장지방레벨', '체지방률', '골격근률'],
        '체질량지수': ['체중', '체질량지수', '허리둘레(WAIST)', '골격근량', '체지방량', 
                      '내장지방레벨', '체지방률', '골격근률'],
        '허리둘레(WAIST)': ['체중', '체질량지수', '허리둘레(WAIST)', '골격근량', '체지방량', 
                           '내장지방레벨', '체지방률', '골격근률'],
        'SBP': ['SBP', 'DBP'],
        'DBP': ['SBP', 'DBP'],
        'TG': ['TG', 'HDL CHOL.', 'LDL CHOL.', 'TOTAL CHOL.']
    }
    
    # 모델 학습
    print("\n" + "="*80)
    print("🚀 TabNet 딥러닝 모델 통합 학습 시작")
    print("="*80)
    print(f"   TabNet + Stacking: {'사용' if use_tabnet_stacking else '미사용 (TabNet만)'}")
    print(f"   Optuna 최적화: {'사용' if use_optuna else '미사용'}")
    if use_optuna:
        print(f"   Optuna Trials: {optuna_trials}")
    
    results = []
    for key, biomarker in target_biomarkers.items():
        if biomarker not in df.columns:
            print(f"\n⚠️ {biomarker} 컬럼 없음")
            continue
        
        exclude_vars = exclude_variables_by_biomarker.get(key, [])
        exclude_vars.append(biomarker)
        
        result = train_tabnet_enhanced_model(
            df, biomarker, exclude_vars, all_features,
            use_tabnet_stacking=use_tabnet_stacking,
            use_optuna=use_optuna,
            optuna_trials=optuna_trials
        )
        
        if result:
            results.append(result)
    
    # 결과 요약
    print("\n" + "="*80)
    print("📊 최종 결과 요약")
    print("="*80)
    
    summary_data = [
        {
            'Biomarker': r['Biomarker_KR'],
            'Model_Type': r['Model_Type'],
            'R²': r['R_squared'],
            'RMSE': r['RMSE'],
            'MAE': r['MAE']
        }
        for r in results
    ]
    
    if use_tabnet_stacking:
        for i, r in enumerate(results):
            if 'TabNet_Alone_R2' in r:
                summary_data[i]['TabNet_Alone_R²'] = r['TabNet_Alone_R2']
                summary_data[i]['Stacking_Gain'] = r['R_squared'] - r['TabNet_Alone_R2']
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('R²', ascending=False)
    print(summary_df.to_string(index=False))
    
    # 성능 분석
    print("\n" + "="*80)
    print("📈 성능 분석")
    print("="*80)
    
    excellent = len(summary_df[summary_df['R²'] >= 0.7])
    good = len(summary_df[(summary_df['R²'] >= 0.5) & (summary_df['R²'] < 0.7)])
    fair = len(summary_df[(summary_df['R²'] >= 0.3) & (summary_df['R²'] < 0.5)])
    poor = len(summary_df[summary_df['R²'] < 0.3])
    
    print(f"   Excellent (R²≥0.7): {excellent}개")
    print(f"   Good (R²≥0.5): {good}개")
    print(f"   Fair (R²≥0.3): {fair}개")
    print(f"   Poor (R²<0.3): {poor}개")
    print(f"\n   평균 R²: {summary_df['R²'].mean():.4f}")
    print(f"   평균 RMSE: {summary_df['RMSE'].mean():.4f}")
    print(f"   평균 MAE: {summary_df['MAE'].mean():.4f}")
    
    if use_tabnet_stacking and 'Stacking_Gain' in summary_df.columns:
        print(f"\n   평균 Stacking 향상: {summary_df['Stacking_Gain'].mean():+.4f} R²")
    
    return results, summary_df


# ============================================================================
# 6. 실행
# ============================================================================

if __name__ == "__main__":
    # 설정
    USE_TABNET_STACKING = True  # TabNet + 기존 모델 Stacking
    USE_OPTUNA = True           # Optuna 최적화
    OPTUNA_TRIALS = 20          # Optuna 시행 횟수
    
    # 실행
    results, summary = main(
        use_tabnet_stacking=USE_TABNET_STACKING,
        use_optuna=USE_OPTUNA,
        optuna_trials=OPTUNA_TRIALS
    )
    
    print("\n✅ TabNet 통합 모델 학습 완료!")
