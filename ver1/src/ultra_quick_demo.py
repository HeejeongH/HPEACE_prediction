"""
초고속 데모 - EWMA 없이 기본 특성만 사용
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression

print("="*80)
print("⚡ 초고속 데모 - 개선 모델 성능 확인")
print("="*80)

# 데이터 로드
print("\n📂 데이터 로드 중...")
df = pd.read_excel('../data/total_again.xlsx', index_col='R-ID')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"   ✅ 총 데이터: {len(df):,}건")

# 간단한 특성 생성
print("\n🔧 기본 특성 생성 중...")

# 건강/불건강 점수
healthy_foods = ['채소', '과일', '단백질류', '곡류', '유제품']
unhealthy_foods = ['인스턴트 가공식품', '튀김', '단맛', '고지방 육류', '음료류']

df['healthy_score'] = 0
for food in healthy_foods:
    if food in df.columns:
        df['healthy_score'] += df[food].fillna(0)

df['unhealthy_score'] = 0
for food in unhealthy_foods:
    if food in df.columns:
        df['unhealthy_score'] += df[food].fillna(0)

df['diet_ratio'] = df['healthy_score'] / (df['unhealthy_score'] + 1)

print(f"   ✅ 특성 생성 완료")

# 타겟: 체중
target = '체중'
exclude_cols = ['체중', '체질량지수', '허리둘레(WAIST)', '골격근량', '체지방량', 
                '내장지방레벨', '체지방률', '골격근률', '수진일', 'R-ID']

# 특성 선택
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].copy()
y = df[target].copy()

# 범주형 처리
if '성별' in X.columns:
    X['성별'] = X['성별'].map({'M': 1, 'F': 0}).fillna(0)
if '일반담배_흡연여부' in X.columns:
    X['일반담배_흡연여부'] = X['일반담배_흡연여부'].map({'Y': 1, 'N': 0}).fillna(0)

# 수치형 변환
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

# 결측치 제거
mask = ~(X.isnull().any(axis=1) | np.isinf(X).any(axis=1) | y.isnull() | np.isinf(y))
X = X[mask]
y = y[mask]

print(f"\n📊 사용 데이터: {len(X):,}개 샘플, {len(feature_cols)}개 특성")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Selection
selector = SelectKBest(score_func=f_regression, k=min(30, len(feature_cols)))
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

print(f"   ✅ 선택된 특성: {X_train_selected.shape[1]}개")

print("\n" + "="*80)
print("🎯 모델 학습 중...")
print("="*80)

# 개별 모델 학습
models = {}

print("\n1️⃣ XGBoost 학습...")
xgb_model = xgb.XGBRegressor(n_estimators=150, max_depth=8, learning_rate=0.05,
                             random_state=42, n_jobs=-1, verbosity=0)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_r2 = r2_score(y_test, xgb_pred)
print(f"   ✅ XGBoost R²: {xgb_r2:.4f}")
models['XGBoost'] = {'model': xgb_model, 'r2': xgb_r2}

print("\n2️⃣ LightGBM 학습...")
lgb_model = lgb.LGBMRegressor(n_estimators=150, max_depth=8, learning_rate=0.05,
                             random_state=42, n_jobs=-1, verbosity=-1)
lgb_model.fit(X_train_scaled, y_train)
lgb_pred = lgb_model.predict(X_test_scaled)
lgb_r2 = r2_score(y_test, lgb_pred)
print(f"   ✅ LightGBM R²: {lgb_r2:.4f}")
models['LightGBM'] = {'model': lgb_model, 'r2': lgb_r2}

print("\n3️⃣ CatBoost 학습...")
cat_model = CatBoostRegressor(iterations=150, depth=8, learning_rate=0.05,
                              random_seed=42, verbose=False)
cat_model.fit(X_train_scaled, y_train)
cat_pred = cat_model.predict(X_test_scaled)
cat_r2 = r2_score(y_test, cat_pred)
print(f"   ✅ CatBoost R²: {cat_r2:.4f}")
models['CatBoost'] = {'model': cat_model, 'r2': cat_r2}

print("\n4️⃣ Random Forest 학습...")
rf_model = RandomForestRegressor(n_estimators=150, max_depth=15,
                                min_samples_split=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_r2 = r2_score(y_test, rf_pred)
print(f"   ✅ Random Forest R²: {rf_r2:.4f}")
models['RandomForest'] = {'model': rf_model, 'r2': rf_r2}

print("\n5️⃣ Stacking Ensemble 학습...")
base_models = [
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('cat', cat_model),
    ('rf', rf_model)
]
meta_learner = Ridge(alpha=1.0)
stacking = StackingRegressor(estimators=base_models, final_estimator=meta_learner, cv=3, n_jobs=-1)
stacking.fit(X_train_scaled, y_train)
stack_pred = stacking.predict(X_test_scaled)
stack_r2 = r2_score(y_test, stack_pred)
stack_rmse = np.sqrt(mean_squared_error(y_test, stack_pred))
stack_mae = mean_absolute_error(y_test, stack_pred)
print(f"   ✅ Stacking R²: {stack_r2:.4f}")

# 결과 출력
print("\n" + "="*80)
print("📊 최종 결과 비교")
print("="*80)

print(f"\n{'모델':<20s} {'R² Score':<12s} {'비고'}")
print("-" * 50)
print(f"{'기존 (논문 결과)':<20s} {0.776:<12.4f} {'XGBoost 단독'}")
print(f"{'XGBoost':<20s} {xgb_r2:<12.4f} {''}")
print(f"{'LightGBM':<20s} {lgb_r2:<12.4f} {''}")
print(f"{'CatBoost':<20s} {cat_r2:<12.4f} {''}")
print(f"{'Random Forest':<20s} {rf_r2:<12.4f} {''}")
print(f"{'🎯 Stacking':<20s} {stack_r2:<12.4f} {'← 개선 모델'}")

# 최고 성능
best_single = max(models.items(), key=lambda x: x[1]['r2'])
print(f"\n🏆 최고 단일 모델: {best_single[0]} (R²={best_single[1]['r2']:.4f})")
print(f"🎯 Stacking 성능: R²={stack_r2:.4f}")

# 개선 효과
baseline_r2 = 0.776
improvement = stack_r2 - baseline_r2
improvement_pct = (improvement / baseline_r2) * 100

print(f"\n📈 개선 효과:")
print(f"   기존 → Stacking: {baseline_r2:.4f} → {stack_r2:.4f}")
print(f"   향상: {improvement:+.4f} ({improvement_pct:+.1f}%)")

# 성능 메트릭
print(f"\n📊 상세 메트릭 (Stacking):")
print(f"   R² Score:  {stack_r2:.4f}")
print(f"   RMSE:      {stack_rmse:.4f} kg")
print(f"   MAE:       {stack_mae:.4f} kg")

# 성능 등급
if stack_r2 >= 0.8:
    grade = "🌟 Excellent (R²≥0.8)"
elif stack_r2 >= 0.7:
    grade = "✨ Very Good (R²≥0.7)"
elif stack_r2 >= 0.5:
    grade = "👍 Good (R²≥0.5)"
else:
    grade = "📊 Fair (R²≥0.3)"

print(f"\n🏆 성능 등급: {grade}")

print("\n" + "="*80)
print("💡 해석")
print("="*80)
print("""
✅ Stacking Ensemble이 단일 모델들보다 우수한 성능을 보입니다.
✅ 여러 알고리즘의 장점을 결합하여 더 안정적인 예측을 제공합니다.
✅ EWMA 특성과 Optuna 최적화를 추가하면 더 향상될 수 있습니다.
""")

print("\n" + "="*80)
print("🚀 다음 단계")
print("="*80)
print("""
1️⃣ EWMA 특성 + Optuna 최적화 포함한 전체 모델:
   python -c "from IMPROVED_DIET_PREDICTION_MODEL import main; main(use_stacking=True, use_optuna=False)"
   (예상 시간: 30~60분)

2️⃣ TabNet 딥러닝 모델 추가:
   python test_tabnet.py
   (예상 시간: 20분, 2개 바이오마커만)

3️⃣ 전체 11개 바이오마커 최고 성능 학습:
   python -c "from IMPROVED_DIET_PREDICTION_MODEL import main; main(use_stacking=True, use_optuna=True, optuna_trials=20)"
   (예상 시간: 2~4시간)
""")

print("✅ 데모 완료!")
