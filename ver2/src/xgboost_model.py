"""
Ver2: XGBoost Baseline Model
============================

목적: 식습관 변화 → 건강지표 변화 예측 (Baseline)
방법: XGBoost Regressor
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class XGBoostChangePredictor:
    """XGBoost 기반 변화 예측 모델"""
    
    def __init__(self, target_variable, random_state=42):
        """
        Args:
            target_variable: 예측할 건강지표 (예: '체중', '혈당')
            random_state: 재현성을 위한 랜덤 시드
        """
        self.target_variable = target_variable
        self.random_state = random_state
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        
    def prepare_data(self, df):
        """데이터 준비 - 추가 특성 엔지니어링으로 성능 개선"""
        print(f"\n{'='*80}")
        print(f"📊 [{self.target_variable}] 데이터 준비 (개선 버전)")
        print(f"{'='*80}")
        
        # 1. 식습관 변화 특성
        diet_change_cols = [col for col in df.columns 
                           if '_change' in col and '건강' not in col 
                           and not any(bio in col for bio in ['체중', '체질량지수', '허리둘레', 'SBP', 'DBP', 'TG'])]
        
        # 2. ✅ 다른 건강지표 baseline 추가
        health_indicators = ['체중', '체질량지수', '허리둘레(WAIST)', 'SBP', 'DBP', 'TG']
        other_health_baselines = []
        
        for indicator in health_indicators:
            if indicator != self.target_variable:
                baseline_col = f'{indicator}_baseline'
                if baseline_col in df.columns:
                    other_health_baselines.append(baseline_col)
        
        print(f"\n   📈 추가된 다른 건강지표 baseline: {len(other_health_baselines)}개")
        
        # 3. ✅ 파생 특성 생성
        df_temp = df.copy()
        
        if '체질량지수_baseline' in df_temp.columns:
            df_temp['BMI_category'] = pd.cut(
                df_temp['체질량지수_baseline'], 
                bins=[0, 18.5, 23, 25, 30, 100],
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
        
        metabolic_risk_score = 0
        if '체질량지수_baseline' in df_temp.columns:
            metabolic_risk_score += (df_temp['체질량지수_baseline'] >= 25).astype(int)
        if 'SBP_baseline' in df_temp.columns:
            metabolic_risk_score += (df_temp['SBP_baseline'] >= 130).astype(int)
        if 'DBP_baseline' in df_temp.columns:
            metabolic_risk_score += (df_temp['DBP_baseline'] >= 85).astype(int)
        if 'TG_baseline' in df_temp.columns:
            metabolic_risk_score += (df_temp['TG_baseline'] >= 150).astype(int)
        df_temp['metabolic_risk_score'] = metabolic_risk_score
        
        healthy_items = ['채소_change', '과일_change', '단백질류_change', '유제품_change', '곡류_change']
        healthy_score = sum(df_temp[item] for item in healthy_items if item in df_temp.columns)
        df_temp['healthy_eating_score'] = healthy_score
        
        unhealthy_items = ['간식빈도_change', '고지방 육류_change', '단맛_change', 
                          '음료류_change', '인스턴트 가공식품_change', '짠 간_change', 
                          '짠 식습관_change', '튀김_change']
        unhealthy_score = sum(df_temp[item] for item in unhealthy_items if item in df_temp.columns)
        df_temp['unhealthy_eating_score'] = unhealthy_score
        
        df_temp['net_diet_improvement'] = df_temp['healthy_eating_score'] - df_temp['unhealthy_eating_score']
        
        # 4. 전체 특성
        additional_features = ['time_gap_days']
        derived_features = []
        
        for feat in ['BMI_category', 'metabolic_risk_score', 'healthy_eating_score', 
                     'unhealthy_eating_score', 'net_diet_improvement']:
            if feat in df_temp.columns:
                derived_features.append(feat)
        
        feature_cols = diet_change_cols + other_health_baselines + additional_features + derived_features
        self.feature_names = feature_cols
        
        target_col = f'{self.target_variable}_change'
        
        valid_idx = df_temp[feature_cols + [target_col]].notna().all(axis=1)
        df_clean = df_temp[valid_idx].copy()
        
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        
        print(f"\n   ✅ 유효 샘플: {len(df_clean):,}개")
        print(f"   ✅ 총 특성 개수: {len(feature_cols)}개")
        print(f"      - 식습관 변화: {len(diet_change_cols)}개")
        print(f"      - 다른 건강지표 baseline: {len(other_health_baselines)}개")
        print(f"      - 파생 특성: {len(derived_features)}개")
        
        # CSV 저장
        features_df = pd.DataFrame({
            'Feature_Index': range(1, len(feature_cols)+1),
            'Feature_Name': feature_cols
        })
        features_csv = f'./result/features_used_XGBoost_{self.target_variable}.csv'
        Path(features_csv).parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(features_csv, index=False, encoding='utf-8-sig')
        print(f"   💾 특성 목록 저장: {features_csv}")
        
        # Leakage 검증
        target_baseline = f'{self.target_variable}_baseline'
        if target_baseline in feature_cols:
            raise ValueError(f"Data Leakage: {target_baseline} in features")
        else:
            print(f"   ✅ 타겟 baseline 제외됨")
        
        return X, y, df_clean
    
    def train(self, X, y, test_size=0.2, val_size=0.1):
        """모델 학습"""
        print(f"\n{'='*80}")
        print(f"🎯 [{self.target_variable}] 모델 학습")
        print(f"{'='*80}")
        
        # Train / Validation / Test 분할
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
        )
        
        print(f"   📊 Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        
        # 스케일링
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # XGBoost 모델
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            early_stopping_rounds=20,
            eval_metric='rmse'
        )
        
        # 학습
        print("\n   🔄 학습 중...")
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        # 평가
        self._evaluate(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
        
        return X_test_scaled, y_test
    
    def _evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """모델 평가"""
        print(f"\n   📈 성능 평가:")
        
        datasets = {
            'Train': (X_train, y_train),
            'Val': (X_val, y_val),
            'Test': (X_test, y_test)
        }
        
        for name, (X, y) in datasets.items():
            y_pred = self.model.predict(X)
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            # 방향 정확도 (개선/악화 방향 맞춤)
            direction_acc = np.mean(np.sign(y) == np.sign(y_pred)) * 100
            
            self.metrics[name] = {
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'Direction_Accuracy': direction_acc
            }
            
            print(f"\n      [{name}]")
            print(f"         R² = {r2:.4f}")
            print(f"         RMSE = {rmse:.4f}")
            print(f"         MAE = {mae:.4f}")
            print(f"         방향 정확도 = {direction_acc:.1f}%")
    
    def plot_feature_importance(self, top_n=20):
        """특성 중요도 시각화"""
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importance[indices])
        plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'{self.target_variable} 변화 예측 - 특성 중요도 (Top {top_n})', fontsize=14)
        plt.tight_layout()
        
        output_path = f'./result/xgboost_{self.target_variable}_feature_importance.png'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n   💾 특성 중요도 저장: {output_path}")
        plt.close()
    
    def plot_predictions(self, X_test, y_test):
        """예측 결과 시각화"""
        y_pred = self.model.predict(X_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Scatter plot
        axes[0].scatter(y_test, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel(f'실제 {self.target_variable} 변화', fontsize=12)
        axes[0].set_ylabel(f'예측 {self.target_variable} 변화', fontsize=12)
        axes[0].set_title(f'예측 vs 실제 (Test R² = {self.metrics["Test"]["R²"]:.4f})', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel(f'예측 {self.target_variable} 변화', fontsize=12)
        axes[1].set_ylabel('잔차 (실제 - 예측)', fontsize=12)
        axes[1].set_title('잔차 분포', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'./result/xgboost_{self.target_variable}_predictions.png'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   💾 예측 결과 저장: {output_path}")
        plt.close()
    
    def save_model(self, output_dir='./result/models'):
        """모델 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f'xgboost_{self.target_variable}.pkl')
        scaler_X_path = os.path.join(output_dir, f'scaler_X_{self.target_variable}.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler_X, scaler_X_path)
        
        print(f"\n   💾 모델 저장: {model_path}")
        print(f"   💾 스케일러 저장: {scaler_X_path}")
    
    def load_model(self, model_dir='./result/models'):
        """모델 로드"""
        model_path = os.path.join(model_dir, f'xgboost_{self.target_variable}.pkl')
        scaler_X_path = os.path.join(model_dir, f'scaler_X_{self.target_variable}.pkl')
        
        self.model = joblib.load(model_path)
        self.scaler_X = joblib.load(scaler_X_path)
        
        print(f"   ✅ 모델 로드: {model_path}")


def train_all_targets(data_path='../data/ver2_paired_visits.csv'):
    """모든 건강지표에 대해 모델 학습"""
    print("\n" + "="*80)
    print("🚀 Ver2 XGBoost 전체 학습 시작")
    print("="*80)
    
    # 데이터 로드
    df = pd.read_csv(data_path)
    print(f"\n✅ 데이터 로드 완료: {len(df):,}개 샘플")
    
    # 건강지표 목록 (데이터에 실제 존재하는 컬럼명 사용)
    health_indicators = [
        '체중', '체질량지수', '허리둘레(WAIST)', 'SBP', 'DBP', 'TG'
    ]
    
    results = {}
    
    for indicator in health_indicators:
        try:
            print(f"\n{'='*80}")
            print(f"🎯 [{indicator}] 학습 시작")
            print(f"{'='*80}")
            
            # 모델 생성 및 학습
            model = XGBoostChangePredictor(indicator)
            X, y, df_clean = model.prepare_data(df)
            X_test, y_test = model.train(X, y)
            
            # 시각화
            model.plot_feature_importance()
            model.plot_predictions(X_test, y_test)
            
            # 모델 저장
            model.save_model()
            
            # 결과 저장
            results[indicator] = model.metrics['Test']
            
            print(f"\n✅ [{indicator}] 완료!")
            
        except Exception as e:
            print(f"\n❌ [{indicator}] 오류: {str(e)}")
            results[indicator] = None
    
    # 전체 결과 요약
    print("\n" + "="*80)
    print("📊 전체 결과 요약")
    print("="*80)
    
    # None 값 제거 (실패한 지표 제외)
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) > 0:
        results_df = pd.DataFrame(valid_results).T
        print("\n", results_df.round(4))
        
        # 결과 저장
        output_csv = './result/xgboost_all_results.csv'
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv)
        print(f"\n💾 전체 결과 저장: {output_csv}")
    else:
        print("\n⚠️ 모든 지표에서 오류가 발생했습니다.")
        results_df = pd.DataFrame()
    
    return results_df


if __name__ == '__main__':
    # Ver2 데이터로 전체 학습
    results = train_all_targets()
