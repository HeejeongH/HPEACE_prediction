"""
Ver3: 건강지표 변화 예측 모델
================================

목표: 식습관 변화 + Baseline 정보를 활용하여 건강지표 변화 예측

모델 구조:
1. TabNet (딥러닝)
2. XGBoost, LightGBM, CatBoost (앙상블)
3. Stacking 메타 학습

저자: SNUH Prediction Team
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import warnings
import joblib
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class HealthIndicatorPredictor:
    """건강지표 변화 예측 모델 클래스"""
    
    def __init__(self, random_state: int = 42):
        """
        Parameters
        ----------
        random_state : int
            랜덤 시드
        """
        self.random_state = random_state
        self.scalers = {}
        self.models = {}
        self.feature_importances = {}
        self.results = {}
        
        # 예측할 건강지표
        self.target_vars = [
            '체중_change',
            '체질량지수_change',
            '허리둘레(WAIST)_change',
            'SBP_change',
            'DBP_change',
            'TG_change'
        ]
        
    def prepare_features(self, df: pd.DataFrame, target: str) -> Tuple:
        """
        특성 및 타겟 준비
        
        Parameters
        ----------
        df : DataFrame
            전처리된 paired visits 데이터
        target : str
            예측할 건강지표 (예: '체중_change')
            
        Returns
        -------
        X : DataFrame
            특성 데이터
        y : Series
            타겟 데이터
        feature_names : List[str]
            특성 이름 리스트
        """
        # 타겟 추출
        y = df[target].copy()
        
        # 특성 선택
        # 1. 식습관 baseline
        diet_baseline = [col for col in df.columns if col.endswith('_baseline') 
                        and not col.startswith(('체중', '체질량지수', '허리둘레', 'SBP', 'DBP', 'TG', 'HDL', 'glucose', 'HbA1c'))]
        
        # 2. 식습관 change
        diet_change = [col for col in df.columns if '_change' in col 
                      and not col.startswith(('체중', '체질량지수', '허리둘레', 'SBP', 'DBP', 'TG', 'HDL', 'glucose', 'HbA1c', 'mets', 'monthly'))]
        
        # 3. 건강지표 baseline (타겟 제외)
        health_baseline = [col for col in df.columns if col.endswith('_baseline') 
                          and col.startswith(('체중', '체질량지수', '허리둘레', 'SBP', 'DBP', 'TG', 'HDL', 'glucose'))]
        
        # 4. 인구통계학적 변수
        demographic = ['sex', 'age_baseline', 'time_gap_days']
        
        # 5. 고급 특성
        advanced = ['healthy_score_baseline', 'healthy_score_change',
                   'unhealthy_score_baseline', 'unhealthy_score_change',
                   'diet_improvement_score', 'diet_diversity_baseline',
                   'baseline_risk']
        
        # 전체 특성 결합
        feature_cols = []
        for col_list in [diet_baseline, diet_change, health_baseline, demographic, advanced]:
            feature_cols.extend([col for col in col_list if col in df.columns])
        
        # 중복 제거
        feature_cols = list(dict.fromkeys(feature_cols))
        
        # 타겟의 baseline과 change는 제외 (data leakage 방지)
        target_base = target.replace('_change', '_baseline')
        if target_base in feature_cols:
            feature_cols.remove(target_base)
        
        # 특성 데이터 생성
        X = df[feature_cols].copy()
        
        # 범주형 변수 인코딩 (sex)
        if 'sex' in X.columns:
            X['sex'] = X['sex'].map({'M': 1, 'F': 0})
        
        # 결측치 처리
        X = X.fillna(X.median())
        
        return X, y, feature_cols
    
    def build_tabnet_model(self, input_dim: int) -> TabNetRegressor:
        """TabNet 모델 생성"""
        model = TabNetRegressor(
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            lambda_sparse=1e-4,
            momentum=0.3,
            clip_value=2.0,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"gamma": 0.95, "step_size": 20},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            verbose=0,
            seed=self.random_state
        )
        return model
    
    def build_xgboost_model(self) -> xgb.XGBRegressor:
        """XGBoost 모델 생성"""
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        return model
    
    def build_lightgbm_model(self) -> lgb.LGBMRegressor:
        """LightGBM 모델 생성"""
        model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        return model
    
    def build_catboost_model(self) -> CatBoostRegressor:
        """CatBoost 모델 생성"""
        model = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_state=self.random_state,
            verbose=0
        )
        return model
    
    def train_single_target(self, 
                           df: pd.DataFrame, 
                           target: str,
                           use_ensemble: bool = True) -> Dict:
        """
        단일 건강지표에 대한 모델 학습
        
        Parameters
        ----------
        df : DataFrame
            전처리된 데이터
        target : str
            예측할 건강지표
        use_ensemble : bool
            앙상블 사용 여부
            
        Returns
        -------
        result : Dict
            학습 결과 및 모델 정보
        """
        print(f"\n{'='*80}")
        print(f"🎯 Target: {target}")
        print(f"{'='*80}")
        
        # 데이터 준비
        X, y, feature_names = self.prepare_features(df, target)
        
        print(f"   특성 수: {X.shape[1]}개")
        print(f"   샘플 수: {len(X):,}개")
        
        # Train/Test 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target] = scaler
        
        # 모델 학습
        models = {}
        predictions_train = {}
        predictions_test = {}
        
        # 1. TabNet
        print(f"\n   📊 Training TabNet...")
        tabnet = self.build_tabnet_model(X_train.shape[1])
        tabnet.fit(
            X_train_scaled, y_train.values,
            eval_set=[(X_test_scaled, y_test.values)],
            max_epochs=100,
            patience=20,
            batch_size=256,
            virtual_batch_size=128,
            eval_metric=['rmse']
        )
        
        models['TabNet'] = tabnet
        predictions_train['TabNet'] = tabnet.predict(X_train_scaled)
        predictions_test['TabNet'] = tabnet.predict(X_test_scaled)
        
        tabnet_r2 = r2_score(y_test, predictions_test['TabNet'])
        tabnet_rmse = np.sqrt(mean_squared_error(y_test, predictions_test['TabNet']))
        print(f"      R² = {tabnet_r2:.4f}, RMSE = {tabnet_rmse:.4f}")
        
        if use_ensemble:
            # 2. XGBoost
            print(f"   📊 Training XGBoost...")
            xgb_model = self.build_xgboost_model()
            xgb_model.fit(X_train, y_train)
            models['XGBoost'] = xgb_model
            predictions_train['XGBoost'] = xgb_model.predict(X_train)
            predictions_test['XGBoost'] = xgb_model.predict(X_test)
            
            xgb_r2 = r2_score(y_test, predictions_test['XGBoost'])
            print(f"      R² = {xgb_r2:.4f}")
            
            # 3. LightGBM
            print(f"   📊 Training LightGBM...")
            lgb_model = self.build_lightgbm_model()
            lgb_model.fit(X_train, y_train)
            models['LightGBM'] = lgb_model
            predictions_train['LightGBM'] = lgb_model.predict(X_train)
            predictions_test['LightGBM'] = lgb_model.predict(X_test)
            
            lgb_r2 = r2_score(y_test, predictions_test['LightGBM'])
            print(f"      R² = {lgb_r2:.4f}")
            
            # 4. CatBoost
            print(f"   📊 Training CatBoost...")
            cat_model = self.build_catboost_model()
            cat_model.fit(X_train, y_train)
            models['CatBoost'] = cat_model
            predictions_train['CatBoost'] = cat_model.predict(X_train)
            predictions_test['CatBoost'] = cat_model.predict(X_test)
            
            cat_r2 = r2_score(y_test, predictions_test['CatBoost'])
            print(f"      R² = {cat_r2:.4f}")
            
            # 5. Stacking (메타 학습)
            print(f"\n   🔗 Stacking Ensemble...")
            
            # 메타 특성 생성
            meta_train = np.column_stack([predictions_train[name] for name in models.keys()])
            meta_test = np.column_stack([predictions_test[name] for name in models.keys()])
            
            # 메타 모델 (Ridge)
            meta_model = Ridge(alpha=1.0)
            meta_model.fit(meta_train, y_train)
            
            models['Stacking'] = meta_model
            final_pred_train = meta_model.predict(meta_train)
            final_pred_test = meta_model.predict(meta_test)
            
            stacking_r2 = r2_score(y_test, final_pred_test)
            stacking_rmse = np.sqrt(mean_squared_error(y_test, final_pred_test))
            print(f"      R² = {stacking_r2:.4f}, RMSE = {stacking_rmse:.4f}")
            
        else:
            final_pred_train = predictions_train['TabNet']
            final_pred_test = predictions_test['TabNet']
        
        # 성능 평가
        train_r2 = r2_score(y_train, final_pred_train)
        test_r2 = r2_score(y_test, final_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, final_pred_test))
        test_mae = mean_absolute_error(y_test, final_pred_test)
        
        print(f"\n   ✅ 최종 성능:")
        print(f"      Train R² = {train_r2:.4f}")
        print(f"      Test R²  = {test_r2:.4f}")
        print(f"      RMSE     = {test_rmse:.4f}")
        print(f"      MAE      = {test_mae:.4f}")
        
        # 특성 중요도 (TabNet)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': tabnet.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n   📊 Top 10 중요 특성:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
        
        # 결과 저장
        result = {
            'target': target,
            'models': models,
            'scaler': scaler,
            'feature_names': feature_names,
            'feature_importance': feature_importance,
            'performance': {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'rmse': test_rmse,
                'mae': test_mae
            },
            'predictions': {
                'y_train': y_train,
                'y_test': y_test,
                'pred_train': final_pred_train,
                'pred_test': final_pred_test
            }
        }
        
        self.models[target] = models
        self.feature_importances[target] = feature_importance
        self.results[target] = result
        
        return result
    
    def train_all_targets(self, 
                         df: pd.DataFrame,
                         use_ensemble: bool = True) -> Dict:
        """
        모든 건강지표에 대한 모델 학습
        
        Parameters
        ----------
        df : DataFrame
            전처리된 데이터
        use_ensemble : bool
            앙상블 사용 여부
            
        Returns
        -------
        all_results : Dict
            모든 타겟의 학습 결과
        """
        print("\n" + "="*80)
        print("🚀 Ver3: 건강지표 변화 예측 모델 학습 시작")
        print("="*80)
        
        all_results = {}
        
        for target in self.target_vars:
            if target not in df.columns:
                print(f"\n⚠️ Warning: {target} not found in data, skipping...")
                continue
            
            result = self.train_single_target(df, target, use_ensemble)
            all_results[target] = result
        
        # 전체 요약
        print("\n" + "="*80)
        print("📊 전체 모델 성능 요약")
        print("="*80)
        
        summary_df = pd.DataFrame([
            {
                'Target': target,
                'Train R²': result['performance']['train_r2'],
                'Test R²': result['performance']['test_r2'],
                'RMSE': result['performance']['rmse'],
                'MAE': result['performance']['mae']
            }
            for target, result in all_results.items()
        ])
        
        print(summary_df.to_string(index=False))
        
        avg_test_r2 = summary_df['Test R²'].mean()
        print(f"\n✅ 평균 Test R² = {avg_test_r2:.4f}")
        
        return all_results
    
    def predict(self, df: pd.DataFrame, target: str) -> np.ndarray:
        """
        새로운 데이터에 대한 예측
        
        Parameters
        ----------
        df : DataFrame
            예측할 데이터
        target : str
            예측할 건강지표
            
        Returns
        -------
        predictions : np.ndarray
            예측 결과
        """
        if target not in self.models:
            raise ValueError(f"Model for {target} not trained yet!")
        
        X, _, _ = self.prepare_features(df, target)
        X_scaled = self.scalers[target].transform(X)
        
        # Stacking 모델이 있으면 사용
        if 'Stacking' in self.models[target]:
            # 각 기본 모델의 예측
            predictions = []
            for name, model in self.models[target].items():
                if name == 'Stacking':
                    continue
                if name == 'TabNet':
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                predictions.append(pred)
            
            # 메타 모델로 최종 예측
            meta_features = np.column_stack(predictions)
            final_pred = self.models[target]['Stacking'].predict(meta_features)
        else:
            # TabNet만 사용
            final_pred = self.models[target]['TabNet'].predict(X_scaled)
        
        return final_pred
    
    def save_models(self, save_dir: str):
        """모델 저장"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for target, models in self.models.items():
            target_dir = os.path.join(save_dir, target.replace('_change', ''))
            os.makedirs(target_dir, exist_ok=True)
            
            # TabNet 저장
            if 'TabNet' in models:
                models['TabNet'].save_model(os.path.join(target_dir, 'tabnet_model'))
            
            # 다른 모델들 저장
            for name, model in models.items():
                if name != 'TabNet':
                    joblib.dump(model, os.path.join(target_dir, f'{name.lower()}_model.pkl'))
            
            # Scaler 저장
            joblib.dump(self.scalers[target], os.path.join(target_dir, 'scaler.pkl'))
            
            # Feature importance 저장
            self.feature_importances[target].to_csv(
                os.path.join(target_dir, 'feature_importance.csv'),
                index=False
            )
        
        print(f"\n💾 모델 저장 완료: {save_dir}")


if __name__ == "__main__":
    # 실행 예제
    print("Loading preprocessed data...")
    df = pd.read_csv('../data/ver3_paired_data.csv')
    
    # 모델 학습
    predictor = HealthIndicatorPredictor(random_state=42)
    results = predictor.train_all_targets(df, use_ensemble=True)
    
    # 모델 저장
    predictor.save_models('../models/health_predictor')
    
    print("\n✅ 학습 완료!")
