"""
Ver3: MetS 발생/개선 예측 분류 모델
====================================

목표: 식습관 변화를 통해 MetS 발생 또는 개선 예측

분류 타겟:
1. new_onset: MetS 발생 (baseline 없음 → follow-up 있음)
2. remission: MetS 개선 (baseline 있음 → follow-up 없음)
3. persistent: MetS 지속 (baseline 있음 → follow-up 있음)
4. stable_no_mets: MetS 없음 유지 (baseline 없음 → follow-up 없음)

저자: SNUH Prediction Team
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, accuracy_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import warnings
import joblib
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class MetSPredictor:
    """MetS 발생/개선 예측 모델 클래스"""
    
    def __init__(self, random_state: int = 42):
        """
        Parameters
        ----------
        random_state : int
            랜덤 시드
        """
        self.random_state = random_state
        self.scaler = None
        self.label_encoder = None
        self.models = {}
        self.feature_importances = {}
        self.results = {}
        
        # 분류 클래스
        self.class_names = ['stable_no_mets', 'new_onset', 'remission', 'persistent']
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple:
        """
        MetS 예측을 위한 특성 준비
        
        Parameters
        ----------
        df : DataFrame
            전처리된 paired visits 데이터
            
        Returns
        -------
        X : DataFrame
            특성 데이터
        y : Series
            타겟 데이터 (MetS 변화 패턴)
        feature_names : List[str]
            특성 이름 리스트
        """
        # 타겟 추출 및 인코딩
        y = df['mets_transition'].copy()
        
        # 특성 선택
        # 1. 식습관 baseline
        diet_baseline = [col for col in df.columns if col.endswith('_baseline') 
                        and not col.startswith(('체중', '체질량지수', '허리둘레', 'SBP', 'DBP', 'TG', 'HDL', 'glucose', 'HbA1c', 'mets'))]
        
        # 2. 식습관 change (핵심!)
        diet_change = [col for col in df.columns if '_change' in col 
                      and not col.startswith(('체중', '체질량지수', '허리둘레', 'SBP', 'DBP', 'TG', 'HDL', 'glucose', 'HbA1c', 'mets', 'monthly'))]
        
        # 3. 건강지표 baseline (MetS 관련만)
        health_baseline = ['체중_baseline', '체질량지수_baseline', '허리둘레(WAIST)_baseline',
                          'SBP_baseline', 'DBP_baseline', 'TG_baseline', 
                          'HDL_baseline', 'glucose_baseline']
        
        # 4. Baseline MetS 정보 (매우 중요!)
        mets_baseline = ['mets_diagnosis_baseline', 'mets_count_baseline',
                        'mets_waist_baseline', 'mets_tg_baseline', 
                        'mets_hdl_baseline', 'mets_bp_baseline', 
                        'mets_glucose_baseline']
        
        # 5. 인구통계학적 변수
        demographic = ['sex', 'age_baseline', 'time_gap_days']
        
        # 6. 고급 특성
        advanced = ['healthy_score_baseline', 'healthy_score_change',
                   'unhealthy_score_baseline', 'unhealthy_score_change',
                   'diet_improvement_score', 'diet_diversity_baseline',
                   'baseline_risk']
        
        # 전체 특성 결합
        feature_cols = []
        for col_list in [diet_baseline, diet_change, health_baseline, 
                        mets_baseline, demographic, advanced]:
            feature_cols.extend([col for col in col_list if col in df.columns])
        
        # 중복 제거
        feature_cols = list(dict.fromkeys(feature_cols))
        
        # 특성 데이터 생성
        X = df[feature_cols].copy()
        
        # 범주형 변수 처리
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # sex 인코딩
        if 'sex' in X.columns:
            X['sex'] = X['sex'].map({'M': 1, 'F': 0})
        
        # 나머지 범주형 변수 제거
        for col in categorical_cols:
            if col in X.columns and col != 'sex':
                X = X.drop(columns=[col])
        
        # 결측치 처리 (수치형만)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # 최종 feature 이름 업데이트
        final_feature_names = X.columns.tolist()
        
        return X, y, final_feature_names
    
    def build_tabnet_classifier(self, n_classes: int) -> TabNetClassifier:
        """TabNet 분류 모델 생성"""
        model = TabNetClassifier(
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
    
    def build_xgboost_classifier(self, n_classes: int) -> xgb.XGBClassifier:
        """XGBoost 분류 모델 생성"""
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
            num_class=n_classes if n_classes > 2 else None
        )
        return model
    
    def build_lightgbm_classifier(self, n_classes: int) -> lgb.LGBMClassifier:
        """LightGBM 분류 모델 생성"""
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
            objective='multiclass' if n_classes > 2 else 'binary',
            num_class=n_classes if n_classes > 2 else None
        )
        return model
    
    def build_catboost_classifier(self, n_classes: int) -> CatBoostClassifier:
        """CatBoost 분류 모델 생성"""
        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_state=self.random_state,
            verbose=0,
            loss_function='MultiClass' if n_classes > 2 else 'Logloss'
        )
        return model
    
    def train(self, df: pd.DataFrame, use_ensemble: bool = True) -> Dict:
        """
        MetS 예측 모델 학습
        
        Parameters
        ----------
        df : DataFrame
            전처리된 데이터
        use_ensemble : bool
            앙상블 사용 여부
            
        Returns
        -------
        result : Dict
            학습 결과 및 모델 정보
        """
        print(f"\n{'='*80}")
        print(f"🎯 MetS 발생/개선 예측 모델 학습")
        print(f"{'='*80}")
        
        # 데이터 준비
        X, y, feature_names = self.prepare_features(df)
        
        # Label 인코딩
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        
        print(f"\n   클래스 수: {n_classes}개")
        print(f"   특성 수: {X.shape[1]}개")
        print(f"   샘플 수: {len(X):,}개")
        
        # 클래스 분포
        print(f"\n   📊 클래스 분포:")
        for class_name in self.label_encoder.classes_:
            count = (y == class_name).sum()
            pct = count / len(y) * 100
            print(f"      {class_name}: {count:,}개 ({pct:.1f}%)")
        
        # Train/Test 분할 (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=self.random_state,
            stratify=y_encoded
        )
        
        # 스케일링
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 모델 학습
        models = {}
        predictions_train = {}
        predictions_test = {}
        predictions_proba_test = {}
        
        # 1. TabNet
        print(f"\n   📊 Training TabNet Classifier...")
        tabnet = self.build_tabnet_classifier(n_classes)
        tabnet.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            max_epochs=100,
            patience=20,
            batch_size=256,
            virtual_batch_size=128,
            eval_metric=['accuracy']
        )
        
        models['TabNet'] = tabnet
        predictions_train['TabNet'] = tabnet.predict(X_train_scaled)
        predictions_test['TabNet'] = tabnet.predict(X_test_scaled)
        predictions_proba_test['TabNet'] = tabnet.predict_proba(X_test_scaled)
        
        tabnet_acc = accuracy_score(y_test, predictions_test['TabNet'])
        tabnet_f1 = f1_score(y_test, predictions_test['TabNet'], average='weighted')
        print(f"      Accuracy = {tabnet_acc:.4f}, F1 = {tabnet_f1:.4f}")
        
        if use_ensemble:
            # 2. XGBoost
            print(f"   📊 Training XGBoost Classifier...")
            xgb_model = self.build_xgboost_classifier(n_classes)
            xgb_model.fit(X_train, y_train)
            models['XGBoost'] = xgb_model
            predictions_train['XGBoost'] = xgb_model.predict(X_train)
            predictions_test['XGBoost'] = xgb_model.predict(X_test)
            predictions_proba_test['XGBoost'] = xgb_model.predict_proba(X_test)
            
            xgb_acc = accuracy_score(y_test, predictions_test['XGBoost'])
            print(f"      Accuracy = {xgb_acc:.4f}")
            
            # 3. LightGBM
            print(f"   📊 Training LightGBM Classifier...")
            lgb_model = self.build_lightgbm_classifier(n_classes)
            lgb_model.fit(X_train, y_train)
            models['LightGBM'] = lgb_model
            predictions_train['LightGBM'] = lgb_model.predict(X_train)
            predictions_test['LightGBM'] = lgb_model.predict(X_test)
            predictions_proba_test['LightGBM'] = lgb_model.predict_proba(X_test)
            
            lgb_acc = accuracy_score(y_test, predictions_test['LightGBM'])
            print(f"      Accuracy = {lgb_acc:.4f}")
            
            # 4. CatBoost
            print(f"   📊 Training CatBoost Classifier...")
            cat_model = self.build_catboost_classifier(n_classes)
            cat_model.fit(X_train, y_train)
            models['CatBoost'] = cat_model
            predictions_train['CatBoost'] = cat_model.predict(X_train)
            predictions_test['CatBoost'] = cat_model.predict(X_test)
            predictions_proba_test['CatBoost'] = cat_model.predict_proba(X_test)
            
            cat_acc = accuracy_score(y_test, predictions_test['CatBoost'])
            print(f"      Accuracy = {cat_acc:.4f}")
            
            # 5. Voting (평균 확률)
            print(f"\n   🔗 Ensemble Voting...")
            avg_proba = np.mean([predictions_proba_test[name] for name in models.keys()], axis=0)
            final_pred = np.argmax(avg_proba, axis=1)
            
            ensemble_acc = accuracy_score(y_test, final_pred)
            ensemble_f1 = f1_score(y_test, final_pred, average='weighted')
            print(f"      Accuracy = {ensemble_acc:.4f}, F1 = {ensemble_f1:.4f}")
            
        else:
            final_pred = predictions_test['TabNet']
            avg_proba = predictions_proba_test['TabNet']
        
        # 성능 평가
        accuracy = accuracy_score(y_test, final_pred)
        f1 = f1_score(y_test, final_pred, average='weighted')
        
        print(f"\n   ✅ 최종 성능:")
        print(f"      Accuracy = {accuracy:.4f}")
        print(f"      F1 Score = {f1:.4f}")
        
        # 클래스별 성능
        print(f"\n   📊 Classification Report:")
        report = classification_report(
            y_test, final_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        )
        print(report)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, final_pred)
        print(f"\n   📊 Confusion Matrix:")
        cm_df = pd.DataFrame(
            cm,
            index=self.label_encoder.classes_,
            columns=self.label_encoder.classes_
        )
        print(cm_df)
        
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
            'models': models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': feature_names,
            'feature_importance': feature_importance,
            'performance': {
                'accuracy': accuracy,
                'f1_score': f1,
                'classification_report': report,
                'confusion_matrix': cm
            },
            'predictions': {
                'y_test': y_test,
                'y_test_original': self.label_encoder.inverse_transform(y_test),
                'pred_test': final_pred,
                'pred_test_original': self.label_encoder.inverse_transform(final_pred),
                'pred_proba': avg_proba
            }
        }
        
        self.models = models
        self.feature_importances = feature_importance
        self.results = result
        
        return result
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        새로운 데이터에 대한 MetS 예측
        
        Parameters
        ----------
        df : DataFrame
            예측할 데이터
            
        Returns
        -------
        predictions : np.ndarray
            예측된 클래스 (인코딩된 값)
        probabilities : np.ndarray
            클래스별 확률
        """
        X, _, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # 모든 모델의 확률 평균
        probas = []
        for name, model in self.models.items():
            if name == 'TabNet':
                proba = model.predict_proba(X_scaled)
            else:
                proba = model.predict_proba(X)
            probas.append(proba)
        
        avg_proba = np.mean(probas, axis=0)
        predictions = np.argmax(avg_proba, axis=1)
        
        return predictions, avg_proba
    
    def predict_with_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """레이블과 함께 예측 반환"""
        predictions, probabilities = self.predict(df)
        
        result_df = pd.DataFrame({
            'predicted_class': self.label_encoder.inverse_transform(predictions)
        })
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            result_df[f'prob_{class_name}'] = probabilities[:, i]
        
        return result_df
    
    def save_model(self, save_dir: str):
        """모델 저장"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # TabNet 저장
        if 'TabNet' in self.models:
            self.models['TabNet'].save_model(os.path.join(save_dir, 'tabnet_classifier'))
        
        # 다른 모델들 저장
        for name, model in self.models.items():
            if name != 'TabNet':
                joblib.dump(model, os.path.join(save_dir, f'{name.lower()}_classifier.pkl'))
        
        # Scaler 및 Label Encoder 저장
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))
        
        # Feature importance 저장
        self.feature_importances.to_csv(
            os.path.join(save_dir, 'feature_importance.csv'),
            index=False
        )
        
        print(f"\n💾 모델 저장 완료: {save_dir}")


if __name__ == "__main__":
    # 실행 예제
    print("Loading preprocessed data...")
    df = pd.read_csv('../data/ver3_paired_data.csv')
    
    # 모델 학습
    predictor = MetSPredictor(random_state=42)
    result = predictor.train(df, use_ensemble=True)
    
    # 모델 저장
    predictor.save_model('../models/mets_predictor')
    
    print("\n✅ 학습 완료!")
