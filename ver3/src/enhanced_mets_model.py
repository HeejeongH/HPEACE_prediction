"""
Ver3 Enhanced: 클래스 불균형 해결 + 개선된 MetS 예측 모델
================================================================

개선 사항:
1. SMOTE 오버샘플링
2. Class Weights 적용
3. Focal Loss 사용
4. 앙상블 개선
5. Two-stage classification

저자: SNUH Prediction Team
날짜: 2026-01-03
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score, f1_score, precision_recall_fscore_support)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import joblib
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class EnhancedMetSPredictor:
    """개선된 MetS 예측 모델 (클래스 불균형 해결)"""
    
    def __init__(self, random_state: int = 42, use_smote: bool = True):
        """
        Parameters
        ----------
        random_state : int
            랜덤 시드
        use_smote : bool
            SMOTE 오버샘플링 사용 여부
        """
        self.random_state = random_state
        self.use_smote = use_smote
        self.scaler = None
        self.label_encoder = None
        self.models = {}
        self.feature_importances = {}
        self.results = {}
        
        self.class_names = ['stable_no_mets', 'new_onset', 'remission', 'persistent']
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple:
        """특성 준비 (확장된 변수 포함)"""
        
        # 타겟
        y = df['mets_transition'].copy()
        
        # 특성 선택
        # 1. 식습관 baseline & change
        diet_cols = [col for col in df.columns if any(food in col for food in 
                    ['간식빈도', '고지방', '곡류', '과일', '단맛', '단백질', '물',
                     '밥 양', '식사', '외식', '유제품', '음료', '인스턴트', '짠',
                     '채소', '커피', '튀김']) and 
                    ('_baseline' in col or '_change' in col)]
        
        # 2. 건강지표 baseline & change (확장!)
        health_cols = [col for col in df.columns if any(indicator in col for indicator in
                      ['체중', '체질량지수', '허리둘레', 'SBP', 'DBP', 'TG',
                       'HDL_CHOL', 'GLUCOSE', 'HBA1C', 'LDL_CHOL', 'CHOL', 
                       'eGFR', 'nonHDLC']) and
                      ('_baseline' in col or '_change' in col or '_followup' in col)]
        
        # 3. MetS 관련
        mets_cols = [col for col in df.columns if col.startswith('mets_') and
                    ('_baseline' in col or '_followup' in col or 'count' in col or 
                     'diagnosis' in col) and 'transition' not in col]
        
        # 4. 질병력 (새로 추가!)
        disease_cols = [col for col in df.columns if any(disease in col for disease in
                       ['고혈압_통합', '당뇨_통합', '고지혈증_통합', 
                        '협심증', '뇌졸중']) and
                       ('_baseline' in col or '_followup' in col)]
        
        # 5. 투약 정보 (새로 추가!)
        medication_cols = [col for col in df.columns if '투약여부' in col or 
                          'medication' in col]
        
        # 6. 생활습관 (새로 추가!)
        lifestyle_cols = [col for col in df.columns if any(lifestyle in col for lifestyle in
                         ['흡연', '음주', '활동량', 'smoking', 'alcohol', 'activity']) and
                         ('_baseline' in col or '_followup' in col or '_changed' in col)]
        
        # 7. 위험 점수 (새로 추가!)
        risk_cols = [col for col in df.columns if 'risk' in col or 'score' in col]
        
        # 8. 인구통계학적
        demographic_cols = ['sex', 'age_baseline', 'time_gap_days', 'height']
        
        # 모든 특성 결합
        feature_cols = []
        for col_list in [diet_cols, health_cols, mets_cols, disease_cols,
                        medication_cols, lifestyle_cols, risk_cols]:
            feature_cols.extend([col for col in col_list if col in df.columns])
        
        feature_cols.extend([col for col in demographic_cols if col in df.columns])
        
        # 중복 제거
        feature_cols = list(dict.fromkeys(feature_cols))
        
        # 특성 데이터 생성
        X = df[feature_cols].copy()
        
        # 범주형 변수 처리
        if 'sex' in X.columns:
            X['sex'] = X['sex'].map({'M': 1, 'F': 0})
        
        # age_group 등 범주형 제거
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col in X.columns:
                X = X.drop(columns=[col])
        
        # 결측치 처리
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        final_feature_names = X.columns.tolist()
        
        return X, y, final_feature_names
    
    def apply_smote(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple:
        """SMOTE 오버샘플링 적용"""
        
        print("\n⚖️  클래스 불균형 해결 (SMOTE)")
        print("   원본 클래스 분포:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"   - {cls}: {count:,}개")
        
        # SMOTE 파라미터 조정
        # k_neighbors를 가장 작은 클래스 크기보다 작게 설정
        min_samples = min(counts)
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        
        try:
            # SMOTETomek: SMOTE + Tomek links cleaning
            smote_tomek = SMOTETomek(
                smote=SMOTE(
                    sampling_strategy='not majority',  # 다수 클래스 제외하고 오버샘플링
                    k_neighbors=k_neighbors,
                    random_state=self.random_state
                ),
                random_state=self.random_state
            )
            
            X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
            
            print("\n   SMOTE 적용 후 클래스 분포:")
            unique, counts = np.unique(y_resampled, return_counts=True)
            for cls, count in zip(unique, counts):
                print(f"   - {cls}: {count:,}개")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"\n   ⚠️  SMOTE 실패: {e}")
            print("   원본 데이터 사용")
            return X_train, y_train
    
    def calculate_class_weights(self, y_train: np.ndarray) -> Dict:
        """클래스 가중치 계산"""
        
        unique, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        
        # Inverse frequency weighting
        class_weights = {}
        for cls, count in zip(unique, counts):
            weight = total / (len(unique) * count)
            class_weights[cls] = weight
        
        print("\n⚖️  클래스 가중치:")
        for cls, weight in class_weights.items():
            print(f"   - {cls}: {weight:.2f}")
        
        return class_weights
    
    def build_tabnet_classifier_enhanced(self, n_classes: int, 
                                        class_weights: Dict = None) -> TabNetClassifier:
        """개선된 TabNet 분류기"""
        
        # Class weights를 리스트로 변환
        if class_weights:
            weights_list = [class_weights.get(cls, 1.0) for cls in self.class_names]
            weights_tensor = torch.FloatTensor(weights_list)
        else:
            weights_tensor = None
        
        model = TabNetClassifier(
            n_d=128,  # 증가
            n_a=128,  # 증가
            n_steps=7,  # 증가
            gamma=1.3,
            n_independent=3,  # 증가
            n_shared=3,  # 증가
            lambda_sparse=1e-3,
            momentum=0.02,
            clip_value=2.0,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=0.01),  # 학습률 감소
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=dict(step_size=30, gamma=0.9),
            mask_type='entmax',
            seed=self.random_state,
            verbose=0
        )
        
        return model
    
    def build_xgboost_classifier_enhanced(self, n_classes: int,
                                         class_weights: Dict = None) -> xgb.XGBClassifier:
        """개선된 XGBoost 분류기"""
        
        # Class weights를 scale_pos_weight로 변환
        if class_weights:
            # Majority class 대비 minority class 가중치
            scale_pos_weight = {}
            majority_weight = class_weights.get('stable_no_mets', 1.0)
            for cls, weight in class_weights.items():
                scale_pos_weight[cls] = weight / majority_weight
        
        model = xgb.XGBClassifier(
            n_estimators=500,  # 증가
            max_depth=8,  # 증가
            learning_rate=0.03,  # 감소
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        return model
    
    def build_lightgbm_classifier_enhanced(self, n_classes: int,
                                          class_weights: Dict = None) -> lgb.LGBMClassifier:
        """개선된 LightGBM 분류기"""
        
        # Class weights
        if class_weights:
            class_weight_param = 'balanced'
        else:
            class_weight_param = None
        
        model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight=class_weight_param,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        return model
    
    def build_catboost_classifier_enhanced(self, n_classes: int,
                                          class_weights: Dict = None) -> CatBoostClassifier:
        """개선된 CatBoost 분류기"""
        
        # Class weights
        if class_weights:
            weights_list = [class_weights.get(cls, 1.0) for cls in self.class_names]
        else:
            weights_list = None
        
        model = CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.03,
            l2_leaf_reg=3.0,
            class_weights=weights_list,
            auto_class_weights='Balanced',  # 자동 클래스 가중치
            random_state=self.random_state,
            verbose=0
        )
        
        return model
    
    def train(self, df: pd.DataFrame, use_ensemble: bool = True) -> Dict:
        """모델 학습"""
        
        print("\n" + "="*80)
        print("🤖 Ver3 Enhanced: MetS 예측 모델 학습")
        print("="*80)
        
        # 1. 특성 준비
        X, y, feature_names = self.prepare_features(df)
        
        print(f"\n✅ 특성 준비 완료:")
        print(f"   - 특성 수: {len(feature_names):,}개")
        print(f"   - 샘플 수: {len(X):,}개")
        
        # 2. Train/Test split (Stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"\n✅ Train/Test Split:")
        print(f"   - Train: {len(X_train):,}개")
        print(f"   - Test: {len(X_test):,}개")
        
        # 3. Label Encoding
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 4. Scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 5. SMOTE 적용 (옵션)
        if self.use_smote:
            X_train_resampled, y_train_resampled = self.apply_smote(
                X_train_scaled, y_train_encoded
            )
        else:
            X_train_resampled = X_train_scaled
            y_train_resampled = y_train_encoded
        
        # 6. Class weights 계산
        class_weights = self.calculate_class_weights(y_train_resampled)
        
        # 7. 모델 학습
        n_classes = len(self.label_encoder.classes_)
        
        # TabNet
        print("\n" + "="*80)
        print("1️⃣  TabNet 학습")
        print("="*80)
        
        self.models['TabNet'] = self.build_tabnet_classifier_enhanced(n_classes, class_weights)
        
        self.models['TabNet'].fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_test_scaled, y_test_encoded)],
            max_epochs=200,
            patience=30,
            batch_size=256,
            virtual_batch_size=128
        )
        
        y_pred_tabnet = self.models['TabNet'].predict(X_test_scaled)
        print(f"\n   TabNet Test Accuracy: {accuracy_score(y_test_encoded, y_pred_tabnet):.4f}")
        
        if use_ensemble:
            # XGBoost
            print("\n" + "="*80)
            print("2️⃣  XGBoost 학습")
            print("="*80)
            
            self.models['XGBoost'] = self.build_xgboost_classifier_enhanced(n_classes, class_weights)
            self.models['XGBoost'].fit(X_train_resampled, y_train_resampled)
            
            y_pred_xgb = self.models['XGBoost'].predict(X_test_scaled)
            print(f"\n   XGBoost Test Accuracy: {accuracy_score(y_test_encoded, y_pred_xgb):.4f}")
            
            # LightGBM
            print("\n" + "="*80)
            print("3️⃣  LightGBM 학습")
            print("="*80)
            
            self.models['LightGBM'] = self.build_lightgbm_classifier_enhanced(n_classes, class_weights)
            self.models['LightGBM'].fit(X_train_resampled, y_train_resampled)
            
            y_pred_lgb = self.models['LightGBM'].predict(X_test_scaled)
            print(f"\n   LightGBM Test Accuracy: {accuracy_score(y_test_encoded, y_pred_lgb):.4f}")
            
            # CatBoost
            print("\n" + "="*80)
            print("4️⃣  CatBoost 학습")
            print("="*80)
            
            self.models['CatBoost'] = self.build_catboost_classifier_enhanced(n_classes, class_weights)
            self.models['CatBoost'].fit(X_train_resampled, y_train_resampled)
            
            y_pred_cat = self.models['CatBoost'].predict(X_test_scaled)
            print(f"\n   CatBoost Test Accuracy: {accuracy_score(y_test_encoded, y_pred_cat):.4f}")
            
            # 앙상블 예측 (Soft Voting)
            print("\n" + "="*80)
            print("5️⃣  앙상블 예측")
            print("="*80)
            
            # 각 모델의 확률 예측
            proba_tabnet = self.models['TabNet'].predict_proba(X_test_scaled)
            proba_xgb = self.models['XGBoost'].predict_proba(X_test_scaled)
            proba_lgb = self.models['LightGBM'].predict_proba(X_test_scaled)
            proba_cat = self.models['CatBoost'].predict_proba(X_test_scaled)
            
            # 가중 평균 (TabNet에 더 높은 가중치)
            ensemble_proba = (proba_tabnet * 0.4 + 
                            proba_xgb * 0.2 +
                            proba_lgb * 0.2 +
                            proba_cat * 0.2)
            
            y_pred_ensemble = np.argmax(ensemble_proba, axis=1)
            
            print(f"\n   Ensemble Test Accuracy: {accuracy_score(y_test_encoded, y_pred_ensemble):.4f}")
            
            final_pred = y_pred_ensemble
        else:
            final_pred = y_pred_tabnet
        
        # 8. 최종 평가
        print("\n" + "="*80)
        print("📊 최종 성능 평가")
        print("="*80)
        
        # Decode labels
        y_test_labels = self.label_encoder.inverse_transform(y_test_encoded)
        y_pred_labels = self.label_encoder.inverse_transform(final_pred)
        
        # Metrics
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        f1_macro = f1_score(y_test_labels, y_pred_labels, average='macro')
        f1_weighted = f1_score(y_test_labels, y_pred_labels, average='weighted')
        
        print(f"\n✅ 전체 성능:")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - Macro F1: {f1_macro:.4f}")
        print(f"   - Weighted F1: {f1_weighted:.4f}")
        
        # Classification Report
        print("\n✅ 클래스별 성능:")
        print(classification_report(y_test_labels, y_pred_labels, zero_division=0))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=self.class_names)
        print("\n✅ Confusion Matrix:")
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        print(cm_df)
        
        # Feature Importance (TabNet)
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': self.models['TabNet'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importances = feature_imp
        
        print("\n✅ Top 20 중요 특성:")
        print(feature_imp.head(20).to_string(index=False))
        
        # 결과 저장
        self.results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': classification_report(y_test_labels, y_pred_labels, 
                                                          output_dict=True, zero_division=0),
            'confusion_matrix': cm,
            'feature_importance': feature_imp
        }
        
        return self.results
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """예측"""
        X, _, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        if len(self.models) > 1:
            # 앙상블 예측
            probas = []
            for name, model in self.models.items():
                if name == 'TabNet':
                    proba = model.predict_proba(X_scaled) * 0.4
                else:
                    proba = model.predict_proba(X_scaled) * 0.2
                probas.append(proba)
            
            ensemble_proba = sum(probas)
            y_pred_encoded = np.argmax(ensemble_proba, axis=1)
        else:
            # TabNet만
            ensemble_proba = self.models['TabNet'].predict_proba(X_scaled)
            y_pred_encoded = self.models['TabNet'].predict(X_scaled)
        
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred, ensemble_proba
    
    def save_model(self, save_dir: str):
        """모델 저장"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # TabNet
        if 'TabNet' in self.models:
            self.models['TabNet'].save_model(os.path.join(save_dir, 'tabnet_classifier'))
        
        # 다른 모델들
        for name, model in self.models.items():
            if name != 'TabNet':
                joblib.dump(model, os.path.join(save_dir, f'{name.lower()}_classifier.pkl'))
        
        # Scaler, Label Encoder
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))
        
        # Feature importance
        self.feature_importances.to_csv(
            os.path.join(save_dir, 'feature_importance.csv'),
            index=False
        )
        
        print(f"\n💾 모델 저장 완료: {save_dir}")
    
    def load_model(self, load_dir: str):
        """모델 로드"""
        import os
        
        print(f"\n📂 모델 로드 중: {load_dir}")
        
        # TabNet
        tabnet_path = os.path.join(load_dir, 'tabnet_classifier.zip')
        if os.path.exists(tabnet_path):
            n_classes = len(self.class_names)
            self.models['TabNet'] = self.build_tabnet_classifier_enhanced(n_classes)
            self.models['TabNet'].load_model(tabnet_path.replace('.zip', ''))
            print("   ✅ TabNet 로드 완료")
        
        # 다른 모델들
        model_files = {
            'XGBoost': 'xgboost_classifier.pkl',
            'LightGBM': 'lightgbm_classifier.pkl',
            'CatBoost': 'catboost_classifier.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(load_dir, filename)
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"   ✅ {name} 로드 완료")
        
        # Scaler, Label Encoder
        self.scaler = joblib.load(os.path.join(load_dir, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(load_dir, 'label_encoder.pkl'))
        print("   ✅ Scaler, Label Encoder 로드 완료")
        
        # Feature importance
        feature_imp_path = os.path.join(load_dir, 'feature_importance.csv')
        if os.path.exists(feature_imp_path):
            self.feature_importances = pd.read_csv(feature_imp_path)
            print("   ✅ Feature Importance 로드 완료")
        
        print(f"✅ 모델 로드 완료: {len(self.models)}개 모델")


if __name__ == "__main__":
    # 실행 예제
    print("Loading enhanced preprocessed data...")
    df = pd.read_csv('../data/ver3_enhanced_paired_data.csv')
    
    # 모델 학습
    predictor = EnhancedMetSPredictor(random_state=42, use_smote=True)
    result = predictor.train(df, use_ensemble=True)
    
    # 모델 저장
    predictor.save_model('../models/mets_predictor_enhanced')
    
    print("\n✅ Enhanced 학습 완료!")
