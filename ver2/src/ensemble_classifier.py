"""
Ver2: Ensemble Classification Model
====================================

목적: 회귀 대신 분류 문제로 접근하여 높은 정확도 달성
방법: Random Forest + XGBoost + LightGBM 앙상블

타겟: 3-class 분류
- 0: 감소 (decrease)
- 1: 유지 (maintain)
- 2: 증가 (increase)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            confusion_matrix, classification_report, roc_auc_score)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class EnsembleClassifier:
    """앙상블 분류 모델"""
    
    def __init__(self, target_variable, random_state=42):
        """
        Args:
            target_variable: 예측할 건강지표 (예: '체중', 'BMI')
            random_state: 재현성을 위한 랜덤 시드
        """
        self.target_variable = target_variable
        self.random_state = random_state
        self.model = None
        self.scaler_X = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        self.class_names = ['감소', '유지', '증가']
        
    def prepare_data(self, df):
        """데이터 준비"""
        print(f"\n{'='*80}")
        print(f"📊 [{self.target_variable}] 분류 데이터 준비")
        print(f"{'='*80}")
        
        # 1. 식습관 변화 특성
        diet_change_cols = [col for col in df.columns 
                           if '_change' in col and '건강' not in col 
                           and not any(bio in col for bio in ['체중', '체질량지수', '허리둘레', 'SBP', 'DBP', 'TG'])]
        
        # 2. 독립적 건강지표 baseline
        obesity_indicators = ['체중', '체질량지수', '허리둘레(WAIST)']
        bp_indicators = ['SBP', 'DBP']
        metabolic_indicators = ['TG']
        
        other_health_baselines = []
        
        if self.target_variable in obesity_indicators:
            for indicator in bp_indicators + metabolic_indicators:
                baseline_col = f'{indicator}_baseline'
                if baseline_col in df.columns:
                    other_health_baselines.append(baseline_col)
        elif self.target_variable in bp_indicators:
            for indicator in obesity_indicators + metabolic_indicators:
                baseline_col = f'{indicator}_baseline'
                if baseline_col in df.columns:
                    other_health_baselines.append(baseline_col)
        elif self.target_variable in metabolic_indicators:
            for indicator in obesity_indicators + bp_indicators:
                baseline_col = f'{indicator}_baseline'
                if baseline_col in df.columns:
                    other_health_baselines.append(baseline_col)
        
        # 3. 파생 특성
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
        
        # 타겟: 분류 레이블
        target_col = f'{self.target_variable}_class'
        
        if target_col not in df_temp.columns:
            raise ValueError(f"분류 타겟 '{target_col}'이 없습니다. 데이터 전처리를 먼저 실행하세요.")
        
        # NaN 제거
        valid_idx = df_temp[feature_cols + [target_col]].notna().all(axis=1)
        df_clean = df_temp[valid_idx].copy()
        
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        
        print(f"\n   ✅ 유효 샘플: {len(df_clean):,}개")
        print(f"   ✅ 특성 개수: {len(feature_cols)}개")
        print(f"   ✅ 타겟: {target_col} (3-class 분류)")
        
        # 클래스 분포
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n   📊 클래스 분포:")
        for cls, cnt in zip(unique, counts):
            print(f"      {self.class_names[cls]}({cls}): {cnt:,}개 ({cnt/len(y)*100:.1f}%)")
        
        return X, y, df_clean
    
    def train(self, X, y, test_size=0.2, val_size=0.1):
        """앙상블 모델 학습"""
        print(f"\n{'='*80}")
        print(f"🎯 [{self.target_variable}] 앙상블 분류 모델 학습")
        print(f"{'='*80}")
        
        # Train / Val / Test 분할
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"   📊 Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        
        # 스케일링
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # 3개 모델 생성
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1
        )
        
        # Voting Ensemble
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('xgb', xgb_model),
                ('lgb', lgb_model)
            ],
            voting='soft',  # 확률 기반 투표
            n_jobs=-1
        )
        
        print(f"\n   🔄 앙상블 학습 중 (RF + XGBoost + LightGBM)...")
        
        # 학습
        self.model.fit(X_train_scaled, y_train)
        
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
            y_pred_proba = self.model.predict_proba(X)
            
            accuracy = accuracy_score(y, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
            
            self.metrics[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            }
            
            print(f"\n      [{name}]")
            print(f"         Accuracy = {accuracy:.4f}")
            print(f"         Precision = {precision:.4f}")
            print(f"         Recall = {recall:.4f}")
            print(f"         F1-Score = {f1:.4f}")
    
    def plot_confusion_matrix(self, X_test, y_test):
        """혼동 행렬 시각화"""
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.xlabel('예측', fontsize=12)
        plt.ylabel('실제', fontsize=12)
        plt.title(f'{self.target_variable} 혼동 행렬 (Accuracy={self.metrics["Test"]["Accuracy"]:.4f})', 
                 fontsize=14, fontweight='bold')
        
        output_path = f'./result/ensemble_{self.target_variable}_confusion_matrix.png'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   💾 혼동 행렬 저장: {output_path}")
        plt.close()
    
    def plot_feature_importance(self, top_n=20):
        """특성 중요도 시각화 (Random Forest 기준)"""
        rf_model = self.model.named_estimators_['rf']
        importance = rf_model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importance[indices])
        plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'{self.target_variable} 특성 중요도 (Top {top_n})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        output_path = f'./result/ensemble_{self.target_variable}_feature_importance.png'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   💾 특성 중요도 저장: {output_path}")
        plt.close()
    
    def save_model(self, output_dir='./result/models'):
        """모델 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f'ensemble_{self.target_variable}.pkl')
        scaler_path = os.path.join(output_dir, f'scaler_X_{self.target_variable}_clf.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler_X, scaler_path)
        
        print(f"\n   💾 모델 저장: {model_path}")
        print(f"   💾 스케일러 저장: {scaler_path}")


def train_all_targets(data_path='../data/ver2_paired_visits.csv'):
    """모든 건강지표에 대해 분류 모델 학습"""
    print("\n" + "="*80)
    print("🚀 Ver2 앙상블 분류 모델 전체 학습")
    print("="*80)
    
    # 데이터 로드
    df = pd.read_csv(data_path)
    print(f"\n✅ 데이터 로드 완료: {len(df):,}개 샘플")
    
    # 건강지표 목록
    health_indicators = [
        '체중', '체질량지수', '허리둘레(WAIST)', 'SBP', 'DBP', 'TG'
    ]
    
    results = {}
    
    for indicator in health_indicators:
        try:
            print(f"\n{'='*80}")
            print(f"🎯 [{indicator}] 분류 모델 학습 시작")
            print(f"{'='*80}")
            
            # 모델 생성 및 학습
            model = EnsembleClassifier(indicator)
            X, y, df_clean = model.prepare_data(df)
            X_test, y_test = model.train(X, y)
            
            # 시각화
            model.plot_confusion_matrix(X_test, y_test)
            model.plot_feature_importance()
            
            # 모델 저장
            model.save_model()
            
            # 결과 저장
            results[indicator] = model.metrics['Test']
            
            print(f"\n✅ [{indicator}] 완료!")
            
        except Exception as e:
            print(f"\n❌ [{indicator}] 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            results[indicator] = None
    
    # 전체 결과 요약
    print("\n" + "="*80)
    print("📊 앙상블 분류 모델 전체 결과 요약")
    print("="*80)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) > 0:
        results_df = pd.DataFrame(valid_results).T
        print("\n", results_df.round(4))
        
        # 결과 저장
        output_csv = './result/ensemble_all_results.csv'
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv)
        print(f"\n💾 전체 결과 저장: {output_csv}")
    else:
        print("\n⚠️ 모든 지표에서 오류가 발생했습니다.")
        results_df = pd.DataFrame()
    
    return results_df


if __name__ == '__main__':
    # 전체 학습
    results = train_all_targets()
