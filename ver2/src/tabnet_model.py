"""
Ver2: TabNet Model for Change Prediction
========================================

목적: TabNet의 Sequential Attention을 활용한 변화 예측
특징: 
- 해석 가능한 특성 선택
- Attention 메커니즘으로 중요 특성 자동 식별
- Ver1의 TabNet을 Ver2 (변화 예측)에 적용
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class TabNetChangePredictor:
    """TabNet 기반 변화 예측 모델"""
    
    def __init__(self, target_variable, device='auto', random_state=42):
        """
        Args:
            target_variable: 예측할 건강지표 (예: '체중', '혈당')
            device: 'auto', 'cuda', 'cpu'
            random_state: 재현성을 위한 랜덤 시드
        """
        self.target_variable = target_variable
        self.random_state = random_state
        self.model = None
        self.scaler_X = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        
        # Device 설정
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"\n   🖥️  사용 디바이스: {self.device}")
        
    def prepare_data(self, df):
        """데이터 준비 - Data Leakage 제거"""
        print(f"\n{'='*80}")
        print(f"📊 [{self.target_variable}] 데이터 준비")
        print(f"{'='*80}")
        
        # 특성 선택: 식습관 변화만 사용 (baseline 제외하여 leakage 방지)
        # '_change'가 포함되어 있고, 건강지표가 아닌 컬럼들
        diet_change_cols = [col for col in df.columns 
                           if '_change' in col and '건강' not in col 
                           and not any(bio in col for bio in ['체중', '체질량지수', '허리둘레', 'SBP', 'DBP', 'TG'])]
        
        # 추가 특성: 시간 간격만 사용
        # ⚠️ baseline 제거: 타겟과 수학적으로 직접 연결되어 leakage 발생
        additional_features = ['time_gap_days']
        
        # 전체 특성
        feature_cols = diet_change_cols + additional_features
        self.feature_names = feature_cols
        
        # 타겟: 건강지표 변화
        target_col = f'{self.target_variable}_change'
        
        # NaN 제거
        valid_idx = df[feature_cols + [target_col]].notna().all(axis=1)
        df_clean = df[valid_idx].copy()
        
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values.reshape(-1, 1)
        
        print(f"   ✅ 유효 샘플: {len(df_clean):,}개")
        print(f"   ✅ 특성 개수: {len(feature_cols)}개 (식습관 변화만)")
        print(f"   ✅ 타겟: {target_col}")
        print(f"   ⚠️  Baseline 제거: Data leakage 방지")
        
        return X, y, df_clean
    
    def train(self, X, y, test_size=0.2, val_size=0.1, 
              max_epochs=200, patience=20, batch_size=256):
        """TabNet 모델 학습"""
        print(f"\n{'='*80}")
        print(f"🎯 [{self.target_variable}] TabNet 학습")
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
        
        # TabNet 모델 생성
        self.model = TabNetRegressor(
            n_d=16,                    # Dimension of prediction layer
            n_a=16,                    # Dimension of attention layer
            n_steps=5,                 # Number of sequential decision steps
            gamma=1.5,                 # Relaxation parameter
            n_independent=2,           # Number of independent GLU layers
            n_shared=2,                # Number of shared GLU layers
            lambda_sparse=1e-4,        # Sparsity regularization
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params=dict(mode='min', patience=5, factor=0.5),
            mask_type='entmax',        # "sparsemax" or "entmax"
            seed=self.random_state,
            device_name=self.device,
            verbose=0
        )
        
        # 학습
        print(f"\n   🔄 TabNet 학습 중 (최대 {max_epochs} epochs)...")
        
        self.model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_name=['val'],
            eval_metric=['rmse'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        # 학습 곡선 저장
        self._plot_learning_curves()
        
        # 평가
        self._evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
        
        return X_test, y_test
    
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
            
            # 방향 정확도
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
    
    def _plot_learning_curves(self):
        """학습 곡선 시각화"""
        if not hasattr(self.model, 'history'):
            return
        
        history = self.model.history
        
        plt.figure(figsize=(12, 5))
        
        # Loss curve
        plt.subplot(1, 2, 1)
        try:
            # Try accessing history as dict or object
            loss_data = None
            val_data = None
            
            if hasattr(history, 'history'):
                loss_data = history.history.get('loss', None)
                val_data = history.history.get('val_0_rmse', None)
            else:
                try:
                    loss_data = history.get('loss', None) if hasattr(history, 'get') else None
                    val_data = history.get('val_0_rmse', None) if hasattr(history, 'get') else None
                except:
                    pass
            
            if loss_data is not None:
                plt.plot(loss_data, label='Train Loss', linewidth=2)
            if val_data is not None:
                plt.plot(val_data, label='Val RMSE', linewidth=2)
                
        except Exception as e:
            print(f"   ⚠️  학습 곡선 데이터 로드 실패: {e}")
            
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{self.target_variable} TabNet 학습 곡선', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Learning rate
        plt.subplot(1, 2, 2)
        try:
            lr_data = None
            if hasattr(history, 'history'):
                lr_data = history.history.get('lr', None)
            else:
                try:
                    lr_data = history.get('lr', None) if hasattr(history, 'get') else None
                except:
                    pass
                    
            if lr_data is not None:
                plt.plot(lr_data, linewidth=2, color='orange')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Learning Rate', fontsize=12)
                plt.title('Learning Rate Schedule', fontsize=14)
                plt.grid(True, alpha=0.3)
        except Exception:
            pass
        
        plt.tight_layout()
        output_path = f'./result/tabnet_{self.target_variable}_learning_curve.png'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n   💾 학습 곡선 저장: {output_path}")
        plt.close()
    
    def plot_feature_importance(self, top_n=20):
        """특성 중요도 시각화"""
        # TabNet의 feature importance
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        plt.barh(range(top_n), importance[indices])
        plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'{self.target_variable} TabNet 특성 중요도 (Top {top_n})', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Pie chart (top 10)
        plt.subplot(1, 2, 2)
        top_10_indices = indices[:10]
        top_10_importance = importance[top_10_indices]
        top_10_names = [self.feature_names[i][:15] for i in top_10_indices]  # 이름 짧게
        
        plt.pie(top_10_importance, labels=top_10_names, autopct='%1.1f%%', startangle=90)
        plt.title('Top 10 특성 비율', fontsize=14)
        
        plt.tight_layout()
        output_path = f'./result/tabnet_{self.target_variable}_feature_importance.png'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   💾 특성 중요도 저장: {output_path}")
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
        axes[0].set_title(f'TabNet 예측 vs 실제 (Test R² = {self.metrics["Test"]["R²"]:.4f})', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = y_test.flatten() - y_pred.flatten()
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel(f'예측 {self.target_variable} 변화', fontsize=12)
        axes[1].set_ylabel('잔차', fontsize=12)
        axes[1].set_title('잔차 분포', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'./result/tabnet_{self.target_variable}_predictions.png'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   💾 예측 결과 저장: {output_path}")
        plt.close()
    
    def plot_attention_masks(self, X_sample, sample_idx=0):
        """Attention mask 시각화 (TabNet의 핵심 특징)"""
        try:
            # Explain 함수로 attention mask 추출
            explain_matrix, masks = self.model.explain(X_sample[:10])  # 샘플 10개만
            
            # masks가 딕셔너리인 경우 처리
            if isinstance(masks, dict):
                # 딕셔너리에서 masks 데이터 추출
                if 'masks' in masks:
                    masks = masks['masks']
                else:
                    # 딕셔너리의 첫 번째 값 사용
                    masks = list(masks.values())[0]
            
            # numpy array로 변환
            if not isinstance(masks, np.ndarray):
                masks = np.array(masks)
            
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            
            # masks shape: (n_samples, n_features) 또는 (n_steps, n_samples, n_features)
            # 평균 attention 사용
            if len(masks.shape) == 3:
                # (n_steps, n_samples, n_features) -> (n_samples, n_features)
                avg_masks = masks.mean(axis=0)
            else:
                avg_masks = masks
            
            for i in range(min(10, avg_masks.shape[0])):
                mask = avg_masks[i]
                
                # Mask를 특성별로 시각화
                axes[i].barh(range(len(self.feature_names)), mask, height=0.8)
                axes[i].set_yticks(range(len(self.feature_names)))
                axes[i].set_yticklabels([name[:20] for name in self.feature_names], fontsize=8)
                axes[i].set_xlabel('Attention', fontsize=10)
                axes[i].set_title(f'Sample {i+1}', fontsize=12)
                axes[i].grid(True, alpha=0.3, axis='x')
            
            plt.suptitle(f'{self.target_variable} TabNet Attention Masks', fontsize=16)
            plt.tight_layout()
            
            output_path = f'./result/tabnet_{self.target_variable}_attention_masks.png'
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"   💾 Attention masks 저장: {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"   ⚠️  Attention masks 시각화 건너뜀: {str(e)}")
            plt.close('all')
    
    def save_model(self, output_dir='./result/models'):
        """모델 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f'tabnet_{self.target_variable}.zip')
        self.model.save_model(model_path)
        
        print(f"\n   💾 모델 저장: {model_path}")
    
    def load_model(self, model_dir='./result/models'):
        """모델 로드"""
        model_path = os.path.join(model_dir, f'tabnet_{self.target_variable}.zip')
        
        self.model = TabNetRegressor()
        self.model.load_model(model_path)
        
        print(f"   ✅ 모델 로드: {model_path}")


def train_all_targets(data_path='../data/ver2_paired_visits.csv'):
    """
    모든 건강지표에 대해 TabNet 학습
    
    Args:
        data_path: Ver2 paired visits 데이터 경로 (기본: ver2/data/ver2_paired_visits.csv)
    """
    """모든 건강지표에 대해 TabNet 학습"""
    print("\n" + "="*80)
    print("🚀 Ver2 TabNet 전체 학습 시작")
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
            print(f"🎯 [{indicator}] TabNet 학습 시작")
            print(f"{'='*80}")
            
            # 모델 생성 및 학습
            model = TabNetChangePredictor(indicator)
            X, y, df_clean = model.prepare_data(df)
            X_test, y_test = model.train(X, y, max_epochs=200, patience=20)
            
            # 시각화
            model.plot_feature_importance()
            model.plot_predictions(X_test, y_test)
            
            # Attention masks (샘플)
            if len(X_test) >= 10:
                model.plot_attention_masks(X_test)
            
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
    print("📊 TabNet 전체 결과 요약")
    print("="*80)
    
    # None 값 제거 (실패한 지표 제외)
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) > 0:
        results_df = pd.DataFrame(valid_results).T
        print("\n", results_df.round(4))
        
        # 결과 저장
        output_csv = './result/tabnet_all_results.csv'
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
