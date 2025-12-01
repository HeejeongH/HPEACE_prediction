"""
Ver2: LSTM Model for Change Prediction
======================================

목적: 시계열 패턴을 활용한 식습관 변화 → 건강지표 변화 예측
방법: LSTM (Long Short-Term Memory) Neural Network
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class ChangeDataset(Dataset):
    """PyTorch Dataset for change prediction"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMChangePredictor(nn.Module):
    """LSTM 기반 변화 예측 모델"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super(LSTMChangePredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len=1, features) for single time step
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last hidden state
        out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out


class LSTMModelWrapper:
    """LSTM 모델 래퍼 클래스"""
    
    def __init__(self, target_variable, device='cuda', random_state=42):
        self.target_variable = target_variable
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        print(f"\n   🖥️  사용 디바이스: {self.device}")
    
    def prepare_data(self, df):
        """데이터 준비 - 추가 특성 엔지니어링으로 성능 개선"""
        print(f"\n{'='*80}")
        print(f"📊 [{self.target_variable}] 데이터 준비 (개선 버전)")
        print(f"{'='*80}")
        
        # 1. 식습관 변화 특성
        diet_change_cols = [col for col in df.columns 
                           if '_change' in col and '건강' not in col 
                           and not any(bio in col for bio in ['체중', '체질량지수', '허리둘레', 'SBP', 'DBP', 'TG'])]
        
        # 2. ✅ 다른 건강지표 baseline 추가 (독립적 지표만)
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
            other_bp = [bp for bp in bp_indicators if bp != self.target_variable]
            for indicator in other_bp:
                baseline_col = f'{indicator}_baseline'
                if baseline_col in other_health_baselines:
                    other_health_baselines.remove(baseline_col)
        
        elif self.target_variable in metabolic_indicators:
            for indicator in obesity_indicators + bp_indicators:
                baseline_col = f'{indicator}_baseline'
                if baseline_col in df.columns:
                    other_health_baselines.append(baseline_col)
        
        print(f"\n   📈 추가된 다른 건강지표 baseline: {len(other_health_baselines)}개")
        
        # 3. ✅ 파생 특성 생성
        from pathlib import Path
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
        y = df_clean[target_col].values.reshape(-1, 1)
        
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
        features_csv = f'./result/features_used_LSTM_{self.target_variable}.csv'
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
    
    def train(self, X, y, test_size=0.2, val_size=0.1, 
              epochs=100, batch_size=64, learning_rate=0.001):
        """모델 학습"""
        print(f"\n{'='*80}")
        print(f"🎯 [{self.target_variable}] LSTM 학습")
        print(f"{'='*80}")
        
        # Train / Val / Test 분할
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
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # Reshape for LSTM (batch, seq_len=1, features)
        X_train_scaled = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
        X_val_scaled = X_val_scaled.reshape(-1, 1, X_val_scaled.shape[1])
        X_test_scaled = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])
        
        # DataLoaders
        train_dataset = ChangeDataset(X_train_scaled, y_train_scaled)
        val_dataset = ChangeDataset(X_val_scaled, y_val_scaled)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 모델 생성
        input_dim = X_train_scaled.shape[2]
        self.model = LSTMChangePredictor(input_dim).to(self.device)
        
        # Loss & Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        
        # 학습
        print(f"\n   🔄 학습 중 ({epochs} epochs)...")
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"      ⚠️  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        # 평가
        self._evaluate(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
        
        # 학습 곡선 그리기
        self._plot_learning_curves(train_losses, val_losses)
        
        return X_test_scaled, y_test
    
    def _evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """모델 평가"""
        print(f"\n   📈 성능 평가:")
        
        self.model.eval()
        datasets = {
            'Train': (X_train, y_train),
            'Val': (X_val, y_val),
            'Test': (X_test, y_test)
        }
        
        with torch.no_grad():
            for name, (X, y) in datasets.items():
                X_tensor = torch.FloatTensor(X).to(self.device)
                y_pred_scaled = self.model(X_tensor).cpu().numpy()
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
                
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                mae = mean_absolute_error(y, y_pred)
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
    
    def _plot_learning_curves(self, train_losses, val_losses):
        """학습 곡선 시각화"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title(f'{self.target_variable} LSTM 학습 곡선', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = f'./result/lstm_{self.target_variable}_learning_curve.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n   💾 학습 곡선 저장: {output_path}")
        plt.close()
    
    def plot_predictions(self, X_test, y_test):
        """예측 결과 시각화"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Scatter plot
        axes[0].scatter(y_test, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel(f'실제 {self.target_variable} 변화', fontsize=12)
        axes[0].set_ylabel(f'예측 {self.target_variable} 변화', fontsize=12)
        axes[0].set_title(f'LSTM 예측 vs 실제 (Test R² = {self.metrics["Test"]["R²"]:.4f})', fontsize=14)
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
        output_path = f'./result/lstm_{self.target_variable}_predictions.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   💾 예측 결과 저장: {output_path}")
        plt.close()
    
    def save_model(self, output_dir='./result/models'):
        """모델 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f'lstm_{self.target_variable}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names
        }, model_path)
        
        print(f"\n   💾 모델 저장: {model_path}")


def train_all_targets(data_path='../data/ver2_paired_visits.csv'):
    """모든 건강지표에 대해 LSTM 학습"""
    print("\n" + "="*80)
    print("🚀 Ver2 LSTM 전체 학습 시작")
    print("="*80)
    
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
            print(f"🎯 [{indicator}] LSTM 학습 시작")
            print(f"{'='*80}")
            
            model = LSTMModelWrapper(indicator)
            X, y, df_clean = model.prepare_data(df)
            X_test, y_test = model.train(X, y, epochs=100)
            
            model.plot_predictions(X_test, y_test)
            model.save_model()
            
            results[indicator] = model.metrics['Test']
            
            print(f"\n✅ [{indicator}] 완료!")
            
        except Exception as e:
            print(f"\n❌ [{indicator}] 오류: {str(e)}")
            results[indicator] = None
    
    # 결과 요약
    print("\n" + "="*80)
    print("📊 LSTM 전체 결과 요약")
    print("="*80)
    
    results_df = pd.DataFrame(results).T
    print("\n", results_df.round(4))
    
    results_df.to_csv('./result/lstm_all_results.csv')
    print(f"\n💾 전체 결과 저장: ./result/lstm_all_results.csv")
    
    return results_df


if __name__ == '__main__':
    results = train_all_targets()
