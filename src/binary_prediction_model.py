"""
2-Class 예측 모델: 변화 감지 (유지 vs 변화)

목적: 대사증후군 질병의 변화를 예측
- Class 0: 유지 (질병 상태 변화 없음)
- Class 1: 변화 (개선 또는 악화)

장점:
- 클래스 불균형 완화 (3:1 ~ 5:1)
- 학습 용이성 향상
- 임상적 의미 (변화 감지 자체가 중요)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from data_import import SingleDietImpactDataset, DataLoaderManager
from resampling import create_resampled_dataset
from MetS_prediction_model import EarlyStopping
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


class BinaryDiseasePredictor(nn.Module):
    """
    2-Class 질병 변화 예측 모델
    
    Architecture:
    - 입력 차원: 75 (diet:19, demo:4, life:3, bio:11, delta:22, inter:6, pca:10)
    - Hidden layers: 4층 (256 → 128 → 64 → 32)
    - 출력: 2-class (유지 vs 변화)
    """
    
    def __init__(self, input_dims, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        
        total_dim = sum(input_dims.values())
        
        self.encoder = nn.Sequential(
            # Layer 1: 75 → 256
            nn.Linear(total_dim, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2: 256 → 128
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 3: 128 → 64
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 4: 64 → 32
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output: 32 → 2
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, diet, demo, life, bio, delta, inter, pca):
        """
        Forward pass
        
        Args:
            diet: (batch, 19)
            demo: (batch, 4)
            life: (batch, 3)
            bio: (batch, 11)
            delta: (batch, 22)
            inter: (batch, 6)
            pca: (batch, 10)
        
        Returns:
            logits: (batch, 2) - 2-class logits
        """
        x = torch.cat([diet, demo, life, bio, delta, inter, pca], dim=1)
        return self.encoder(x)


def convert_to_binary_targets(df, disease_names):
    """
    3-class 타겟을 2-class로 변환
    
    Args:
        df: DataFrame with {disease}_delta columns
        disease_names: List of disease names
    
    Returns:
        df_binary: DataFrame with binary targets
            - 0: 유지 (원래 class 1)
            - 1: 변화 (원래 class 0 + class 2)
    """
    df_binary = df.copy()
    
    for disease in disease_names:
        target_col = f'{disease}_delta'
        
        if target_col in df_binary.columns:
            # 1 (유지) → 0 (유지)
            # 0 (개선) + 2 (악화) → 1 (변화)
            df_binary[target_col] = (df_binary[target_col] != 1).astype(int)
    
    return df_binary


def create_binary_loaders(train_df, val_df, test_df, disease_names, 
                          batch_size=64, resample_method='smote'):
    """
    2-Class 데이터 로더 생성
    
    Args:
        train_df, val_df, test_df: DataFrames (already converted to binary)
        disease_names: List of disease names
        batch_size: Batch size
        resample_method: 'smote' or None
    
    Returns:
        train_loaders: Dict of train DataLoaders per disease
        val_loaders: Dict of val DataLoaders per disease
        test_loaders: Dict of test DataLoaders per disease
        input_dims: Dict of input dimensions
    """
    # SMOTE 적용
    if resample_method == 'smote':
        print(f"SMOTE 리샘플링 적용 중...")
        resampled_train = create_resampled_dataset(train_df, resample_method)
        
        # Train 로더: SMOTE 데이터 사용
        train_loaders = {}
        for disease_name in disease_names:
            train_loaders[disease_name] = DataLoader(
                resampled_train, 
                batch_size=batch_size, 
                shuffle=True,
                drop_last=True
            )
        
        # Val/Test 로더: 원본 데이터 사용
        _, val_loaders, test_loaders = DataLoaderManager.create_disease_loaders(
            train_df, val_df, test_df, batch_size=batch_size
        )
    else:
        # SMOTE 없이 원본 데이터 사용
        train_loaders, val_loaders, test_loaders = DataLoaderManager.create_disease_loaders(
            train_df, val_df, test_df, batch_size=batch_size
        )
    
    # 차원 확인
    sample_batch = next(iter(train_loaders[disease_names[0]]))
    input_dims = {
        'diet': sample_batch['diet'].shape[1],
        'demo': sample_batch['demo'].shape[1],
        'life': sample_batch['life'].shape[1],
        'bio': sample_batch['bio'].shape[1],
        'delta': sample_batch['delta'].shape[1],
        'inter': sample_batch['interaction'].shape[1],
        'pca': sample_batch['pca'].shape[1]
    }
    
    return train_loaders, val_loaders, test_loaders, input_dims


def analyze_class_distribution(train_df, val_df, test_df, disease_names):
    """
    클래스 분포 분석
    
    Args:
        train_df, val_df, test_df: DataFrames
        disease_names: List of disease names
    
    Returns:
        None (prints analysis)
    """
    print("\n" + "=" * 80)
    print("2-Class 분포 분석")
    print("=" * 80)
    
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    for idx, disease in enumerate(disease_names):
        target_col = f'{disease}_delta'
        
        if target_col in all_data.columns:
            counts = all_data[target_col].value_counts().sort_index()
            total = len(all_data[target_col].dropna())
            
            print(f"\n{idx+1}. {disease}:")
            print(f"   유지 (0): {counts.get(0, 0):5d} ({counts.get(0, 0)/total*100:5.1f}%)")
            print(f"   변화 (1): {counts.get(1, 0):5d} ({counts.get(1, 0)/total*100:5.1f}%)")
            
            if len(counts) == 2:
                ratio = counts.get(0, 0) / counts.get(1, 0)
                print(f"   불균형 비율: {ratio:.1f}:1", end="")
                
                if ratio < 3:
                    print(" ✅ 우수")
                elif ratio < 5:
                    print(" ✅ 양호")
                else:
                    print(" ⚠️ 보통")


def train_binary_model(model, train_loader, val_loader, device='cuda',
                       lr=1e-4, weight_decay=6e-4, patience=15, max_epochs=100):
    """
    2-Class 모델 학습
    
    Args:
        model: BinaryDiseasePredictor
        train_loader, val_loader: DataLoaders
        device: Device
        lr: Learning rate
        weight_decay: Weight decay
        patience: Early stopping patience
        max_epochs: Maximum epochs
    
    Returns:
        model: Trained model (best weights loaded)
        train_losses: List of train losses
        val_losses: List of val losses
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    
    train_losses = []
    val_losses = []
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            target = batch['target'].long().to(device).squeeze()
            optimizer.zero_grad()
            
            outputs = model(
                batch['diet'].to(device),
                batch['demo'].to(device),
                batch['life'].to(device),
                batch['bio'].to(device),
                batch['delta'].to(device),
                batch['interaction'].to(device),
                batch['pca'].to(device)
            )
            
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        train_loss = total_loss / batch_count
        
        # Validation
        model.eval()
        val_loss = 0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                target = batch['target'].long().to(device).squeeze()
                
                outputs = model(
                    batch['diet'].to(device),
                    batch['demo'].to(device),
                    batch['life'].to(device),
                    batch['bio'].to(device),
                    batch['delta'].to(device),
                    batch['interaction'].to(device),
                    batch['pca'].to(device)
                )
                
                val_loss += criterion(outputs, target).item()
                val_count += 1
        
        val_loss_avg = val_loss / val_count
        scheduler.step(val_loss_avg)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss_avg)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{max_epochs} - Train: {train_loss:.4f}, Val: {val_loss_avg:.4f}")
        
        if early_stopping(val_loss_avg, model):
            print(f"  조기 종료: Epoch {epoch+1}")
            break
    
    # 최적 모델 로드
    early_stopping.load_best_model(model)
    
    return model, train_losses, val_losses


def evaluate_binary_model(model, test_loader, device='cuda'):
    """
    2-Class 모델 평가
    
    Args:
        model: BinaryDiseasePredictor
        test_loader: DataLoader
        device: Device
    
    Returns:
        metrics: Dict with accuracy, f1, roc_auc, pr_auc
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            target = batch['target'].long().to(device).squeeze()
            
            outputs = model(
                batch['diet'].to(device),
                batch['demo'].to(device),
                batch['life'].to(device),
                batch['bio'].to(device),
                batch['delta'].to(device),
                batch['interaction'].to(device),
                batch['pca'].to(device)
            )
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 메트릭 계산
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='binary')
    
    all_probs_array = np.array(all_probs)
    roc_auc = roc_auc_score(all_targets, all_probs_array[:, 1])
    pr_auc = average_precision_score(all_targets, all_probs_array[:, 1])
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def train_and_evaluate_all_diseases(train_df, val_df, test_df, disease_names,
                                    batch_size=64, hidden_dim=64, dropout_rate=0.3,
                                    lr=1e-4, weight_decay=6e-4, max_epochs=100,
                                    use_smote=True):
    """
    전체 질병에 대해 2-Class 모델 학습 및 평가
    
    Args:
        train_df, val_df, test_df: DataFrames
        disease_names: List of disease names
        batch_size: Batch size
        hidden_dim: Hidden dimension
        dropout_rate: Dropout rate
        lr: Learning rate
        weight_decay: Weight decay
        max_epochs: Maximum epochs
        use_smote: Whether to use SMOTE
    
    Returns:
        results: Dict of results for each disease
    """
    # 1. 타겟 변환
    print("\n타겟을 2-Class로 변환 중...")
    train_2class = convert_to_binary_targets(train_df, disease_names)
    val_2class = convert_to_binary_targets(val_df, disease_names)
    test_2class = convert_to_binary_targets(test_df, disease_names)
    
    # 2. 클래스 분포 분석
    analyze_class_distribution(train_2class, val_2class, test_2class, disease_names)
    
    # 3. 데이터 로더 생성
    if use_smote:
        print("\nSMOTE 리샘플링 적용 중...")
        resampled_train = create_resampled_dataset(train_2class, 'smote')
        
        train_loaders = {}
        for disease_name in disease_names:
            train_loaders[disease_name] = DataLoader(
                resampled_train, 
                batch_size=batch_size, 
                shuffle=True
            )
        
        _, val_loaders, test_loaders = DataLoaderManager.create_disease_loaders(
            train_2class, val_2class, test_2class, batch_size=batch_size
        )
    else:
        train_loaders, val_loaders, test_loaders = DataLoaderManager.create_disease_loaders(
            train_2class, val_2class, test_2class, batch_size=batch_size
        )
    
    # 4. 차원 확인
    sample_batch = next(iter(train_loaders[disease_names[0]]))
    input_dims = {
        'diet': sample_batch['diet'].shape[1],
        'demo': sample_batch['demo'].shape[1],
        'life': sample_batch['life'].shape[1],
        'bio': sample_batch['bio'].shape[1],
        'delta': sample_batch['delta'].shape[1],
        'inter': sample_batch['interaction'].shape[1],
        'pca': sample_batch['pca'].shape[1]
    }
    
    print(f"\n데이터 차원:")
    print(f"  diet={input_dims['diet']}, demo={input_dims['demo']}, life={input_dims['life']}, bio={input_dims['bio']}")
    print(f"  delta={input_dims['delta']}, inter={input_dims['inter']}, pca={input_dims['pca']}")
    print(f"  총: {sum(input_dims.values())}차원")
    
    # 5. 각 질병별 학습 및 평가
    results = {}
    criterion = nn.CrossEntropyLoss()
    
    print("\n" + "=" * 80)
    print("2-Class 모델 학습 시작")
    print("=" * 80)
    
    for idx, disease in enumerate(disease_names):
        print(f"\n{'='*60}")
        print(f"질병 {idx+1}/{len(disease_names)}: {disease}")
        print(f"{'='*60}")
        
        # 모델 생성
        model = BinaryDiseasePredictor(
            input_dims, 
            hidden_dim=hidden_dim, 
            dropout_rate=dropout_rate
        ).to(device)
        
        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        early_stopping = EarlyStopping(patience=15, min_delta=0.001)
        
        # 학습
        train_binary_model(
            model, 
            train_loaders[disease], 
            val_loaders[disease], 
            criterion,
            optimizer, 
            scheduler, 
            early_stopping, 
            max_epochs=max_epochs,
            device=device
        )
        
        # 최적 모델 로드
        early_stopping.load_best_model(model)
        
        # 평가
        metrics = evaluate_binary_model(model, test_loaders[disease], device=device)
        results[disease] = metrics
        
        print(f"\n결과:")
        print(f"  - Accuracy: {metrics['accuracy']:.3f}")
        print(f"  - F1 Score: {metrics['f1_score']:.3f}")
        print(f"  - ROC AUC: {metrics['roc_auc']:.3f}")
        print(f"  - PR AUC: {metrics['pr_auc']:.3f}")
        
        # 메모리 정리
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    return results


def print_final_comparison(results_2class, results_3class=None):
    """
    최종 결과 비교 출력
    
    Args:
        results_2class: 2-class 결과
        results_3class: 3-class 결과 (optional)
    """
    print("\n" + "=" * 80)
    print("최종 결과 요약")
    print("=" * 80)
    
    avg_f1 = np.mean([r['f1_score'] for r in results_2class.values()])
    avg_acc = np.mean([r['accuracy'] for r in results_2class.values()])
    avg_roc = np.mean([r['roc_auc'] for r in results_2class.values()])
    avg_pr = np.mean([r['pr_auc'] for r in results_2class.values()])
    
    print(f"\n2-Class (유지 vs 변화):")
    print(f"  - 평균 F1: {avg_f1:.3f}")
    print(f"  - 평균 Accuracy: {avg_acc:.3f}")
    print(f"  - 평균 ROC AUC: {avg_roc:.3f}")
    print(f"  - 평균 PR AUC: {avg_pr:.3f}")
    
    if results_3class:
        print(f"\n3-class (개선/유지/악화):")
        print(f"  - 평균 F1: {results_3class['avg_f1']:.3f}")
        print(f"  - 평균 Accuracy: {results_3class['avg_acc']:.3f}")
        print(f"  - 평균 ROC AUC: {results_3class['avg_roc']:.3f}")
        
        improvement = (avg_f1 - results_3class['avg_f1'])
        improvement_pct = improvement / results_3class['avg_f1'] * 100
        
        print(f"\n개선폭:")
        print(f"  절대값: +{improvement:.3f} F1")
        print(f"  상대값: +{improvement_pct:.1f}%")
    
    print(f"\n임상 활용 가능성 평가:")
    if avg_f1 >= 0.70:
        print("  🎉🎉 우수! (F1 ≥ 0.70) - 임상 활용 적극 권장")
    elif avg_f1 >= 0.60:
        print("  ✅✅ 임상 활용 가능! (F1 ≥ 0.60)")
    elif avg_f1 >= 0.55:
        print("  ✅ 유망 (F1 ≥ 0.55) - 추가 개선 권장")
    else:
        print("  ⚠️ 추가 개선 필요 (F1 < 0.55)")
    
    print("\n질병별 상세 결과:")
    for disease, metrics in results_2class.items():
        print(f"  {disease}:")
        print(f"    F1: {metrics['f1_score']:.3f} | Acc: {metrics['accuracy']:.3f} | ROC: {metrics['roc_auc']:.3f} | PR: {metrics['pr_auc']:.3f}")


if __name__ == "__main__":
    print("Binary Prediction Model (2-Class)")
    print("이 모듈은 main.ipynb에서 import하여 사용하세요.")
    print("\n사용 예시:")
    print("```python")
    print("from binary_prediction_model import train_and_evaluate_all_diseases")
    print("")
    print("results = train_and_evaluate_all_diseases(")
    print("    final_train, final_val, final_test, mets_cols,")
    print("    batch_size=64, hidden_dim=64, use_smote=True")
    print(")")
    print("```")
