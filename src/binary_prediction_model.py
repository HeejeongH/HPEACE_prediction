"""
2-Class 예측 모델: 악화/변화 감지

목적: 대사증후군 질병의 변화를 예측
- 'worsening': 0=비악화(개선+유지), 1=악화
- 'change': 0=유지, 1=변화(개선+악화)

git 히스토리(커밋 d7bffb8)에서 복원. create_binary_loaders()의 SMOTE 분기가
5개 질병의 리샘플링 결과를 구분 없이 하나의 DataLoader로 합쳐 모든 질병이
서로 다른 질병의 라벨까지 섞어서 학습하던 버그를 수정함
(f1_improvement_evaluation.py가 쓰는 질병별 Subset 필터링 패턴과 동일하게 정정).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from data_import import SingleDietImpactDataset, DataLoaderManager
from resampling import create_resampled_dataset
from MetS_prediction_model import EarlyStopping
from torch.utils.data import DataLoader, Subset
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


class BinaryDiseasePredictor(nn.Module):
    """
    2-Class 질병 변화 예측 모델
    Architecture: 입력(diet+demo+life+bio+delta+inter+pca) -> 4층 MLP -> 2-class
    """

    def __init__(self, input_dims, hidden_dim=64, dropout_rate=0.3):
        super().__init__()

        total_dim = sum(input_dims.values())

        self.encoder = nn.Sequential(
            nn.Linear(total_dim, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, diet, demo, life, bio, delta, inter, pca):
        x = torch.cat([diet, demo, life, bio, delta, inter, pca], dim=1)
        return self.encoder(x)


def convert_to_binary_targets(df, disease_names, target_type='worsening'):
    """
    3-class 타겟(0=개선,1=유지,2=악화)을 2-class로 변환

    target_type:
        'worsening': 0=비악화(개선+유지), 1=악화 (임상적으로 조기개입에 의미있음, 기본값)
        'change'   : 0=유지, 1=변화(개선+악화)
    """
    df_binary = df.copy()

    for disease in disease_names:
        target_col = f'{disease}_delta'

        if target_col in df_binary.columns:
            if target_type == 'worsening':
                df_binary[target_col] = (df_binary[target_col] == 2).astype(int)
            elif target_type == 'change':
                df_binary[target_col] = (df_binary[target_col] != 1).astype(int)
            else:
                raise ValueError(f"Invalid target_type: {target_type}. Use 'worsening' or 'change'.")

    return df_binary


def create_binary_loaders(train_df, val_df, test_df, disease_names,
                           batch_size=64, resample_method='smote'):
    """
    2-Class 데이터 로더 생성 (질병별 SMOTE 적용, 질병별 Subset으로 분리 — 버그 수정판)
    """
    if resample_method == 'smote':
        resampled_train = create_resampled_dataset(train_df, resample_method)

        train_loaders = {}
        for disease_name in disease_names:
            disease_indices = [
                i for i, item in enumerate(resampled_train)
                if item['disease_name'] == disease_name
            ]
            subset = Subset(resampled_train, disease_indices)
            train_loaders[disease_name] = DataLoader(
                subset, batch_size=batch_size, shuffle=True, drop_last=True
            )

        _, val_loaders, test_loaders = DataLoaderManager.create_disease_loaders(
            train_df, val_df, test_df, batch_size=batch_size
        )
    else:
        train_loaders, val_loaders, test_loaders = DataLoaderManager.create_disease_loaders(
            train_df, val_df, test_df, batch_size=batch_size
        )

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
            print(f"   0: {counts.get(0, 0):5d} ({counts.get(0, 0)/total*100:5.1f}%)")
            print(f"   1: {counts.get(1, 0):5d} ({counts.get(1, 0)/total*100:5.1f}%)")

            if len(counts) == 2 and counts.get(1, 0) > 0:
                ratio = counts.get(0, 0) / counts.get(1, 0)
                print(f"   불균형 비율: {ratio:.1f}:1")


def train_binary_model(model, train_loader, val_loader, device='cuda',
                        lr=1e-4, weight_decay=6e-4, patience=15, max_epochs=100,
                        class_weight=None):
    """2-Class 모델 학습. class_weight=[w0, w1]로 소수 클래스 가중치 부여 가능."""
    if class_weight is not None:
        weights = torch.tensor(class_weight).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"  클래스 가중치: {class_weight}")
    else:
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

    early_stopping.load_best_model(model)

    return model, train_losses, val_losses


def evaluate_binary_model(model, test_loader, device='cuda'):
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

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)

    all_probs_array = np.array(all_probs)
    try:
        roc_auc = roc_auc_score(all_targets, all_probs_array[:, 1])
        pr_auc = average_precision_score(all_targets, all_probs_array[:, 1])
    except ValueError:
        roc_auc, pr_auc = float('nan'), float('nan')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
