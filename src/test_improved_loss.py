"""
Loss 함수 개선 테스트 스크립트
기존 main.ipynb 코드를 활용하여 개선된 Loss 함수들을 테스트
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("        Loss 함수 개선 테스트 (Performance Boost Test)")
print("="*70)
print("\n현재 베이스라인: F1 Score = 0.506")
print("목표: F1 Score > 0.60\n")

# Step 1: 모듈 확인
print("[1/5] 필수 모듈 확인 중...")
try:
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, accuracy_score
    print("  ✓ torch, numpy, pandas, sklearn 로드 완료")
except ImportError as e:
    print(f"  ✗ 모듈 import 실패: {e}")
    print("\n해결 방법:")
    print("  pip install torch scikit-learn numpy pandas")
    sys.exit(1)

# Step 2: 프로젝트 모듈 import
print("\n[2/5] 프로젝트 모듈 로딩 중...")
try:
    from data_import import DataPreprocessor, DataLoaderManager
    from feature_engineering import FeatureEngineer
    from MetS_prediction_model import MultiDiseasePredictor, EarlyStopping
    from train_eval_function import evaluate_model_custom
    from loss_functions import (
        loss_methods_configs, 
        improved_loss_methods_configs,
        calculate_improved_class_weights
    )
    from resampling import prepare_balanced_data
    from utils import set_seed
    
    print("  ✓ 모든 프로젝트 모듈 로드 완료")
except ImportError as e:
    print(f"  ✗ 프로젝트 모듈 import 실패: {e}")
    print("\n현재 경로:", os.getcwd())
    print("파일 확인:", os.listdir('.'))
    sys.exit(1)

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  ✓ Device: {device}")
set_seed(42)

# Step 3: 데이터 준비
print("\n[3/5] 데이터 준비 중...")
try:
    data_path = '../data/total_again.xlsx'
    disease_name = 'mets_transition'
    
    # 데이터 로드
    preprocessor = DataPreprocessor(data_path, seed=42)
    df = preprocessor.load_and_preprocess_data()
    print(f"  ✓ 데이터 로드 완료: {df.shape}")
    
    # 피처 엔지니어링
    engineer = FeatureEngineer(preprocessor)
    df = engineer.create_interaction_features(df)
    df = engineer.create_pca_features(df)
    print(f"  ✓ 피처 엔지니어링 완료")
    
    # 데이터 선택 (max_disease_change)
    df = preprocessor.select_max_disease_change_per_patient(df)
    print(f"  ✓ 데이터 선택 후: {df.shape}")
    
    # 데이터 분할
    from sklearn.model_selection import train_test_split
    
    target_col = disease_name
    train_val_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df[target_col]
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.15/(1-0.15), random_state=42, stratify=train_val_df[target_col]
    )
    
    print(f"  ✓ 데이터 분할: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # SMOTE 적용
    train_df, val_df, test_df = prepare_balanced_data(
        train_df, val_df, test_df, disease_name, method='smote'
    )
    print(f"  ✓ SMOTE 적용 완료: Train={len(train_df)}")
    
    # DataLoader 생성
    loader_manager = DataLoaderManager(
        train_df, val_df, test_df,
        disease_names=[disease_name],
        batch_size=64,
        feature_engineer=engineer
    )
    
    train_loader = loader_manager.train_loaders[disease_name]
    val_loader = loader_manager.val_loaders[disease_name]
    test_loader = loader_manager.test_loaders[disease_name]
    
    print(f"  ✓ DataLoader 생성 완료")
    
except Exception as e:
    print(f"  ✗ 데이터 준비 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Loss 함수 설정
print("\n[4/5] 개선된 Loss 함수 설정 중...")
try:
    # 개선된 Loss 설정 가져오기
    improved_loss_configs = improved_loss_methods_configs(
        {disease_name: train_loader}, disease_name, device
    )
    
    # 테스트할 Loss 함수 선택 (빠른 테스트를 위해 일부만)
    test_losses = {
        'CrossEntropy': improved_loss_configs['CrossEntropy'],
        'FocalLoss_gamma2.0': improved_loss_configs['FocalLoss_gamma2.0'],
        'WeightedCE_inverse': improved_loss_configs['WeightedCE_inverse'],
    }
    
    print(f"  ✓ {len(test_losses)}개 Loss 함수 준비 완료")
    print(f"    - {', '.join(test_losses.keys())}")
    
except Exception as e:
    print(f"  ✗ Loss 함수 설정 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: 모델 학습 및 평가
print("\n[5/5] 모델 학습 및 평가 시작...")
print(f"  각 Loss 함수당 약 5-10분 소요 예상 (Early Stopping 적용)")
print("="*70)

results = {}

# 차원 정보
dims = {
    'diet_dim': len(preprocessor.ffq_cols),
    'demo_dim': len(preprocessor.demo_cols),
    'life_dim': len(preprocessor.lifestyle_cols),
    'bio_dim': len(preprocessor.biomarker_cols),
    'change_dim': len([col for col in df.columns if '_delta' in col and col != target_col]),
    'inter_dim': engineer.n_interaction_features,
    'pca_dim': engineer.n_pca_features
}

print(f"\n모델 차원 정보: {dims}")
print("="*70)

for loss_name, criterion in test_losses.items():
    print(f"\n\n{'='*70}")
    print(f"  Testing: {loss_name}")
    print(f"{'='*70}\n")
    
    try:
        # 모델 초기화 (Optuna 최적 하이퍼파라미터 사용)
        model = MultiDiseasePredictor(
            **dims,
            disease_names=[disease_name],
            dropout_rate=0.357,   # Optuna 최적값
            l1_lambda=0.000366,   # Optuna 최적값
            l2_lambda=0.000596    # Optuna 최적값
        ).to(device)
        
        # 학습 설정
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=6e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        early_stopping = EarlyStopping(patience=15, min_delta=0.001)
        
        # 학습 루프
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(100):
            # Train
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                diet_data = batch['diet'].to(device)
                demo_data = batch['demo'].to(device)
                life_data = batch['life'].to(device)
                bio_data = batch['bio'].to(device)
                change_data = batch['delta'].to(device)
                inter_data = batch['interaction'].to(device)
                pca_data = batch['pca'].to(device)
                target = batch['target'].long().to(device).squeeze()
                
                outputs = model(
                    diet_data, demo_data, life_data, bio_data, 
                    change_data, inter_data, pca_data, disease_name
                )
                loss = criterion(outputs['disease_logits'], target) + model.regularization_loss()
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            train_loss = total_loss / batch_count
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_total_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    diet_data = batch['diet'].to(device)
                    demo_data = batch['demo'].to(device)
                    life_data = batch['life'].to(device)
                    bio_data = batch['bio'].to(device)
                    change_data = batch['delta'].to(device)
                    inter_data = batch['interaction'].to(device)
                    pca_data = batch['pca'].to(device)
                    target = batch['target'].long().to(device).squeeze()
                    
                    outputs = model(
                        diet_data, demo_data, life_data, bio_data,
                        change_data, inter_data, pca_data, disease_name
                    )
                    val_loss = criterion(outputs['disease_logits'], target)
                    val_total_loss += val_loss.item()
                    val_batch_count += 1
            
            val_loss_avg = val_total_loss / val_batch_count
            val_losses.append(val_loss_avg)
            
            scheduler.step(val_loss_avg)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss_avg:.4f}")
            
            if early_stopping(val_loss_avg, model):
                print(f"\n✓ Early stopping at epoch {epoch}")
                break
        
        # 최고 모델 로드
        early_stopping.load_best_model(model)
        
        # 평가
        accuracy, f1, roc_aucs, pr_aucs = evaluate_model_custom(
            model, test_loader, device, disease_name, 'deep'
        )
        
        results[loss_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_aucs': roc_aucs,
            'pr_aucs': pr_aucs
        }
        
        print(f"\n{'='*70}")
        print(f"  {loss_name} 결과:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC (Macro): {roc_aucs.get('Macro', 0):.4f}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n✗ {loss_name} 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        continue

# 최종 결과 출력
print("\n\n" + "="*70)
print("               최종 결과 요약 (Final Results)")
print("="*70)
print(f"\n{'Loss 함수':<30} {'Accuracy':>10} {'F1 Score':>10} {'ROC AUC':>10}")
print("-"*70)

for loss_name, result in results.items():
    print(f"{loss_name:<30} {result['accuracy']:>10.4f} {result['f1_score']:>10.4f} {result['roc_aucs'].get('Macro', 0):>10.4f}")

# 최고 성능 찾기
if results:
    best_loss = max(results.items(), key=lambda x: x[1]['f1_score'])
    print("\n" + "="*70)
    print(f"🏆 최고 성능: {best_loss[0]}")
    print(f"   F1 Score: {best_loss[1]['f1_score']:.4f}")
    print(f"   개선폭: +{best_loss[1]['f1_score'] - 0.506:.4f} (vs 베이스라인 0.506)")
    print("="*70)
else:
    print("\n결과 없음 (모든 테스트 실패)")

print("\n테스트 완료!")
