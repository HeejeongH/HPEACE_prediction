"""
2-Class 예측 모델 실행 스크립트

목적: 대사증후군 질병의 변화를 예측하는 2-class 모델 학습 및 평가
- Class 0: 유지 (질병 상태 변화 없음)
- Class 1: 변화 (개선 또는 악화)

실행 방법:
    python run_binary_model.py

결과 저장 위치:
    ../result/binary_model/
"""

import os
import sys
import pickle
import pandas as pd
import torch
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트
from data_import import DataPreprocessor
from feature_engineering import apply_to_all_data
from binary_prediction_model import (
    BinaryDiseasePredictor,
    convert_to_binary_targets,
    create_binary_loaders,
    train_binary_model,
    evaluate_binary_model
)

# CUDA 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")
print("=" * 80)

# 저장 디렉토리 생성
save_dir = '../result/binary_model'
os.makedirs(save_dir, exist_ok=True)

print("📊 Step 1: 데이터 준비")
print("-" * 80)

# 데이터 로드
data_path = '../data/total_again.xlsx'
data_processor = DataPreprocessor(file_path=data_path, seed=42, normalize=False)

# 데이터 전처리
final_train, final_val, final_test, _ = data_processor.process_all(
    normalize=True,
    selection_strategy='max_disease_change'
)

# 피처 엔지니어링 적용
final_train, final_val, final_test, actual_change_dim = apply_to_all_data(
    final_train, final_val, final_test
)

print(f"✅ 데이터 준비 완료")
print(f"   - Train: {len(final_train)}")
print(f"   - Val: {len(final_val)}")
print(f"   - Test: {len(final_test)}")
print(f"   - Change Dim: {actual_change_dim}")
print()

# 질병 목록
disease_names = [
    'Increased waist circumference',
    'Elevated blood pressure',
    'Impaired fasting glucose',
    'Elevated triglycerides',
    'Decreased HDL-C'
]

print("🔄 Step 2: 2-Class 타겟 변환 및 클래스 분포 확인")
print("-" * 80)

# 2-class 타겟 변환
final_train = convert_to_binary_targets(final_train, disease_names)
final_val = convert_to_binary_targets(final_val, disease_names)
final_test = convert_to_binary_targets(final_test, disease_names)

# 클래스 분포 확인
print("\n📊 2-Class 클래스 분포:")
for disease in disease_names:
    target_col = f'{disease}_delta'
    if target_col in final_train.columns:
        train_dist = final_train[target_col].value_counts().sort_index()
        total = len(final_train)
        maintain = train_dist.get(0, 0)
        change = train_dist.get(1, 0)
        
        imbalance_ratio = maintain / change if change > 0 else float('inf')
        
        print(f"\n{disease}:")
        print(f"  - 유지(0): {maintain} ({maintain/total*100:.1f}%)")
        print(f"  - 변화(1): {change} ({change/total*100:.1f}%)")
        print(f"  - 불균형 비율: {imbalance_ratio:.1f}:1")

print()

print("🔧 Step 3: 데이터 로더 생성 (SMOTE 적용)")
print("-" * 80)

# 데이터 로더 생성
train_loaders, val_loaders, test_loaders, input_dims = create_binary_loaders(
    final_train, final_val, final_test,
    disease_names=disease_names,
    batch_size=64,
    resample_method='smote'
)

print(f"✅ 데이터 로더 생성 완료")
print(f"   - Input Dimensions:")
for key, value in input_dims.items():
    print(f"     {key}: {value}")
print()

print("🎯 Step 4: 모델 학습 및 평가")
print("-" * 80)

# 하이퍼파라미터 설정
hyperparams = {
    'hidden_dim': 64,
    'dropout_rate': 0.3,
    'lr': 1e-4,
    'weight_decay': 6e-4,
    'patience': 15,
    'max_epochs': 100
}

print(f"하이퍼파라미터:")
for key, value in hyperparams.items():
    print(f"  - {key}: {value}")
print()

# 질병별 학습 및 평가
final_results = {}

for idx, disease in enumerate(disease_names, 1):
    print(f"\n{'='*80}")
    print(f"질병 {idx}/{len(disease_names)}: {disease}")
    print(f"{'='*80}")
    
    # 데이터 로더 가져오기
    train_loader = train_loaders[disease]
    val_loader = val_loaders[disease]
    test_loader = test_loaders[disease]
    
    # 모델 생성
    model = BinaryDiseasePredictor(
        input_dims=input_dims,
        hidden_dim=hyperparams['hidden_dim'],
        dropout_rate=hyperparams['dropout_rate']
    ).to(device)
    
    # 모델 학습
    print(f"\n🔨 학습 시작...")
    best_model, train_losses, val_losses = train_binary_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=hyperparams['lr'],
        weight_decay=hyperparams['weight_decay'],
        patience=hyperparams['patience'],
        max_epochs=hyperparams['max_epochs']
    )
    
    # 모델 평가
    print(f"\n📊 평가 중...")
    metrics = evaluate_binary_model(
        model=best_model,
        test_loader=test_loader,
        device=device
    )
    
    # 결과 저장
    final_results[disease] = {
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc'],
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    # 결과 출력
    print(f"\n✅ {disease} 결과:")
    print(f"   - Accuracy: {metrics['accuracy']:.4f}")
    print(f"   - F1 Score: {metrics['f1']:.4f}")
    print(f"   - ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"   - PR AUC: {metrics['pr_auc']:.4f}")
    
    # 메모리 정리
    del model, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()

print()
print("=" * 80)
print("📈 최종 결과 요약")
print("=" * 80)

# 평균 성능 계산
avg_accuracy = sum([r['accuracy'] for r in final_results.values()]) / len(final_results)
avg_f1 = sum([r['f1'] for r in final_results.values()]) / len(final_results)
avg_roc_auc = sum([r['roc_auc'] for r in final_results.values()]) / len(final_results)
avg_pr_auc = sum([r['pr_auc'] for r in final_results.values()]) / len(final_results)

print(f"\n평균 성능:")
print(f"  - 평균 Accuracy: {avg_accuracy:.4f}")
print(f"  - 평균 F1 Score: {avg_f1:.4f}")
print(f"  - 평균 ROC AUC: {avg_roc_auc:.4f}")
print(f"  - 평균 PR AUC: {avg_pr_auc:.4f}")
print()

# 질병별 상세 성능
print("\n질병별 상세 성능:")
for disease, metrics in final_results.items():
    print(f"\n{disease}:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - F1 Score: {metrics['f1']:.4f}")
    print(f"  - ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  - PR AUC: {metrics['pr_auc']:.4f}")

print()
print("=" * 80)
print("💾 결과 저장")
print("=" * 80)

# 결과 저장
# 1. 상세 결과 (CSV)
detailed_results = []
for disease, metrics in final_results.items():
    detailed_results.append({
        'Disease': disease,
        'Accuracy': metrics['accuracy'],
        'F1_Score': metrics['f1'],
        'ROC_AUC': metrics['roc_auc'],
        'PR_AUC': metrics['pr_auc']
    })

df_detailed = pd.DataFrame(detailed_results)
detailed_path = os.path.join(save_dir, 'detailed_results.csv')
df_detailed.to_csv(detailed_path, index=False)
print(f"✅ 상세 결과 저장: {detailed_path}")

# 2. 요약 결과 (CSV)
summary_results = {
    'Average_Accuracy': [avg_accuracy],
    'Average_F1': [avg_f1],
    'Average_ROC_AUC': [avg_roc_auc],
    'Average_PR_AUC': [avg_pr_auc],
    'Model_Type': ['2-Class Binary Prediction'],
    'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
}

df_summary = pd.DataFrame(summary_results)
summary_path = os.path.join(save_dir, 'summary_results.csv')
df_summary.to_csv(summary_path, index=False)
print(f"✅ 요약 결과 저장: {summary_path}")

# 3. 전체 결과 (Pickle)
results_dict = {
    'final_results': final_results,
    'averages': {
        'accuracy': avg_accuracy,
        'f1': avg_f1,
        'roc_auc': avg_roc_auc,
        'pr_auc': avg_pr_auc
    },
    'hyperparameters': hyperparams,
    'disease_names': disease_names,
    'input_dims': input_dims
}

pickle_path = os.path.join(save_dir, 'binary_model_results.pkl')
with open(pickle_path, 'wb') as f:
    pickle.dump(results_dict, f)
print(f"✅ 전체 결과 저장: {pickle_path}")

print()
print("=" * 80)
print("🎯 임상 활용 가능성 평가")
print("=" * 80)

if avg_f1 >= 0.60:
    print("✅ 임상 활용 가능 (F1 ≥ 0.60)")
    print("   → 다음 단계: SHAP 해석, 웹 데모 구축")
elif avg_f1 >= 0.55:
    print("⚠️ 임상 활용 가능성 있음 (0.55 ≤ F1 < 0.60)")
    print("   → 다음 단계: 교차 검증, 앙상블 강화")
else:
    print("❌ 추가 개선 필요 (F1 < 0.55)")
    print("   → 다음 단계: 모델 아키텍처 개선, 하이퍼파라미터 재조정")

print()
print("=" * 80)
print("✅ 2-Class 예측 모델 학습 완료!")
print("=" * 80)
