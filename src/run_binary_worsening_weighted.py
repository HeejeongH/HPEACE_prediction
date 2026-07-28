"""
Priority 3: 2-Class 악화(worsening) 예측 모델 학습/평가 (클래스 가중치 적용)

- Class 0: 비악화 (개선 + 유지)
- Class 1: 악화
- 클래스 가중치 = min(sqrt(불균형비율), 10.0), weight=[1.0, w]

git 히스토리에서 삭제됐던 run_binary_worsening.py + run_binary_weighted.py를
병합하고, binary_prediction_model.py의 질병별 리샘플링 버그를 수정한 뒤 재실행.

실행: python run_binary_worsening_weighted.py
결과: ../result/binary_worsening_weighted/
"""
import os
import sys
import pickle
import pandas as pd
import torch
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_import import DataPreprocessor
from feature_engineering import apply_to_all_data
from binary_prediction_model import (
    BinaryDiseasePredictor,
    convert_to_binary_targets,
    create_binary_loaders,
    train_binary_model,
    evaluate_binary_model,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

save_dir = '../result/binary_worsening_weighted'
os.makedirs(save_dir, exist_ok=True)

print("Step 1: 데이터 준비")
data_path = '../data/total_again.xlsx'
data_processor = DataPreprocessor(file_path=data_path, seed=42, normalize=False)

final_train, final_val, final_test, _ = data_processor.process_all(
    normalize=True, selection_strategy='max_disease_change'
)
final_train, final_val, final_test, actual_change_dim, _pca = apply_to_all_data(
    final_train, final_val, final_test
)
print(f"Train: {len(final_train)}, Val: {len(final_val)}, Test: {len(final_test)}")

disease_names = [
    'Increased waist circumference',
    'Elevated blood pressure',
    'Impaired fasting glucose',
    'Elevated triglycerides',
    'Decreased HDL-C',
]

print("\nStep 2: 타겟 변환 (악화 예측: 0=비악화, 1=악화)")
final_train = convert_to_binary_targets(final_train, disease_names, target_type='worsening')
final_val = convert_to_binary_targets(final_val, disease_names, target_type='worsening')
final_test = convert_to_binary_targets(final_test, disease_names, target_type='worsening')

print("\n클래스 분포 및 가중치 (train 기준):")
class_weights_per_disease = {}
for disease in disease_names:
    target_col = f'{disease}_delta'
    dist = final_train[target_col].value_counts().sort_index()
    total = len(final_train)
    no_w, w = dist.get(0, 0), dist.get(1, 0)
    ratio = no_w / w if w > 0 else 1.0
    weight = min(ratio ** 0.5, 10.0)
    class_weights_per_disease[disease] = [1.0, weight]
    print(f"  {disease}: 비악화={no_w}({no_w/total*100:.1f}%) 악화={w}({w/total*100:.1f}%) "
          f"비율={ratio:.1f}:1 가중치=[1.0, {weight:.2f}]")

print("\nStep 3: 데이터 로더 생성 (질병별 SMOTE)")
train_loaders, val_loaders, test_loaders, input_dims = create_binary_loaders(
    final_train, final_val, final_test,
    disease_names=disease_names, batch_size=64, resample_method='smote'
)
print(f"Input dims: {input_dims} (합={sum(input_dims.values())})")

hyperparams = {
    'hidden_dim': 128, 'dropout_rate': 0.3, 'lr': 1e-4,
    'weight_decay': 6e-4, 'patience': 15, 'max_epochs': 100,
}
print(f"\nStep 4: 학습 (hyperparams={hyperparams})")

final_results = {}
for idx, disease in enumerate(disease_names, 1):
    print(f"\n{'='*70}\n질병 {idx}/{len(disease_names)}: {disease}\n{'='*70}")

    model = BinaryDiseasePredictor(
        input_dims=input_dims,
        hidden_dim=hyperparams['hidden_dim'],
        dropout_rate=hyperparams['dropout_rate'],
    ).to(device)

    class_weight = class_weights_per_disease[disease]
    best_model, train_losses, val_losses = train_binary_model(
        model=model,
        train_loader=train_loaders[disease],
        val_loader=val_loaders[disease],
        device=device,
        lr=hyperparams['lr'],
        weight_decay=hyperparams['weight_decay'],
        patience=hyperparams['patience'],
        max_epochs=hyperparams['max_epochs'],
        class_weight=class_weight,
    )

    metrics = evaluate_binary_model(best_model, test_loaders[disease], device=device)
    final_results[disease] = {**metrics, 'class_weight': class_weight}

    print(f"  -> Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
          f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f}")

    del model, best_model
    torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("최종 결과 (악화 예측, 클래스 가중치 적용)")
print("=" * 70)

avg = {k: sum(r[k] for r in final_results.values()) / len(final_results)
       for k in ['accuracy', 'f1', 'roc_auc', 'pr_auc']}
print(f"평균: Acc={avg['accuracy']:.4f} F1={avg['f1']:.4f} "
      f"ROC-AUC={avg['roc_auc']:.4f} PR-AUC={avg['pr_auc']:.4f}")

detailed = [{'Disease': d, 'Accuracy': m['accuracy'], 'F1_Score': m['f1'],
             'ROC_AUC': m['roc_auc'], 'PR_AUC': m['pr_auc'],
             'Class_Weight_1': m['class_weight'][1]}
            for d, m in final_results.items()]
pd.DataFrame(detailed).to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)

pd.DataFrame([{**avg, 'Timestamp': datetime.now().isoformat()}]).to_csv(
    os.path.join(save_dir, 'summary_results.csv'), index=False
)

with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
    pickle.dump({'final_results': final_results, 'averages': avg,
                 'hyperparameters': hyperparams}, f)

print(f"\n저장 완료: {save_dir}")
