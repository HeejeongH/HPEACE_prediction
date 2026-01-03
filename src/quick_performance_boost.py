"""
빠른 성능 개선 스크립트 (기존 환경 활용)
- Loss 함수 개선
- 클래스 가중치 조정
- 앙상블 강화
"""
import sys
import os

# 실행 방법 안내
print("""
╔══════════════════════════════════════════════════════════════╗
║         빠른 성능 개선 가이드 (Quick Performance Boost)        ║
╚══════════════════════════════════════════════════════════════╝

현재 성능: F1 Score = 0.506 (Optuna 최적화 후)
목표 성능: F1 Score > 0.65

📋 개선 전략:

1️⃣ Loss 함수 개선 (가장 효과적!)
   현재: CrossEntropy (기본)
   개선: FocalLoss (gamma=2.0) + 동적 클래스 가중치
   
   예상 향상: +0.05~0.10 F1 Score

2️⃣ 클래스 가중치 동적 계산
   현재: 로그 기반 가중치
   개선: 역빈도 가중치 (Inverse Frequency)
   
   예상 향상: +0.03~0.08 F1 Score

3️⃣ 앙상블 기법 강화
   현재: Voting (RF + LR + DT)
   개선: Stacking (RF + GB + LR + XGBoost)
   
   예상 향상: +0.05~0.12 F1 Score

4️⃣ 데이터 증강 (SMOTE 개선)
   현재: 기본 SMOTE
   개선: SMOTE + Tomek Links (Hybrid)
   
   예상 향상: +0.02~0.05 F1 Score

══════════════════════════════════════════════════════════════

🚀 실행 방법:

main.ipynb를 열고 다음 셀을 추가하세요:

```python
# ============== 성능 개선 코드 시작 ==============

# 1. Loss 함수 개선
from loss_functions import FocalLoss
import numpy as np

def calculate_improved_class_weights(train_loader):
    \"\"\"개선된 클래스 가중치 계산\"\"\"
    class_counts = np.zeros(3)
    
    for batch in train_loader:
        targets = batch['target'].numpy().flatten()
        for t in targets:
            class_counts[t] += 1
    
    total = class_counts.sum()
    # 역빈도 가중치 (더 강한 불균형 보정)
    class_weights = total / (3 * class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 3
    
    print(f"클래스 분포: {class_counts}")
    print(f"개선된 가중치: {class_weights}")
    
    return class_weights

# 2. 개선된 Loss 설정
improved_loss_configs = {
    'FocalLoss_gamma2.0': FocalLoss(gamma=2.0),
    'FocalLoss_gamma2.5': FocalLoss(gamma=2.5),
    'WeightedCE_improved': nn.CrossEntropyLoss(
        weight=torch.tensor(
            calculate_improved_class_weights(train_loader)
        ).float().to(device)
    ),
}

# 3. 각 Loss로 학습 및 비교
best_loss_name = None
best_f1 = 0

for loss_name, criterion in improved_loss_configs.items():
    print(f\"\\n{'='*60}\")
    print(f\"Testing: {loss_name}\")
    print(f\"{'='*60}\")
    
    # 모델 초기화 (최적 하이퍼파라미터 사용)
    model = MultiDiseasePredictor(
        diet_dim=len(preprocessor.ffq_cols),
        demo_dim=len(preprocessor.demo_cols),
        life_dim=len(preprocessor.lifestyle_cols),
        bio_dim=len(preprocessor.biomarker_cols),
        change_dim=actual_change_dim,
        inter_dim=n_interaction_features,
        pca_dim=n_pca_features,
        disease_names=mets_cols,
        dropout_rate=0.357,  # Optuna 최적값
        l1_lambda=0.000366,   # Optuna 최적값
        l2_lambda=0.000596    # Optuna 최적값
    ).to(device)
    
    # 학습 (train_model_with_loss 사용하되 criterion 교체)
    result = train_model_with_improved_loss(
        model, train_loader, val_loader, test_loader,
        disease_name, 'base', criterion
    )
    
    f1 = result['evaluation']['f1_score']
    print(f\"F1 Score: {f1:.4f}\")
    
    if f1 > best_f1:
        best_f1 = f1
        best_loss_name = loss_name
        best_model = result['model']

print(f\"\\n{'='*60}\")
print(f\"🏆 최고 성능: {best_loss_name}\")
print(f\"   F1 Score: {best_f1:.4f} (기존 0.506 대비 +{best_f1-0.506:.4f})\")
print(f\"{'='*60}\")

# ============== 성능 개선 코드 끝 ==============
```

══════════════════════════════════════════════════════════════

📊 예상 결과:

현재 (Baseline):
- F1 Score: 0.506
- Loss: CrossEntropy
- Method: base + SMOTE

개선 후 (Target):
- F1 Score: 0.60~0.65
- Loss: FocalLoss (gamma=2.5) + Weighted CE
- Method: base + SMOTE + Improved Weights

최종 목표 (With Ensemble):
- F1 Score: 0.70+
- Loss: Best from above
- Method: Stacking Ensemble (RF+GB+LR+XGBoost)

══════════════════════════════════════════════════════════════

⚠️ 주의사항:

1. 각 Loss 함수 테스트는 약 10-15분 소요
2. GPU 사용 필수 (CUDA 확인)
3. 메모리 부족 시 batch_size를 32로 감소
4. Early Stopping으로 자동 중단 (patience=15)

══════════════════════════════════════════════════════════════

📝 다음 단계:

1단계: Loss 함수 최적화 (현재)
   → main.ipynb에서 위 코드 실행
   
2단계: 앙상블 강화
   → sklearn_ensemble 방식 개선
   → XGBoost, LightGBM 추가
   
3단계: 데이터 증강
   → SMOTE + Tomek Links
   → ADASYN 테스트
   
4단계: 하이퍼파라미터 재최적화
   → Optuna trials 100회 (현재 30회)
   → 새로운 탐색 공간 추가

══════════════════════════════════════════════════════════════

✨ 자동화 스크립트 (선택사항):

만약 Jupyter 대신 Python 스크립트로 실행하고 싶다면:

```bash
cd /home/user/webapp/src
python3 run_performance_improvement.py
```

이 파일을 만들려면 알려주세요!

══════════════════════════════════════════════════════════════
""")

if __name__ == '__main__':
    print("\n이 파일은 가이드입니다.")
    print("실제 실행은 main.ipynb에서 위 코드를 사용하세요.\n")
