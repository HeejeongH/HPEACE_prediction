# detailed_mets_analysis.py
"""
Ver3: MetS 분류 성능 상세 분석
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
# Windows: plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_latest_results():
    """최신 결과 파일 로드"""
    paired_files = glob.glob('./results/paired_data_*.csv')
    if not paired_files:
        raise FileNotFoundError("결과 파일을 찾을 수 없습니다!")
    
    latest_file = max(paired_files, key=os.path.getmtime)
    print(f"✅ 데이터 로드: {latest_file}")
    return pd.read_csv(latest_file)

def analyze_mets_performance(df):
    """MetS 분류 성능 상세 분석"""
    
    # 1. 클래스 분포
    print("\n" + "="*80)
    print("📊 클래스 분포")
    print("="*80)
    class_counts = df['mets_change_pattern'].value_counts()
    class_props = df['mets_change_pattern'].value_counts(normalize=True) * 100
    
    dist_df = pd.DataFrame({
        '클래스': class_counts.index,
        '샘플 수': class_counts.values,
        '비율 (%)': class_props.values.round(2)
    })
    print(dist_df.to_string(index=False))
    
    # 클래스 불균형
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"\n⚠️  클래스 불균형 비율: {imbalance_ratio:.1f}:1")
    
    # 2. 모델 로드 및 예측
    print("\n" + "="*80)
    print("🤖 모델 로드 및 예측")
    print("="*80)
    
    import sys
    sys.path.insert(0, './src')
    from mets_prediction_model import MetSPredictor
    
    # 최신 모델 찾기
    model_dirs = glob.glob('./results/models/mets_predictor_*')
    latest_model_dir = max(model_dirs, key=os.path.getmtime)
    print(f"✅ 모델 디렉토리: {latest_model_dir}")
    
    # 예측
    predictor = MetSPredictor()
    predictor.load_models(latest_model_dir)
    
    y_true = df['mets_change_pattern'].values
    y_pred, y_prob = predictor.predict(df)
    
    print(f"✅ 예측 완료: {len(y_pred):,}개 샘플")
    
    # 3. 상세 성능 지표
    print("\n" + "="*80)
    print("📈 클래스별 상세 성능")
    print("="*80)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    classes = [c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics_df = pd.DataFrame({
        '클래스': classes,
        'Precision': [report[c]['precision'] for c in classes],
        'Recall': [report[c]['recall'] for c in classes],
        'F1-Score': [report[c]['f1-score'] for c in classes],
        '샘플 수': [int(report[c]['support']) for c in classes]
    })
    
    print(metrics_df.to_string(index=False))
    
    print(f"\n📊 전체 평균:")
    print(f"   - Accuracy:        {report['accuracy']:.4f}")
    print(f"   - Macro Avg F1:    {report['macro avg']['f1-score']:.4f}")
    print(f"   - Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")
    
    # 4. Confusion Matrix
    print("\n" + "="*80)
    print("🔢 Confusion Matrix")
    print("="*80)
    
    class_names = sorted(df['mets_change_pattern'].unique())
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    print("\n절대 개수:")
    print(cm_df.to_string())
    
    # 정규화
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    
    print("\n정규화 (비율, 행 합 = 1.0):")
    print(cm_norm_df.round(4).to_string())
    
    # 5. 오분류 패턴 분석
    print("\n" + "="*80)
    print("🔍 주요 오분류 패턴")
    print("="*80)
    
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                error_rate = cm[i, j] / cm[i].sum()
                if error_rate > 0.1:  # 10% 이상
                    print(f"   실제 '{true_class}' → 예측 '{pred_class}': "
                          f"{cm[i, j]}건 ({error_rate*100:.1f}%)")
    
    # 6. 시각화
    print("\n" + "="*80)
    print("📊 시각화 생성")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 6-1. Confusion Matrix (절대)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0, 0], cbar_kws={'label': '샘플 수'})
    axes[0, 0].set_title('Confusion Matrix (절대 개수)', fontweight='bold')
    axes[0, 0].set_ylabel('실제 (Actual)')
    axes[0, 0].set_xlabel('예측 (Predicted)')
    
    # 6-2. Confusion Matrix (정규화)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0, 1], cbar_kws={'label': '비율'})
    axes[0, 1].set_title('Confusion Matrix (정규화)', fontweight='bold')
    axes[0, 1].set_ylabel('실제 (Actual)')
    axes[0, 1].set_xlabel('예측 (Predicted)')
    
    # 6-3. 클래스별 성능 비교
    x = np.arange(len(metrics_df))
    width = 0.25
    
    ax = axes[1, 0]
    ax.bar(x - width, metrics_df['Precision'], width, label='Precision')
    ax.bar(x, metrics_df['Recall'], width, label='Recall')
    ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score')
    
    ax.set_xlabel('클래스')
    ax.set_ylabel('점수')
    ax.set_title('클래스별 Precision, Recall, F1-Score', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['클래스'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # 6-4. F1-Score vs 샘플 수
    ax = axes[1, 1]
    scatter = ax.scatter(metrics_df['샘플 수'], metrics_df['F1-Score'],
                        s=300, alpha=0.6, c=metrics_df['F1-Score'],
                        cmap='RdYlGn', vmin=0, vmax=1)
    
    for i, row in metrics_df.iterrows():
        ax.annotate(row['클래스'],
                   (row['샘플 수'], row['F1-Score']),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=9, ha='left')
    
    ax.set_xlabel('샘플 수')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score vs 샘플 수 (클래스 불균형 영향)', fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='F1-Score')
    
    plt.tight_layout()
    
    output_path = './results/detailed_mets_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 저장: {output_path}")
    
    plt.show()
    
    return metrics_df, cm_df, class_counts

if __name__ == "__main__":
    df = load_latest_results()
    metrics_df, cm_df, class_counts = analyze_mets_performance(df)
