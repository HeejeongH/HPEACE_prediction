"""
Ver3: MetS 분류 성능 상세 분석
목적: MetS 4-class 분류 모델의 상세 성능 분석
- 클래스별 Precision, Recall, F1-score
- Confusion Matrix 상세 분석
- 클래스 불균형 영향 분석
- 예측 확률 분포 분석

Author: SNUH Prediction Team
Date: 2026-01-03
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import sys

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
# Windows: plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_latest_results():
    """최신 결과 파일 로드"""
    print("\n" + "="*80)
    print("📂 STEP 1: 결과 파일 로드")
    print("="*80)
    
    paired_files = glob.glob('./results/paired_data_*.csv')
    if not paired_files:
        raise FileNotFoundError("결과 파일을 찾을 수 없습니다!")
    
    latest_file = max(paired_files, key=os.path.getmtime)
    print(f"✅ 데이터 로드: {latest_file}")
    
    df = pd.read_csv(latest_file)
    print(f"   - 샘플 수: {len(df):,}개")
    print(f"   - 변수 수: {len(df.columns):,}개")
    
    return df

def analyze_class_distribution(df):
    """클래스 분포 분석"""
    print("\n" + "="*80)
    print("📊 STEP 2: MetS 변화 패턴 분포")
    print("="*80)
    
    # 컬럼명 확인
    mets_col = 'mets_transition'  # 수정된 부분
    
    if mets_col not in df.columns:
        print(f"❌ '{mets_col}' 컬럼을 찾을 수 없습니다!")
        print(f"   사용 가능한 MetS 관련 컬럼:")
        for col in df.columns:
            if 'mets' in col.lower():
                print(f"   - {col}")
        return None
    
    # 클래스 분포
    class_counts = df[mets_col].value_counts()
    class_props = df[mets_col].value_counts(normalize=True) * 100
    
    # 표로 출력
    dist_df = pd.DataFrame({
        '클래스': class_counts.index,
        '샘플 수': class_counts.values,
        '비율 (%)': class_props.values.round(2)
    })
    
    print("\n✅ 클래스 분포:")
    print(dist_df.to_string(index=False))
    
    # 클래스 불균형 비율 계산
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\n⚠️  클래스 불균형 비율: {imbalance_ratio:.1f}:1 "
          f"(최다 클래스 / 최소 클래스)")
    
    if imbalance_ratio > 10:
        print("   → 심각한 클래스 불균형! 모델이 다수 클래스에 편향될 수 있습니다.")
    
    return class_counts, mets_col

def load_model_and_predict(df):
    """모델 로드 및 예측"""
    print("\n" + "="*80)
    print("🤖 STEP 3: 모델 로드 및 예측")
    print("="*80)
    
    # 모듈 경로 추가
    sys.path.insert(0, './src')
    from mets_prediction_model import MetSPredictor
    
    # 최신 모델 찾기
    model_dirs = glob.glob('./results/models/mets_predictor_*')
    if not model_dirs:
        raise FileNotFoundError("학습된 모델을 찾을 수 없습니다!")
    
    latest_model_dir = max(model_dirs, key=os.path.getmtime)
    print(f"✅ 모델 디렉토리: {latest_model_dir}")
    
    # 예측
    predictor = MetSPredictor()
    predictor.load_model(latest_model_dir)
    
    y_pred, y_prob = predictor.predict(df)
    
    print(f"✅ 예측 완료: {len(y_pred):,}개 샘플")
    
    return y_pred, y_prob

def detailed_classification_metrics(y_true, y_pred):
    """상세 분류 성능 지표"""
    print("\n" + "="*80)
    print("📈 STEP 4: 클래스별 상세 성능 지표")
    print("="*80)
    
    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # DataFrame으로 변환
    classes = [c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
    
    metrics_df = pd.DataFrame({
        '클래스': classes,
        'Precision': [report[c]['precision'] for c in classes],
        'Recall': [report[c]['recall'] for c in classes],
        'F1-Score': [report[c]['f1-score'] for c in classes],
        '샘플 수': [int(report[c]['support']) for c in classes]
    })
    
    print("\n✅ 클래스별 성능:")
    print(metrics_df.to_string(index=False))
    
    # 전체 평균 지표
    print(f"\n📊 전체 평균 지표:")
    print(f"   - Overall Accuracy:     {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
    print(f"   - Macro Avg Precision:  {report['macro avg']['precision']:.4f}")
    print(f"   - Macro Avg Recall:     {report['macro avg']['recall']:.4f}")
    print(f"   - Macro Avg F1:         {report['macro avg']['f1-score']:.4f}")
    print(f"   - Weighted Avg F1:      {report['weighted avg']['f1-score']:.4f}")
    
    # 클래스별 분석
    print(f"\n🔍 클래스별 세부 분석:")
    for i, row in metrics_df.iterrows():
        cls = row['클래스']
        precision = row['Precision']
        recall = row['Recall']
        f1 = row['F1-Score']
        support = int(row['샘플 수'])
        
        print(f"\n   [{cls}] (샘플 수: {support:,}개)")
        print(f"   - Precision: {precision:.4f} → 예측한 것 중 {precision*100:.1f}%가 실제로 맞음")
        print(f"   - Recall:    {recall:.4f} → 실제 케이스 중 {recall*100:.1f}%를 찾아냄")
        print(f"   - F1-Score:  {f1:.4f} → Precision과 Recall의 조화평균")
        
        if precision < 0.5:
            print(f"   ⚠️  낮은 Precision! 이 클래스로 예측할 때 오류가 많음")
        if recall < 0.5:
            print(f"   ⚠️  낮은 Recall! 실제 케이스를 많이 놓침")
    
    return metrics_df, report

def confusion_matrix_analysis(y_true, y_pred, class_names):
    """Confusion Matrix 상세 분석"""
    print("\n" + "="*80)
    print("🔢 STEP 5: Confusion Matrix 상세 분석")
    print("="*80)
    
    # Confusion Matrix 계산
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # DataFrame으로 변환
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    print("\n✅ 혼동 행렬 (Actual vs Predicted):")
    print(cm_df.to_string())
    
    # 행별 정규화 (실제 클래스별로 예측 분포)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    
    print("\n✅ 정규화된 혼동 행렬 (행 합 = 1.0):")
    print(cm_norm_df.round(4).to_string())
    
    # 오분류 분석
    print("\n🔍 주요 오분류 패턴:")
    error_found = False
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                error_rate = cm[i, j] / cm[i].sum()
                if error_rate > 0.05:  # 5% 이상 오분류
                    print(f"   - 실제 '{true_class}' → 잘못 예측 '{pred_class}': "
                          f"{cm[i, j]:,}건 ({error_rate*100:.1f}%)")
                    error_found = True
    
    if not error_found:
        print("   ✅ 주요 오분류 패턴 없음 (5% 미만)")
    
    return cm, cm_df, cm_normalized

def plot_visualizations(cm, cm_normalized, class_names, metrics_df):
    """시각화 생성"""
    print("\n" + "="*80)
    print("📊 STEP 6: 시각화 생성")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 6-1. Confusion Matrix (절대 개수)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0, 0], cbar_kws={'label': '샘플 수'})
    axes[0, 0].set_title('Confusion Matrix (절대 개수)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('실제 (Actual)', fontsize=12)
    axes[0, 0].set_xlabel('예측 (Predicted)', fontsize=12)
    
    # 6-2. Confusion Matrix (정규화)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0, 1], cbar_kws={'label': '비율'})
    axes[0, 1].set_title('Confusion Matrix (정규화)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('실제 (Actual)', fontsize=12)
    axes[0, 1].set_xlabel('예측 (Predicted)', fontsize=12)
    
    # 6-3. 클래스별 성능 비교
    x = np.arange(len(metrics_df))
    width = 0.25
    
    ax = axes[1, 0]
    ax.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('클래스', fontsize=12)
    ax.set_ylabel('점수', fontsize=12)
    ax.set_title('클래스별 Precision, Recall, F1-Score', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['클래스'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # 6-4. F1-Score vs 샘플 수
    ax = axes[1, 1]
    scatter = ax.scatter(metrics_df['샘플 수'], metrics_df['F1-Score'],
                        s=300, alpha=0.6, c=metrics_df['F1-Score'],
                        cmap='RdYlGn', vmin=0, vmax=1, edgecolors='black', linewidth=1.5)
    
    for i, row in metrics_df.iterrows():
        ax.annotate(row['클래스'],
                   (row['샘플 수'], row['F1-Score']),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=10, ha='left', fontweight='bold')
    
    ax.set_xlabel('샘플 수', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score vs 샘플 수 (클래스 불균형 영향)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.1])
    plt.colorbar(scatter, ax=ax, label='F1-Score')
    
    plt.tight_layout()
    
    output_path = './results/detailed_mets_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 저장: {output_path}")
    
    plt.show()

def generate_markdown_report(metrics_df, cm_df, cm_normalized, class_counts, report):
    """마크다운 보고서 생성"""
    print("\n" + "="*80)
    print("📝 STEP 7: 마크다운 보고서 생성")
    print("="*80)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_path = f'./results/DETAILED_METS_ANALYSIS_{timestamp}.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Ver3: MetS 분류 성능 상세 분석 보고서\n\n")
        f.write(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # 1. Executive Summary
        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **전체 정확도 (Accuracy)**: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)\n")
        f.write(f"- **Macro Avg F1-Score**: {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"- **Weighted Avg F1-Score**: {report['weighted avg']['f1-score']:.4f}\n")
        f.write(f"- **분석 샘플 수**: {class_counts.sum():,}개\n")
        f.write(f"- **클래스 수**: {len(class_counts)}개\n\n")
        
        # 클래스 불균형
        imbalance_ratio = class_counts.max() / class_counts.min()
        f.write(f"### 클래스 불균형 정도\n\n")
        f.write(f"- **불균형 비율**: {imbalance_ratio:.1f}:1 (최다 클래스 / 최소 클래스)\n")
        if imbalance_ratio > 10:
            f.write(f"- ⚠️ **경고**: 심각한 클래스 불균형!\n\n")
        else:
            f.write(f"- ✅ 비교적 균형 잡힌 데이터셋입니다.\n\n")
        
        # 2. 클래스 분포
        f.write("## 2. 클래스 분포\n\n")
        f.write("| 클래스 | 샘플 수 | 비율 (%) |\n")
        f.write("|--------|---------|----------|\n")
        for cls, count in class_counts.items():
            prop = count / class_counts.sum() * 100
            f.write(f"| {cls} | {count:,} | {prop:.2f}% |\n")
        f.write("\n")
        
        # 3. 클래스별 성능
        f.write("## 3. 클래스별 상세 성능\n\n")
        f.write("| 클래스 | Precision | Recall | F1-Score | 샘플 수 |\n")
        f.write("|--------|-----------|--------|----------|----------|\n")
        for _, row in metrics_df.iterrows():
            f.write(f"| {row['클래스']} | {row['Precision']:.4f} | "
                   f"{row['Recall']:.4f} | {row['F1-Score']:.4f} | {int(row['샘플 수']):,} |\n")
        f.write("\n")
        
        # 4. Confusion Matrix
        f.write("## 4. Confusion Matrix (절대 개수)\n\n")
        f.write("```\n")
        f.write(cm_df.to_string())
        f.write("\n```\n\n")
        
        # 정규화
        f.write("## 5. Confusion Matrix (정규화, 행 합 = 1.0)\n\n")
        f.write("```\n")
        f.write(pd.DataFrame(cm_normalized, 
                             index=cm_df.index, 
                             columns=cm_df.columns).round(4).to_string())
        f.write("\n```\n\n")
        
        # 6. 주요 발견 사항
        f.write("## 6. 주요 발견 사항 (Key Findings)\n\n")
        
        # 최고/최저 성능 클래스
        best_f1_idx = metrics_df['F1-Score'].idxmax()
        worst_f1_idx = metrics_df['F1-Score'].idxmin()
        
        best_class = metrics_df.loc[best_f1_idx, '클래스']
        best_f1 = metrics_df.loc[best_f1_idx, 'F1-Score']
        worst_class = metrics_df.loc[worst_f1_idx, '클래스']
        worst_f1 = metrics_df.loc[worst_f1_idx, 'F1-Score']
        
        f.write(f"### 6.1 최고 성능 클래스\n\n")
        f.write(f"- **클래스**: {best_class}\n")
        f.write(f"- **F1-Score**: {best_f1:.4f}\n")
        f.write(f"- **샘플 수**: {metrics_df.loc[best_f1_idx, '샘플 수']:,.0f}개\n\n")
        
        f.write(f"### 6.2 최저 성능 클래스\n\n")
        f.write(f"- **클래스**: {worst_class}\n")
        f.write(f"- **F1-Score**: {worst_f1:.4f}\n")
        f.write(f"- **샘플 수**: {metrics_df.loc[worst_f1_idx, '샘플 수']:,.0f}개\n\n")
        
        # 7. 결론 및 제언
        f.write("## 7. 결론 및 제언\n\n")
        
        f.write("### 7.1 모델 강점\n\n")
        if report['accuracy'] > 0.9:
            f.write(f"- ✅ 매우 높은 전체 정확도 ({report['accuracy']*100:.1f}%)\n")
        
        high_perf_classes = metrics_df[metrics_df['F1-Score'] > 0.8]
        if len(high_perf_classes) > 0:
            f.write(f"- ✅ {len(high_perf_classes)}개 클래스에서 F1-Score > 0.8 달성\n")
        
        f.write("\n### 7.2 개선 제안\n\n")
        
        low_perf_classes = metrics_df[metrics_df['F1-Score'] < 0.5]
        if len(low_perf_classes) > 0:
            f.write(f"- ⚠️ {len(low_perf_classes)}개 클래스에서 F1-Score < 0.5 (개선 필요)\n")
        
        if imbalance_ratio > 10:
            f.write("- 클래스 불균형 해결 방안:\n")
            f.write("  - SMOTE 등 오버샘플링 기법 적용\n")
            f.write("  - Class weights 조정\n")
            f.write("  - Cost-sensitive learning\n")
        
        f.write("\n---\n\n")
        f.write("*Report generated by Ver3 MetS Analysis Pipeline*\n")
    
    print(f"✅ 저장: {report_path}")
    return report_path

def main():
    """메인 분석 파이프라인"""
    print("\n" + "="*80)
    print("🔬 Ver3: MetS 분류 성능 상세 분석")
    print("="*80)
    
    try:
        # 1. 데이터 로드
        df = load_latest_results()
        
        # 2. 클래스 분포 분석
        result = analyze_class_distribution(df)
        if result is None:
            return
        class_counts, mets_col = result
        
        # 3. 모델 로드 및 예측
        y_pred, y_prob = load_model_and_predict(df)
        y_true = df[mets_col].values
        
        # 4. 상세 분류 지표
        class_names = sorted(df[mets_col].unique())
        metrics_df, report = detailed_classification_metrics(y_true, y_pred)
        
        # 5. Confusion Matrix 분석
        cm, cm_df, cm_normalized = confusion_matrix_analysis(y_true, y_pred, class_names)
        
        # 6. 시각화
        plot_visualizations(cm, cm_normalized, class_names, metrics_df)
        
        # 7. 보고서 생성
        report_path = generate_markdown_report(metrics_df, cm_df, cm_normalized, 
                                               class_counts, report)
        
        print("\n" + "="*80)
        print("✅ 상세 분석 완료!")
        print("="*80)
        print(f"\n📁 결과 파일:")
        print(f"   - 시각화: ./results/detailed_mets_analysis.png")
        print(f"   - 보고서: {report_path}")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
