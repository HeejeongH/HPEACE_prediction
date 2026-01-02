"""
Ver3 Enhanced: 전체 파이프라인 실행 스크립트
==============================================

개선 사항:
1. 모든 변수 활용 (HDL, GLUCOSE, HbA1c, 질병력, 투약, 생활습관)
2. SMOTE 오버샘플링으로 클래스 불균형 해결
3. Class Weights + 개선된 모델 아키텍처
4. 앙상블 강화

저자: SNUH Prediction Team
날짜: 2026-01-03
"""

import sys
import os
from datetime import datetime

# 모듈 경로 추가
sys.path.insert(0, './src')

from enhanced_data_preprocessing import EnhancedPairedVisitPreprocessor
from enhanced_mets_model import EnhancedMetSPredictor


def main():
    """메인 파이프라인"""
    
    print("\n" + "="*80)
    print("🚀 Ver3 Enhanced: 대대적 개선 파이프라인")
    print("="*80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = datetime.now()
    
    # ==========================================
    # STEP 1: 데이터 전처리
    # ==========================================
    print("\n" + "="*80)
    print("STEP 1: 데이터 전처리 (Enhanced)")
    print("="*80)
    
    preprocessor = EnhancedPairedVisitPreprocessor(min_time_gap=90, max_time_gap=365)
    
    df, info = preprocessor.preprocess('../data/total_again.xlsx')
    
    # 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'./results/enhanced_paired_data_{timestamp}.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 전처리 데이터 저장: {output_path}")
    
    # ==========================================
    # STEP 2: MetS 예측 모델 학습
    # ==========================================
    print("\n" + "="*80)
    print("STEP 2: MetS 예측 모델 학습 (Enhanced)")
    print("="*80)
    
    predictor = EnhancedMetSPredictor(random_state=42, use_smote=True)
    
    results = predictor.train(df, use_ensemble=True)
    
    # 모델 저장
    model_dir = f'./results/models/mets_predictor_enhanced_{timestamp}'
    predictor.save_model(model_dir)
    
    # ==========================================
    # STEP 3: 결과 보고서 생성
    # ==========================================
    print("\n" + "="*80)
    print("STEP 3: 결과 보고서 생성")
    print("="*80)
    
    report_path = f'./results/ENHANCED_REPORT_{timestamp}.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Ver3 Enhanced: MetS 예측 모델 결과 보고서\n\n")
        f.write(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # 데이터 요약
        f.write("## 1. 데이터 요약\n\n")
        f.write(f"- **총 샘플 수**: {info['n_samples_final']:,}개\n")
        f.write(f"- **특성 수**: {info['n_features_final']:,}개\n")
        f.write(f"- **평균 방문 간격**: {df['time_gap_days'].mean():.1f}일\n")
        f.write(f"- **Baseline MetS 유병률**: {df['mets_diagnosis_baseline'].mean()*100:.1f}%\n")
        f.write(f"- **Follow-up MetS 유병률**: {df['mets_diagnosis_followup'].mean()*100:.1f}%\n\n")
        
        # 개선 사항
        f.write("## 2. 주요 개선 사항\n\n")
        f.write("### 2.1 데이터 확장\n")
        f.write("- ✅ HDL, GLUCOSE, HbA1c 등 건강지표 추가\n")
        f.write("- ✅ 질병력 (고혈압, 당뇨, 고지혈증 등) 추가\n")
        f.write("- ✅ 투약 정보 추가\n")
        f.write("- ✅ 생활습관 (흡연, 음주, 활동량) 추가\n")
        f.write("- ✅ 복합 위험 점수 생성\n\n")
        
        f.write("### 2.2 클래스 불균형 해결\n")
        f.write("- ✅ SMOTE 오버샘플링 적용\n")
        f.write("- ✅ Class Weights 조정\n")
        f.write("- ✅ 개선된 모델 아키텍처\n\n")
        
        # 성능
        f.write("## 3. MetS 예측 성능\n\n")
        f.write(f"- **Accuracy**: {results['accuracy']:.4f}\n")
        f.write(f"- **Macro F1**: {results['f1_macro']:.4f}\n")
        f.write(f"- **Weighted F1**: {results['f1_weighted']:.4f}\n\n")
        
        # Classification Report
        f.write("### Classification Report\n\n")
        f.write("```\n")
        
        report_dict = results['classification_report']
        classes = [c for c in report_dict.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
        
        f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 70 + "\n")
        
        for cls in classes:
            prec = report_dict[cls]['precision']
            rec = report_dict[cls]['recall']
            f1 = report_dict[cls]['f1-score']
            sup = int(report_dict[cls]['support'])
            f.write(f"{cls:<20} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {sup:<10}\n")
        
        f.write("\n")
        f.write(f"{'accuracy':<20} {'':<12} {'':<12} {report_dict['accuracy']:<12.4f} {sum([report_dict[c]['support'] for c in classes]):<10.0f}\n")
        f.write(f"{'macro avg':<20} {report_dict['macro avg']['precision']:<12.4f} {report_dict['macro avg']['recall']:<12.4f} {report_dict['macro avg']['f1-score']:<12.4f} {sum([report_dict[c]['support'] for c in classes]):<10.0f}\n")
        f.write(f"{'weighted avg':<20} {report_dict['weighted avg']['precision']:<12.4f} {report_dict['weighted avg']['recall']:<12.4f} {report_dict['weighted avg']['f1-score']:<12.4f} {sum([report_dict[c]['support'] for c in classes]):<10.0f}\n")
        
        f.write("```\n\n")
        
        # Confusion Matrix
        f.write("### Confusion Matrix\n\n")
        f.write("```\n")
        import pandas as pd
        cm_df = pd.DataFrame(
            results['confusion_matrix'],
            index=['stable_no_mets', 'new_onset', 'remission', 'persistent'],
            columns=['stable_no_mets', 'new_onset', 'remission', 'persistent']
        )
        f.write(cm_df.to_string())
        f.write("\n```\n\n")
        
        # Feature Importance
        f.write("## 4. 주요 영향 특성 (Top 20)\n\n")
        f.write("| Rank | Feature | Importance |\n")
        f.write("|------|---------|------------|\n")
        
        top_features = results['feature_importance'].head(20)
        for idx, row in top_features.iterrows():
            rank = idx + 1
            feature = row['feature']
            importance = row['importance']
            f.write(f"| {rank} | {feature} | {importance:.4f} |\n")
        
        f.write("\n---\n\n")
        
        # 결론
        f.write("## 5. 결론 및 향후 계획\n\n")
        f.write("### 주요 성과\n\n")
        
        # 개선 정도 계산
        baseline_f1 = 0.4639  # Ver3 원본
        improved_f1 = results['f1_macro']
        improvement = ((improved_f1 - baseline_f1) / baseline_f1) * 100
        
        f.write(f"- **Macro F1 개선**: {baseline_f1:.4f} → {improved_f1:.4f} ({improvement:+.1f}%)\n")
        f.write(f"- **클래스 불균형 해결**: SMOTE + Class Weights 적용\n")
        f.write(f"- **특성 확장**: {info['n_features_final']:,}개 특성 활용\n")
        f.write(f"- **앙상블 강화**: TabNet + XGBoost + LightGBM + CatBoost\n\n")
        
        f.write("### 클래스별 개선 사항\n\n")
        
        for cls in ['new_onset', 'persistent', 'remission']:
            if cls in report_dict:
                f1_score = report_dict[cls]['f1-score']
                if f1_score > 0.3:
                    f.write(f"- **{cls}**: F1 = {f1_score:.4f} ✅\n")
                else:
                    f.write(f"- **{cls}**: F1 = {f1_score:.4f} ⚠️ (추가 개선 필요)\n")
        
        f.write("\n")
        
    print(f"💾 보고서 저장: {report_path}")
    
    # ==========================================
    # 완료
    # ==========================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "="*80)
    print("✅ Ver3 Enhanced 파이프라인 완료!")
    print("="*80)
    print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {duration:.1f}분")
    print(f"\n📁 결과 디렉토리: ./results")
    print(f"   - 전처리 데이터: {output_path}")
    print(f"   - 모델: {model_dir}")
    print(f"   - 보고서: {report_path}")
    print("\n")


if __name__ == "__main__":
    main()
