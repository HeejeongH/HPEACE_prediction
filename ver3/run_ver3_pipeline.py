"""
Ver3: 전체 파이프라인 실행 스크립트
===================================

실행 단계:
1. 데이터 전처리 (Paired visits 생성)
2. 건강지표 변화 예측 모델 학습
3. MetS 발생/개선 예측 모델 학습
4. 결과 저장 및 시각화

저자: SNUH Prediction Team
"""

import sys
import os
from datetime import datetime
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_preprocessing import PairedVisitPreprocessor
from health_prediction_model import HealthIndicatorPredictor
from mets_prediction_model import MetSPredictor


class Ver3Pipeline:
    """Ver3 전체 파이프라인 클래스"""
    
    def __init__(self, data_path: str, output_dir: str = './results'):
        """
        Parameters
        ----------
        data_path : str
            원본 데이터 경로
        output_dir : str
            결과 저장 디렉토리
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 결과 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
        
    def run_preprocessing(self, 
                         min_time_gap: int = 90,
                         max_time_gap: int = 365) -> pd.DataFrame:
        """
        Step 1: 데이터 전처리
        
        Parameters
        ----------
        min_time_gap : int
            최소 방문 간격 (일)
        max_time_gap : int
            최대 방문 간격 (일)
            
        Returns
        -------
        processed_df : DataFrame
            전처리된 데이터
        """
        print("\n" + "="*100)
        print("STEP 1: 데이터 전처리")
        print("="*100)
        
        preprocessor = PairedVisitPreprocessor(
            min_time_gap=min_time_gap,
            max_time_gap=max_time_gap
        )
        
        processed_df, info = preprocessor.preprocess(self.data_path)
        
        # 전처리된 데이터 저장
        save_path = os.path.join(self.output_dir, f'paired_data_{self.timestamp}.csv')
        processed_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 전처리 데이터 저장: {save_path}")
        
        # 전처리 정보 저장
        info_df = pd.DataFrame([info])
        info_path = os.path.join(self.output_dir, 'reports', f'preprocessing_info_{self.timestamp}.csv')
        info_df.to_csv(info_path, index=False, encoding='utf-8-sig')
        
        return processed_df
    
    def run_health_prediction(self, df: pd.DataFrame, use_ensemble: bool = True) -> Dict:
        """
        Step 2: 건강지표 변화 예측
        
        Parameters
        ----------
        df : DataFrame
            전처리된 데이터
        use_ensemble : bool
            앙상블 사용 여부
            
        Returns
        -------
        results : Dict
            학습 결과
        """
        print("\n" + "="*100)
        print("STEP 2: 건강지표 변화 예측 모델 학습")
        print("="*100)
        
        predictor = HealthIndicatorPredictor(random_state=42)
        results = predictor.train_all_targets(df, use_ensemble=use_ensemble)
        
        # 모델 저장
        model_dir = os.path.join(self.output_dir, 'models', f'health_predictor_{self.timestamp}')
        predictor.save_models(model_dir)
        
        # 성능 요약 저장
        summary = []
        for target, result in results.items():
            summary.append({
                'Target': target,
                'Train_R2': result['performance']['train_r2'],
                'Test_R2': result['performance']['test_r2'],
                'RMSE': result['performance']['rmse'],
                'MAE': result['performance']['mae']
            })
        
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(self.output_dir, 'reports', 
                                    f'health_prediction_summary_{self.timestamp}.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 성능 요약 저장: {summary_path}")
        
        return results
    
    def run_mets_prediction(self, df: pd.DataFrame, use_ensemble: bool = True) -> Dict:
        """
        Step 3: MetS 발생/개선 예측
        
        Parameters
        ----------
        df : DataFrame
            전처리된 데이터
        use_ensemble : bool
            앙상블 사용 여부
            
        Returns
        -------
        result : Dict
            학습 결과
        """
        print("\n" + "="*100)
        print("STEP 3: MetS 발생/개선 예측 모델 학습")
        print("="*100)
        
        predictor = MetSPredictor(random_state=42)
        result = predictor.train(df, use_ensemble=use_ensemble)
        
        # 모델 저장
        model_dir = os.path.join(self.output_dir, 'models', f'mets_predictor_{self.timestamp}')
        predictor.save_model(model_dir)
        
        # 성능 요약 저장
        summary = {
            'Accuracy': result['performance']['accuracy'],
            'F1_Score': result['performance']['f1_score']
        }
        
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(self.output_dir, 'reports', 
                                    f'mets_prediction_summary_{self.timestamp}.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 성능 요약 저장: {summary_path}")
        
        return result
    
    def create_visualizations(self, 
                            health_results: Dict,
                            mets_result: Dict):
        """
        Step 4: 결과 시각화
        
        Parameters
        ----------
        health_results : Dict
            건강지표 예측 결과
        mets_result : Dict
            MetS 예측 결과
        """
        print("\n" + "="*100)
        print("STEP 4: 결과 시각화")
        print("="*100)
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 건강지표 예측 성능 비교
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Ver3: Health Indicator Prediction Performance', 
                    fontsize=16, fontweight='bold')
        
        targets = list(health_results.keys())
        test_r2 = [health_results[t]['performance']['test_r2'] for t in targets]
        rmse = [health_results[t]['performance']['rmse'] for t in targets]
        mae = [health_results[t]['performance']['mae'] for t in targets]
        
        # R² scores
        axes[0, 0].barh(range(len(targets)), test_r2, color='steelblue')
        axes[0, 0].set_yticks(range(len(targets)))
        axes[0, 0].set_yticklabels([t.replace('_change', '') for t in targets])
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_title('Test R² Scores')
        axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # RMSE
        axes[0, 1].barh(range(len(targets)), rmse, color='coral')
        axes[0, 1].set_yticks(range(len(targets)))
        axes[0, 1].set_yticklabels([t.replace('_change', '') for t in targets])
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_title('Root Mean Squared Error')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # MAE
        axes[1, 0].barh(range(len(targets)), mae, color='lightgreen')
        axes[1, 0].set_yticks(range(len(targets)))
        axes[1, 0].set_yticklabels([t.replace('_change', '') for t in targets])
        axes[1, 0].set_xlabel('MAE')
        axes[1, 0].set_title('Mean Absolute Error')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # 예측 vs 실제 (첫 번째 타겟)
        first_target = targets[0]
        pred_data = health_results[first_target]['predictions']
        axes[1, 1].scatter(pred_data['y_test'], pred_data['pred_test'], 
                          alpha=0.5, s=10)
        axes[1, 1].plot([pred_data['y_test'].min(), pred_data['y_test'].max()],
                       [pred_data['y_test'].min(), pred_data['y_test'].max()],
                       'r--', lw=2)
        axes[1, 1].set_xlabel('Actual')
        axes[1, 1].set_ylabel('Predicted')
        axes[1, 1].set_title(f'Prediction vs Actual: {first_target.replace("_change", "")}')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, 'figures', 
                               f'health_prediction_performance_{self.timestamp}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ 건강지표 예측 성능 그래프 저장: {fig_path}")
        plt.close()
        
        # 2. MetS 예측 Confusion Matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = mets_result['performance']['confusion_matrix']
        class_names = mets_result['label_encoder'].classes_
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Ver3: MetS Transition Prediction - Confusion Matrix', 
                    fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, 'figures', 
                               f'mets_confusion_matrix_{self.timestamp}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ MetS Confusion Matrix 저장: {fig_path}")
        plt.close()
        
        # 3. Feature Importance (건강지표 첫 번째 타겟)
        fig, ax = plt.subplots(figsize=(10, 8))
        first_target = list(health_results.keys())[0]
        importance_df = health_results[first_target]['feature_importance'].head(20)
        
        ax.barh(range(len(importance_df)), importance_df['importance'], color='purple', alpha=0.7)
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'], fontsize=9)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top 20 Feature Importance: {first_target.replace("_change", "")}', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, 'figures', 
                               f'feature_importance_{self.timestamp}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Feature Importance 그래프 저장: {fig_path}")
        plt.close()
        
        print("\n✅ 모든 시각화 완료!")
    
    def generate_report(self, 
                       processed_df: pd.DataFrame,
                       health_results: Dict,
                       mets_result: Dict):
        """
        Step 5: 최종 보고서 생성
        
        Parameters
        ----------
        processed_df : DataFrame
            전처리된 데이터
        health_results : Dict
            건강지표 예측 결과
        mets_result : Dict
            MetS 예측 결과
        """
        print("\n" + "="*100)
        print("STEP 5: 최종 보고서 생성")
        print("="*100)
        
        report_path = os.path.join(self.output_dir, 'reports', 
                                  f'FINAL_REPORT_{self.timestamp}.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Ver3: 식습관 변화 기반 건강지표 및 MetS 예측 모델\n\n")
            f.write(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # 1. 데이터 요약
            f.write("## 1. 데이터 요약\n\n")
            f.write(f"- **총 샘플 수**: {len(processed_df):,}개\n")
            f.write(f"- **특성 수**: {len(processed_df.columns)}개\n")
            f.write(f"- **평균 방문 간격**: {processed_df['time_gap_days'].mean():.1f}일\n")
            f.write(f"- **Baseline MetS 유병률**: {processed_df['mets_diagnosis_baseline'].mean()*100:.1f}%\n")
            f.write(f"- **Follow-up MetS 유병률**: {processed_df['mets_diagnosis_followup'].mean()*100:.1f}%\n\n")
            
            # 2. 건강지표 예측 성능
            f.write("## 2. 건강지표 변화 예측 성능\n\n")
            f.write("| Target | Train R² | Test R² | RMSE | MAE |\n")
            f.write("|--------|----------|---------|------|-----|\n")
            for target, result in health_results.items():
                perf = result['performance']
                f.write(f"| {target.replace('_change', '')} | "
                       f"{perf['train_r2']:.4f} | "
                       f"{perf['test_r2']:.4f} | "
                       f"{perf['rmse']:.4f} | "
                       f"{perf['mae']:.4f} |\n")
            
            avg_r2 = np.mean([r['performance']['test_r2'] for r in health_results.values()])
            f.write(f"\n**평균 Test R²**: {avg_r2:.4f}\n\n")
            
            # 3. MetS 예측 성능
            f.write("## 3. MetS 발생/개선 예측 성능\n\n")
            f.write(f"- **Accuracy**: {mets_result['performance']['accuracy']:.4f}\n")
            f.write(f"- **F1 Score**: {mets_result['performance']['f1_score']:.4f}\n\n")
            
            f.write("### Classification Report\n\n")
            f.write("```\n")
            f.write(mets_result['performance']['classification_report'])
            f.write("```\n\n")
            
            # 4. 주요 특성
            f.write("## 4. 주요 영향 특성 (Top 10)\n\n")
            first_target = list(health_results.keys())[0]
            importance_df = health_results[first_target]['feature_importance'].head(10)
            
            f.write("| Rank | Feature | Importance |\n")
            f.write("|------|---------|------------|\n")
            for idx, (i, row) in enumerate(importance_df.iterrows(), 1):
                f.write(f"| {idx} | {row['feature']} | {row['importance']:.4f} |\n")
            
            f.write("\n---\n\n")
            f.write("## 5. 결론\n\n")
            f.write("Ver3 모델은 두 번 연속 방문자 데이터를 활용하여 식습관 변화가 건강지표 변화와 ")
            f.write("MetS 발생/개선에 미치는 영향을 성공적으로 예측하였습니다.\n\n")
            
            f.write("### 주요 성과\n\n")
            f.write(f"1. **건강지표 예측**: 평균 R² {avg_r2:.4f} 달성\n")
            f.write(f"2. **MetS 예측**: Accuracy {mets_result['performance']['accuracy']:.4f} 달성\n")
            f.write("3. **해석 가능성**: TabNet 기반 feature importance 제공\n")
            f.write("4. **임상 적용성**: 식습관 개선을 통한 건강 관리 가이드라인 제시 가능\n\n")
        
        print(f"✅ 최종 보고서 저장: {report_path}")
    
    def run(self, 
           min_time_gap: int = 90,
           max_time_gap: int = 365,
           use_ensemble: bool = True):
        """
        전체 파이프라인 실행
        
        Parameters
        ----------
        min_time_gap : int
            최소 방문 간격 (일)
        max_time_gap : int
            최대 방문 간격 (일)
        use_ensemble : bool
            앙상블 사용 여부
        """
        start_time = datetime.now()
        
        print("\n" + "="*100)
        print("Ver3: 식습관 변화 기반 건강지표 및 MetS 예측 파이프라인")
        print("="*100)
        print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: 전처리
            processed_df = self.run_preprocessing(min_time_gap, max_time_gap)
            
            # Step 2: 건강지표 예측
            health_results = self.run_health_prediction(processed_df, use_ensemble)
            
            # Step 3: MetS 예측
            mets_result = self.run_mets_prediction(processed_df, use_ensemble)
            
            # Step 4: 시각화
            self.create_visualizations(health_results, mets_result)
            
            # Step 5: 보고서 생성
            self.generate_report(processed_df, health_results, mets_result)
            
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            
            print("\n" + "="*100)
            print("✅ 전체 파이프라인 완료!")
            print("="*100)
            print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"소요 시간: {elapsed_time/60:.1f}분")
            print(f"\n결과 디렉토리: {self.output_dir}")
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ver3 Pipeline')
    parser.add_argument('--data', type=str, default='./data/total_again.xlsx',
                       help='원본 데이터 경로')
    parser.add_argument('--output', type=str, default='./results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--min-gap', type=int, default=90,
                       help='최소 방문 간격 (일)')
    parser.add_argument('--max-gap', type=int, default=365,
                       help='최대 방문 간격 (일)')
    parser.add_argument('--no-ensemble', action='store_true',
                       help='TabNet만 사용 (앙상블 비활성화)')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    pipeline = Ver3Pipeline(
        data_path=args.data,
        output_dir=args.output
    )
    
    pipeline.run(
        min_time_gap=args.min_gap,
        max_time_gap=args.max_gap,
        use_ensemble=not args.no_ensemble
    )


if __name__ == "__main__":
    main()
