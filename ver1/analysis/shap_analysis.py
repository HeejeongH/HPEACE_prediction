"""
SHAP Analysis - 모델 해석력 분석
================================

SHAP (SHapley Additive exPlanations)를 사용한:
1. Feature Importance 분석
2. Feature Interaction 분석
3. Individual Prediction 설명
4. Dependence Plots

Author: Research Team
Date: 2025-11-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not installed. Install with: pip install shap")
    SHAP_AVAILABLE = False

OUTPUT_DIR = Path('./advanced_results/shap_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SHAPAnalyzer:
    """SHAP 기반 모델 해석"""
    
    def __init__(self, model_dir='./result/models'):
        """
        Args:
            model_dir: Ver1 학습된 모델이 저장된 디렉토리
        """
        print("\n" + "="*80)
        print("🔍 SHAP Analysis 초기화")
        print("="*80)
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not found")
        
        self.model_dir = Path(model_dir)
        self.health_indicators = [
            '체중', '체질량지수', '허리둘레(WAIST)', 'SBP', 'DBP', 'TG'
        ]
        
        print(f"\n모델 디렉토리: {self.model_dir}")
        print(f"분석 대상: {len(self.health_indicators)}개 건강지표")
    
    def load_model_and_data(self, indicator, sample_size=1000):
        """
        모델과 데이터 로드
        
        Args:
            indicator: 건강지표 이름
            sample_size: SHAP 계산용 샘플 수 (SHAP은 느리므로 샘플링)
        
        Returns:
            model, X_sample, feature_names
        """
        print(f"\n[{indicator}] 모델 및 데이터 로드...")
        
        # 모델 로드 (실제 Ver1 모델 구조에 맞게 수정 필요)
        # 여기서는 예시 코드
        model_path = self.model_dir / f"{indicator}_model.pkl"
        
        if not model_path.exists():
            print(f"   ⚠️  모델 파일 없음: {model_path}")
            return None, None, None
        
        model = joblib.load(model_path)
        
        # 데이터 로드 (실제 Ver1 데이터 구조에 맞게 수정 필요)
        data_path = Path('../data/total_again.xlsx')
        df = pd.read_excel(data_path)
        
        # 특성 준비 (Ver1과 동일하게)
        # 실제 Ver1 코드의 feature engineering 로직 사용
        
        # 샘플링
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df
        
        print(f"   ✅ 모델 로드 완료")
        print(f"   ✅ 샘플 데이터: {len(df_sample)}개")
        
        return model, df_sample, None
    
    def compute_shap_values(self, model, X, feature_names):
        """
        SHAP 값 계산
        
        Args:
            model: 학습된 모델
            X: 입력 데이터
            feature_names: 특성 이름 목록
        
        Returns:
            shap_values, explainer
        """
        print(f"\n   SHAP 값 계산 중...")
        
        # TreeExplainer (RandomForest, XGBoost 등에 적합)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        print(f"   ✅ SHAP 계산 완료")
        
        return shap_values, explainer
    
    def plot_shap_summary(self, shap_values, X, feature_names, indicator):
        """
        SHAP Summary Plot (Feature Importance + Direction)
        
        Args:
            shap_values: SHAP 값
            X: 입력 데이터
            feature_names: 특성 이름
            indicator: 건강지표 이름
        """
        plt.figure(figsize=(12, 8))
        
        shap.summary_plot(
            shap_values, X,
            feature_names=feature_names,
            show=False,
            max_display=20
        )
        
        plt.title(f'{indicator} SHAP Summary Plot', fontsize=16, pad=20)
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / f'{indicator}_shap_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ SHAP Summary Plot 저장: {output_path}")
        plt.close()
    
    def plot_shap_dependence(self, shap_values, X, feature_names, indicator, top_n=5):
        """
        SHAP Dependence Plot (특성 값에 따른 SHAP 값 변화)
        
        Args:
            shap_values: SHAP 값
            X: 입력 데이터
            feature_names: 특성 이름
            indicator: 건강지표 이름
            top_n: 상위 N개 특성만 플롯
        """
        # Feature importance 계산
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(feature_importance)[-top_n:][::-1]
        
        fig, axes = plt.subplots(
            (top_n + 1) // 2, 2,
            figsize=(16, 4 * ((top_n + 1) // 2))
        )
        axes = axes.flatten() if top_n > 1 else [axes]
        
        for idx, feat_idx in enumerate(top_features_idx):
            if idx >= len(axes):
                break
                
            feat_name = feature_names[feat_idx]
            
            shap.dependence_plot(
                feat_idx,
                shap_values, X,
                feature_names=feature_names,
                ax=axes[idx],
                show=False
            )
            
            axes[idx].set_title(f'{feat_name}', fontsize=12)
        
        # 빈 subplot 숨기기
        for idx in range(len(top_features_idx), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(
            f'{indicator} SHAP Dependence Plots (Top {top_n} Features)',
            fontsize=16, y=1.005
        )
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / f'{indicator}_shap_dependence.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ SHAP Dependence Plot 저장: {output_path}")
        plt.close()
    
    def compute_feature_interactions(self, shap_values, feature_names, indicator):
        """
        Feature Interaction 분석
        
        Args:
            shap_values: SHAP 값
            feature_names: 특성 이름
            indicator: 건강지표 이름
        
        Returns:
            interaction_matrix: 상호작용 행렬
        """
        print(f"\n   Feature Interaction 분석 중...")
        
        # SHAP interaction values는 계산 비용이 매우 높음
        # 여기서는 상위 10개 특성만 분석
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_10_idx = np.argsort(feature_importance)[-10:][::-1]
        
        # Pairwise correlation of SHAP values (간단한 근사치)
        shap_top_10 = shap_values[:, top_10_idx]
        interaction_matrix = np.corrcoef(shap_top_10.T)
        
        # 시각화
        plt.figure(figsize=(10, 8))
        
        top_10_names = [feature_names[i] for i in top_10_idx]
        sns.heatmap(
            interaction_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            xticklabels=top_10_names,
            yticklabels=top_10_names,
            cbar_kws={'label': 'SHAP Correlation'}
        )
        
        plt.title(
            f'{indicator} Feature Interactions (Top 10 Features)',
            fontsize=14, pad=20
        )
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / f'{indicator}_feature_interactions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Feature Interaction 저장: {output_path}")
        plt.close()
        
        return interaction_matrix
    
    def analyze_all_indicators(self):
        """모든 건강지표에 대해 SHAP 분석"""
        print("\n" + "="*80)
        print("🚀 SHAP 분석 시작")
        print("="*80)
        
        results = {}
        
        for indicator in self.health_indicators:
            print(f"\n{'='*80}")
            print(f"[{indicator}] 분석 시작")
            print(f"{'='*80}")
            
            # 모델 및 데이터 로드
            model, X_sample, feature_names = self.load_model_and_data(
                indicator, sample_size=1000
            )
            
            if model is None:
                print(f"   ⚠️  {indicator} 건너뜀 (모델 없음)")
                continue
            
            # SHAP 값 계산
            shap_values, explainer = self.compute_shap_values(
                model, X_sample, feature_names
            )
            
            # 시각화
            self.plot_shap_summary(shap_values, X_sample, feature_names, indicator)
            self.plot_shap_dependence(shap_values, X_sample, feature_names, indicator)
            interaction_matrix = self.compute_feature_interactions(
                shap_values, feature_names, indicator
            )
            
            results[indicator] = {
                'shap_values': shap_values,
                'explainer': explainer,
                'interaction_matrix': interaction_matrix
            }
        
        self.results = results
        
        return results
    
    def generate_summary_report(self):
        """SHAP 분석 요약 보고서"""
        print("\n" + "="*80)
        print("📄 SHAP 분석 요약 보고서 생성")
        print("="*80)
        
        report = []
        report.append("# SHAP Analysis Summary Report")
        report.append("")
        report.append("## Overview")
        report.append("")
        report.append(f"- 분석 대상: {len(self.health_indicators)}개 건강지표")
        report.append(f"- 샘플 크기: 각 1,000개")
        report.append("")
        
        for indicator in self.health_indicators:
            if indicator not in self.results:
                continue
                
            report.append(f"## {indicator}")
            report.append("")
            report.append("### SHAP Summary")
            report.append(f"- Summary Plot: `{indicator}_shap_summary.png`")
            report.append(f"- Dependence Plot: `{indicator}_shap_dependence.png`")
            report.append(f"- Interaction Matrix: `{indicator}_feature_interactions.png`")
            report.append("")
        
        report_text = "\n".join(report)
        
        output_path = OUTPUT_DIR / 'shap_analysis_report.md'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✅ 요약 보고서 저장: {output_path}")
        
        return report_text


def main():
    """메인 실행 함수"""
    print("\n" + "="*80)
    print("🚀 SHAP Analysis 시작")
    print("="*80)
    
    if not SHAP_AVAILABLE:
        print("\n❌ SHAP library가 설치되지 않았습니다.")
        print("설치 명령: pip install shap")
        return
    
    print("\n⚠️  주의: SHAP 분석은 시간이 오래 걸립니다 (지표당 5-10분)")
    print("⚠️  실제 Ver1 모델 파일 경로를 확인하고 코드를 수정하세요.")
    print("")
    
    # Initialize
    analyzer = SHAPAnalyzer()
    
    # Analyze all
    results = analyzer.analyze_all_indicators()
    
    # Generate report
    analyzer.generate_summary_report()
    
    print("\n" + "="*80)
    print("✅ Phase 3 완료: SHAP 해석력 분석")
    print("="*80)
    print(f"\n결과 저장 위치: {OUTPUT_DIR.absolute()}")


if __name__ == '__main__':
    main()
