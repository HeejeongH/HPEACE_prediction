"""
Threshold Analysis - 임계값 분석 및 건강 위험도 평가
=================================================

식습관 점수에 따른 건강 위험 임계값 도출:
1. ROC Curve 기반 최적 임계값
2. Percentile 기반 위험군 분류
3. 임상적 기준 (비만, 고혈압 등)
4. 맞춤형 권고안 생성

Author: Research Team
Date: 2025-11-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('./advanced_results/threshold_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ThresholdAnalyzer:
    """임계값 분석 및 위험도 평가"""
    
    def __init__(self, data_path='../data/total_again.xlsx'):
        """
        Args:
            data_path: Ver1 원본 데이터 경로
        """
        print("\n" + "="*80)
        print("⚠️  Threshold Analysis 초기화")
        print("="*80)
        
        self.df = pd.read_excel(data_path)
        print(f"\n✅ 데이터 로드: {len(self.df):,}개 샘플")
        
        # 건강 위험 기준 (임상적 기준)
        self.clinical_thresholds = {
            '체중': None,  # 개인차가 커서 BMI 사용
            '체질량지수': {
                '저체중': (0, 18.5),
                '정상': (18.5, 23),
                '과체중': (23, 25),
                '비만1단계': (25, 30),
                '비만2단계': (30, 100)
            },
            'SBP': {
                '정상': (0, 120),
                '주의': (120, 130),
                '고혈압전단계': (130, 140),
                '고혈압1기': (140, 160),
                '고혈압2기': (160, 300)
            },
            'DBP': {
                '정상': (0, 80),
                '주의': (80, 85),
                '고혈압전단계': (85, 90),
                '고혈압1기': (90, 100),
                '고혈압2기': (100, 200)
            },
            'TG': {
                '정상': (0, 150),
                '경계': (150, 200),
                '높음': (200, 500),
                '매우높음': (500, 2000)
            }
        }
        
        # 식습관 변수
        self.diet_vars = [
            '간식빈도', '고지방 육류', '단맛', '단백질류', '곡류',
            '과일', '유제품', '음료류', '인스턴트 가공식품',
            '짠 간', '짠 식습관', '채소', '튀김'
        ]
        
    def classify_health_risk(self, indicator):
        """
        임상적 기준에 따른 건강 위험도 분류
        
        Args:
            indicator: 건강지표 이름
        
        Returns:
            df with risk_category column
        """
        print(f"\n[{indicator}] 건강 위험도 분류...")
        
        df_copy = self.df.copy()
        
        if indicator not in self.clinical_thresholds:
            print(f"   ⚠️  {indicator} 임상 기준 없음")
            return None
        
        thresholds = self.clinical_thresholds[indicator]
        
        if thresholds is None:
            return None
        
        # 위험도 분류
        df_copy['risk_category'] = '정상'
        
        for category, (low, high) in thresholds.items():
            mask = (df_copy[indicator] >= low) & (df_copy[indicator] < high)
            df_copy.loc[mask, 'risk_category'] = category
        
        # 분포 출력
        print(f"   위험도 분포:")
        print(df_copy['risk_category'].value_counts())
        
        return df_copy
    
    def analyze_diet_by_risk(self, df_with_risk, indicator):
        """
        위험도별 식습관 차이 분석
        
        Args:
            df_with_risk: risk_category가 포함된 데이터
            indicator: 건강지표 이름
        
        Returns:
            diet_comparison: 위험도별 식습관 비교 DataFrame
        """
        print(f"\n[{indicator}] 위험도별 식습관 분석...")
        
        diet_comparison = []
        
        # 사용 가능한 식습관 변수만
        available_diet_vars = [v for v in self.diet_vars if v in df_with_risk.columns]
        
        for risk in df_with_risk['risk_category'].unique():
            df_risk = df_with_risk[df_with_risk['risk_category'] == risk]
            
            for diet_var in available_diet_vars:
                diet_comparison.append({
                    '위험도': risk,
                    '식습관': diet_var,
                    '평균': df_risk[diet_var].mean(),
                    '표준편차': df_risk[diet_var].std(),
                    'N': len(df_risk)
                })
        
        diet_comp_df = pd.DataFrame(diet_comparison)
        
        # 저장
        output_path = OUTPUT_DIR / f'{indicator}_diet_by_risk.csv'
        diet_comp_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ 위험도별 식습관 저장: {output_path}")
        
        return diet_comp_df
    
    def find_optimal_thresholds(self, df_with_risk, indicator):
        """
        식습관 점수에 대한 최적 임계값 찾기
        
        Args:
            df_with_risk: risk_category가 포함된 데이터
            indicator: 건강지표 이름
        
        Returns:
            optimal_thresholds: 식습관별 최적 임계값
        """
        print(f"\n[{indicator}] 최적 임계값 탐색...")
        
        # 이진 분류: 정상 vs 비정상
        df_with_risk['is_abnormal'] = ~df_with_risk['risk_category'].isin(['정상'])
        
        optimal_thresholds = []
        
        available_diet_vars = [v for v in self.diet_vars if v in df_with_risk.columns]
        
        for diet_var in available_diet_vars:
            # 결측치 제거
            df_clean = df_with_risk[[diet_var, 'is_abnormal']].dropna()
            
            if len(df_clean) < 100:
                continue
            
            # ROC Curve
            fpr, tpr, thresholds = roc_curve(
                df_clean['is_abnormal'],
                df_clean[diet_var]
            )
            
            roc_auc = auc(fpr, tpr)
            
            # Youden's J statistic으로 최적 임계값 찾기
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Percentile 기반 임계값
            p25 = df_clean[diet_var].quantile(0.25)
            p50 = df_clean[diet_var].quantile(0.50)
            p75 = df_clean[diet_var].quantile(0.75)
            
            optimal_thresholds.append({
                '지표': indicator,
                '식습관': diet_var,
                'ROC_AUC': roc_auc,
                '최적임계값': optimal_threshold,
                'Sensitivity': tpr[optimal_idx],
                'Specificity': 1 - fpr[optimal_idx],
                'P25': p25,
                'P50': p50,
                'P75': p75
            })
        
        optimal_df = pd.DataFrame(optimal_thresholds)
        
        # ROC AUC 기준 상위 10개
        optimal_df_sorted = optimal_df.sort_values('ROC_AUC', ascending=False).head(10)
        
        print(f"   Top 10 식습관 (ROC AUC 기준):")
        print(optimal_df_sorted[['식습관', 'ROC_AUC', '최적임계값']].to_string(index=False))
        
        # 저장
        output_path = OUTPUT_DIR / f'{indicator}_optimal_thresholds.csv'
        optimal_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ 최적 임계값 저장: {output_path}")
        
        return optimal_df
    
    def visualize_thresholds(self, df_with_risk, optimal_df, indicator):
        """
        임계값 시각화
        
        Args:
            df_with_risk: risk_category가 포함된 데이터
            optimal_df: 최적 임계값 DataFrame
            indicator: 건강지표 이름
        """
        print(f"\n[{indicator}] 임계값 시각화...")
        
        # Top 5 식습관
        top_5 = optimal_df.nlargest(5, 'ROC_AUC')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_5.iterrows()):
            if idx >= len(axes):
                break
            
            diet_var = row['식습관']
            threshold = row['최적임계값']
            
            # 분포 플롯
            for risk in df_with_risk['risk_category'].unique():
                df_risk = df_with_risk[df_with_risk['risk_category'] == risk]
                
                axes[idx].hist(
                    df_risk[diet_var].dropna(),
                    alpha=0.5,
                    label=risk,
                    bins=20
                )
            
            # 임계값 선
            axes[idx].axvline(
                threshold,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Threshold={threshold:.2f}'
            )
            
            axes[idx].set_xlabel(diet_var, fontsize=12)
            axes[idx].set_ylabel('빈도', fontsize=12)
            axes[idx].set_title(
                f'{diet_var}\n(AUC={row["ROC_AUC"]:.3f})',
                fontsize=12
            )
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)
        
        # 빈 subplot 숨기기
        for idx in range(len(top_5), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(
            f'{indicator} 위험도 예측을 위한 식습관 임계값 (Top 5)',
            fontsize=16, y=1.005
        )
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / f'{indicator}_threshold_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ 임계값 시각화 저장: {output_path}")
        plt.close()
    
    def generate_recommendations(self, optimal_df, indicator):
        """
        맞춤형 건강 권고안 생성
        
        Args:
            optimal_df: 최적 임계값 DataFrame
            indicator: 건강지표 이름
        
        Returns:
            recommendations: 권고안 텍스트
        """
        print(f"\n[{indicator}] 권고안 생성...")
        
        # Top 5 식습관
        top_5 = optimal_df.nlargest(5, 'ROC_AUC')
        
        recommendations = []
        recommendations.append(f"# {indicator} 개선을 위한 식습관 권고안")
        recommendations.append("")
        recommendations.append("## 주요 개선 목표 (중요도 순)")
        recommendations.append("")
        
        for rank, (_, row) in enumerate(top_5.iterrows(), 1):
            diet_var = row['식습관']
            threshold = row['최적임계값']
            auc = row['ROC_AUC']
            p75 = row['P75']
            
            recommendations.append(f"### {rank}. {diet_var}")
            recommendations.append(f"- **중요도**: ROC AUC = {auc:.3f}")
            recommendations.append(f"- **목표 수준**: {threshold:.2f}점 이하 유지")
            recommendations.append(f"- **상위 25% 기준**: {p75:.2f}점")
            recommendations.append("")
        
        recommendations.append("## 실천 방법")
        recommendations.append("")
        recommendations.append("1. **우선순위**: 위에서 제시한 순서대로 개선")
        recommendations.append("2. **단계적 접근**: 한 번에 1-2개씩 개선")
        recommendations.append("3. **지속성**: 최소 3개월 이상 유지")
        recommendations.append("")
        
        rec_text = "\n".join(recommendations)
        
        output_path = OUTPUT_DIR / f'{indicator}_recommendations.md'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rec_text)
        
        print(f"   ✅ 권고안 저장: {output_path}")
        
        return rec_text
    
    def analyze_all_indicators(self):
        """모든 건강지표에 대해 임계값 분석"""
        print("\n" + "="*80)
        print("🚀 임계값 분석 시작")
        print("="*80)
        
        results = {}
        
        for indicator in ['체질량지수', 'SBP', 'DBP', 'TG']:
            print(f"\n{'='*80}")
            print(f"[{indicator}] 분석 시작")
            print(f"{'='*80}")
            
            # 1. 건강 위험도 분류
            df_with_risk = self.classify_health_risk(indicator)
            
            if df_with_risk is None:
                continue
            
            # 2. 위험도별 식습관 비교
            diet_comp_df = self.analyze_diet_by_risk(df_with_risk, indicator)
            
            # 3. 최적 임계값 탐색
            optimal_df = self.find_optimal_thresholds(df_with_risk, indicator)
            
            # 4. 시각화
            self.visualize_thresholds(df_with_risk, optimal_df, indicator)
            
            # 5. 권고안 생성
            recommendations = self.generate_recommendations(optimal_df, indicator)
            
            results[indicator] = {
                'diet_comparison': diet_comp_df,
                'optimal_thresholds': optimal_df,
                'recommendations': recommendations
            }
        
        self.results = results
        
        return results


def main():
    """메인 실행 함수"""
    print("\n" + "="*80)
    print("🚀 Threshold Analysis 시작")
    print("="*80)
    
    # Initialize
    analyzer = ThresholdAnalyzer()
    
    # Analyze all
    results = analyzer.analyze_all_indicators()
    
    print("\n" + "="*80)
    print("✅ Phase 4 완료: 임계값 분석")
    print("="*80)
    print(f"\n결과 저장 위치: {OUTPUT_DIR.absolute()}")
    print(f"\n분석 완료 지표: {len(results)}개")


if __name__ == '__main__':
    main()
