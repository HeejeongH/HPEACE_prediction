"""
Advanced Analysis for Ver1 Results - 논문용 심층 분석
=======================================================

1. 세부 그룹 분석 (Subgroup Analysis)
2. SHAP 해석력 분석 (Interpretability)
3. 임계값 분석 (Threshold Analysis)

Author: Research Team
Date: 2025-11-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Korean font
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Output directory
OUTPUT_DIR = Path('./advanced_results')
OUTPUT_DIR.mkdir(exist_ok=True)


class AdvancedAnalysis:
    """Ver1 결과의 고급 분석"""
    
    def __init__(self, data_path='../data/total_again.xlsx'):
        """
        Args:
            data_path: Ver1 원본 데이터 경로
        """
        print("\n" + "="*80)
        print("🔬 Advanced Analysis 초기화")
        print("="*80)
        
        # Load data
        self.df = pd.read_excel(data_path)
        print(f"\n✅ 데이터 로드: {len(self.df):,}개 샘플")
        
        # Health indicators
        self.health_indicators = [
            '체중', '체질량지수', '허리둘레(WAIST)', 'SBP', 'DBP', 'TG'
        ]
        
        # Demographic variables
        self.demo_vars = ['성별', '나이']
        
        print(f"✅ 건강지표: {len(self.health_indicators)}개")
        print(f"✅ 인구통계 변수: {self.demo_vars}")
        
    def check_data_availability(self):
        """데이터 가용성 확인"""
        print("\n" + "="*80)
        print("📊 데이터 가용성 확인")
        print("="*80)
        
        # Check demographic variables
        print("\n[인구통계 변수]")
        for var in self.demo_vars:
            if var in self.df.columns:
                missing = self.df[var].isna().sum()
                missing_pct = missing / len(self.df) * 100
                print(f"   {var}: {len(self.df) - missing:,}개 ({missing}개 결측, {missing_pct:.1f}%)")
                
                if var == '성별':
                    value_counts = self.df[var].value_counts()
                    print(f"      분포: {dict(value_counts)}")
                elif var == '나이':
                    print(f"      범위: {self.df[var].min():.0f} ~ {self.df[var].max():.0f}세")
                    print(f"      평균: {self.df[var].mean():.1f}세 (SD: {self.df[var].std():.1f})")
        
        # Check health indicators
        print("\n[건강지표]")
        for indicator in self.health_indicators:
            if indicator in self.df.columns:
                missing = self.df[indicator].isna().sum()
                missing_pct = missing / len(self.df) * 100
                print(f"   {indicator}: {len(self.df) - missing:,}개 ({missing}개 결측, {missing_pct:.1f}%)")
        
        return True
    
    def create_subgroups(self):
        """세부 그룹 생성"""
        print("\n" + "="*80)
        print("👥 세부 그룹 생성")
        print("="*80)
        
        df_clean = self.df.copy()
        
        # 1. 나이 그룹 (10년 단위)
        df_clean['나이그룹'] = pd.cut(
            df_clean['나이'], 
            bins=[0, 30, 40, 50, 60, 100],
            labels=['20대', '30대', '40대', '50대', '60대+']
        )
        
        # 2. 성별 그룹
        # 성별 값 확인 후 매핑 필요할 수 있음
        
        # 3. BMI 그룹
        if '체질량지수' in df_clean.columns:
            df_clean['BMI그룹'] = pd.cut(
                df_clean['체질량지수'],
                bins=[0, 18.5, 23, 25, 30, 100],
                labels=['저체중', '정상', '과체중', '비만1단계', '비만2단계']
            )
        
        # 그룹별 샘플 수 출력
        print("\n[나이 그룹]")
        print(df_clean['나이그룹'].value_counts().sort_index())
        
        print("\n[성별 그룹]")
        print(df_clean['성별'].value_counts())
        
        if 'BMI그룹' in df_clean.columns:
            print("\n[BMI 그룹]")
            print(df_clean['BMI그룹'].value_counts().sort_index())
        
        self.df_with_groups = df_clean
        
        # Save
        output_path = OUTPUT_DIR / 'data_with_subgroups.csv'
        df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 그룹 정보 저장: {output_path}")
        
        return df_clean
    
    def analyze_subgroup_statistics(self):
        """세부 그룹별 기술통계"""
        print("\n" + "="*80)
        print("📈 세부 그룹별 기술통계")
        print("="*80)
        
        if not hasattr(self, 'df_with_groups'):
            self.create_subgroups()
        
        results = []
        
        # 나이 그룹별
        for age_group in self.df_with_groups['나이그룹'].dropna().unique():
            df_age = self.df_with_groups[self.df_with_groups['나이그룹'] == age_group]
            
            for indicator in self.health_indicators:
                if indicator in df_age.columns:
                    results.append({
                        '그룹유형': '나이',
                        '그룹': age_group,
                        '지표': indicator,
                        'N': len(df_age[indicator].dropna()),
                        '평균': df_age[indicator].mean(),
                        '표준편차': df_age[indicator].std(),
                        '최소': df_age[indicator].min(),
                        '최대': df_age[indicator].max()
                    })
        
        # 성별 그룹별
        for sex in self.df_with_groups['성별'].dropna().unique():
            df_sex = self.df_with_groups[self.df_with_groups['성별'] == sex]
            
            for indicator in self.health_indicators:
                if indicator in df_sex.columns:
                    results.append({
                        '그룹유형': '성별',
                        '그룹': sex,
                        '지표': indicator,
                        'N': len(df_sex[indicator].dropna()),
                        '평균': df_sex[indicator].mean(),
                        '표준편차': df_sex[indicator].std(),
                        '최소': df_sex[indicator].min(),
                        '최대': df_sex[indicator].max()
                    })
        
        # BMI 그룹별
        if 'BMI그룹' in self.df_with_groups.columns:
            for bmi_group in self.df_with_groups['BMI그룹'].dropna().unique():
                df_bmi = self.df_with_groups[self.df_with_groups['BMI그룹'] == bmi_group]
                
                for indicator in self.health_indicators:
                    if indicator in df_bmi.columns:
                        results.append({
                            '그룹유형': 'BMI',
                            '그룹': bmi_group,
                            '지표': indicator,
                            'N': len(df_bmi[indicator].dropna()),
                            '평균': df_bmi[indicator].mean(),
                            '표준편차': df_bmi[indicator].std(),
                            '최소': df_bmi[indicator].min(),
                            '최대': df_bmi[indicator].max()
                        })
        
        results_df = pd.DataFrame(results)
        
        # Save
        output_path = OUTPUT_DIR / 'subgroup_statistics.csv'
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 세부 그룹 통계 저장: {output_path}")
        
        # Print summary
        print("\n[나이 그룹별 체중 평균]")
        age_weight = results_df[
            (results_df['그룹유형'] == '나이') & 
            (results_df['지표'] == '체중')
        ][['그룹', '평균', '표준편차', 'N']]
        print(age_weight.to_string(index=False))
        
        print("\n[성별 그룹별 체중 평균]")
        sex_weight = results_df[
            (results_df['그룹유형'] == '성별') & 
            (results_df['지표'] == '체중')
        ][['그룹', '평균', '표준편차', 'N']]
        print(sex_weight.to_string(index=False))
        
        return results_df
    
    def visualize_subgroup_distributions(self):
        """세부 그룹 분포 시각화"""
        print("\n" + "="*80)
        print("📊 세부 그룹 분포 시각화")
        print("="*80)
        
        if not hasattr(self, 'df_with_groups'):
            self.create_subgroups()
        
        # 1. 나이 그룹별 건강지표 분포
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, indicator in enumerate(self.health_indicators):
            if indicator in self.df_with_groups.columns:
                self.df_with_groups.boxplot(
                    column=indicator,
                    by='나이그룹',
                    ax=axes[idx]
                )
                axes[idx].set_title(f'{indicator} by 나이그룹')
                axes[idx].set_xlabel('나이그룹')
                axes[idx].set_ylabel(indicator)
        
        plt.suptitle('나이 그룹별 건강지표 분포', fontsize=16, y=1.02)
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / 'subgroup_age_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 나이 그룹 분포 저장: {output_path}")
        plt.close()
        
        # 2. 성별 건강지표 분포
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, indicator in enumerate(self.health_indicators):
            if indicator in self.df_with_groups.columns:
                self.df_with_groups.boxplot(
                    column=indicator,
                    by='성별',
                    ax=axes[idx]
                )
                axes[idx].set_title(f'{indicator} by 성별')
                axes[idx].set_xlabel('성별')
                axes[idx].set_ylabel(indicator)
        
        plt.suptitle('성별 건강지표 분포', fontsize=16, y=1.02)
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / 'subgroup_sex_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 성별 분포 저장: {output_path}")
        plt.close()
        
        # 3. BMI 그룹별 건강지표 분포
        if 'BMI그룹' in self.df_with_groups.columns:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for idx, indicator in enumerate(self.health_indicators):
                if indicator in self.df_with_groups.columns and indicator != '체질량지수':
                    self.df_with_groups.boxplot(
                        column=indicator,
                        by='BMI그룹',
                        ax=axes[idx]
                    )
                    axes[idx].set_title(f'{indicator} by BMI그룹')
                    axes[idx].set_xlabel('BMI그룹')
                    axes[idx].set_ylabel(indicator)
            
            plt.suptitle('BMI 그룹별 건강지표 분포', fontsize=16, y=1.02)
            plt.tight_layout()
            
            output_path = OUTPUT_DIR / 'subgroup_bmi_distribution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ BMI 그룹 분포 저장: {output_path}")
            plt.close()
        
        return True


def main():
    """메인 실행 함수"""
    print("\n" + "="*80)
    print("🚀 Ver1 Advanced Analysis 시작")
    print("="*80)
    print("\n분석 항목:")
    print("   1. 세부 그룹 분석 (나이/성별/BMI)")
    print("   2. SHAP 해석력 분석")
    print("   3. 임계값 분석")
    print("\n" + "="*80)
    
    # Initialize
    analyzer = AdvancedAnalysis()
    
    # Step 1: Check data
    analyzer.check_data_availability()
    
    # Step 2: Create subgroups
    analyzer.create_subgroups()
    
    # Step 3: Analyze subgroup statistics
    analyzer.analyze_subgroup_statistics()
    
    # Step 4: Visualize distributions
    analyzer.visualize_subgroup_distributions()
    
    print("\n" + "="*80)
    print("✅ Phase 1 완료: 세부 그룹 기초 분석")
    print("="*80)
    print(f"\n결과 저장 위치: {OUTPUT_DIR.absolute()}")
    print("\n다음 단계:")
    print("   - 세부 그룹별 모델 학습 (더 높은 정확도 기대)")
    print("   - SHAP 분석 (특성 중요도 및 상호작용)")
    print("   - 임계값 분석 (건강 위험 기준점)")


if __name__ == '__main__':
    main()
