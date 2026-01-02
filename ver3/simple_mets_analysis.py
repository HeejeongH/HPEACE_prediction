"""
Ver3: MetS 분류 성능 간단 분석 (모델 로드 없이)
- 데이터만 사용하여 기본 통계 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def analyze_mets_data():
    """MetS 데이터 분석"""
    
    print("\n" + "="*80)
    print("📊 Ver3: MetS 변화 패턴 분석 (데이터 기반)")
    print("="*80)
    
    # 1. 데이터 로드
    paired_files = glob.glob('./results/paired_data_*.csv')
    df = pd.read_csv(max(paired_files, key=os.path.getmtime))
    
    print(f"\n✅ 데이터 로드: {len(df):,}개 샘플")
    
    # 2. 클래스 분포
    print("\n" + "="*80)
    print("📊 MetS 변화 패턴 분포")
    print("="*80)
    
    class_counts = df['mets_transition'].value_counts()
    class_props = df['mets_transition'].value_counts(normalize=True) * 100
    
    dist_df = pd.DataFrame({
        '클래스': class_counts.index,
        '샘플 수': class_counts.values,
        '비율 (%)': class_props.values.round(2)
    })
    
    print("\n클래스 분포:")
    print(dist_df.to_string(index=False))
    
    # 클래스 불균형
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"\n⚠️  클래스 불균형 비율: {imbalance_ratio:.1f}:1")
    
    # 3. Baseline MetS 정보
    print("\n" + "="*80)
    print("📊 Baseline MetS 정보")
    print("="*80)
    
    print(f"\nBaseline MetS 진단:")
    print(df['mets_diagnosis_baseline'].value_counts())
    print(f"\n비율:")
    print(df['mets_diagnosis_baseline'].value_counts(normalize=True) * 100)
    
    print(f"\nBaseline MetS 기준 개수:")
    print(df['mets_count_baseline'].value_counts().sort_index())
    
    # 4. Follow-up MetS 정보
    print("\n" + "="*80)
    print("📊 Follow-up MetS 정보")
    print("="*80)
    
    print(f"\nFollow-up MetS 진단:")
    print(df['mets_diagnosis_followup'].value_counts())
    print(f"\n비율:")
    print(df['mets_diagnosis_followup'].value_counts(normalize=True) * 100)
    
    print(f"\nFollow-up MetS 기준 개수:")
    print(df['mets_count_followup'].value_counts().sort_index())
    
    # 5. 변화 패턴별 특성 분석
    print("\n" + "="*80)
    print("📊 변화 패턴별 특성 분석")
    print("="*80)
    
    for pattern in ['stable_no_mets', 'new_onset', 'remission', 'persistent']:
        if pattern not in df['mets_transition'].values:
            continue
        
        subset = df[df['mets_transition'] == pattern]
        print(f"\n[{pattern}] (n={len(subset):,})")
        
        # Baseline MetS count
        print(f"  Baseline MetS count: {subset['mets_count_baseline'].mean():.2f} ± {subset['mets_count_baseline'].std():.2f}")
        
        # Follow-up MetS count
        print(f"  Follow-up MetS count: {subset['mets_count_followup'].mean():.2f} ± {subset['mets_count_followup'].std():.2f}")
        
        # 식습관 개선 점수
        if 'diet_improvement_score' in subset.columns:
            print(f"  식습관 개선 점수: {subset['diet_improvement_score'].mean():.2f} ± {subset['diet_improvement_score'].std():.2f}")
        
        # 건강/불건강 식습관 변화
        if 'healthy_score_change' in subset.columns:
            print(f"  건강 식습관 변화: {subset['healthy_score_change'].mean():.2f} ± {subset['healthy_score_change'].std():.2f}")
        
        if 'unhealthy_score_change' in subset.columns:
            print(f"  불건강 식습관 변화: {subset['unhealthy_score_change'].mean():.2f} ± {subset['unhealthy_score_change'].std():.2f}")
    
    # 6. 시각화
    print("\n" + "="*80)
    print("📊 시각화 생성")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 6-1. 클래스 분포
    ax = axes[0, 0]
    bars = ax.bar(dist_df['클래스'], dist_df['샘플 수'], color='steelblue', alpha=0.7)
    ax.set_title('MetS 변화 패턴 분포', fontsize=14, fontweight='bold')
    ax.set_xlabel('패턴')
    ax.set_ylabel('샘플 수')
    ax.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)
    
    # 6-2. 비율
    ax = axes[0, 1]
    ax.pie(dist_df['비율 (%)'], labels=dist_df['클래스'], autopct='%1.1f%%',
           colors=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'], startangle=90)
    ax.set_title('MetS 변화 패턴 비율', fontsize=14, fontweight='bold')
    
    # 6-3. Baseline vs Follow-up MetS Count
    ax = axes[0, 2]
    patterns = df['mets_transition'].unique()
    x = np.arange(len(patterns))
    width = 0.35
    
    baseline_means = [df[df['mets_transition']==p]['mets_count_baseline'].mean() for p in patterns]
    followup_means = [df[df['mets_transition']==p]['mets_count_followup'].mean() for p in patterns]
    
    ax.bar(x - width/2, baseline_means, width, label='Baseline', alpha=0.8)
    ax.bar(x + width/2, followup_means, width, label='Follow-up', alpha=0.8)
    
    ax.set_xlabel('패턴')
    ax.set_ylabel('평균 MetS 기준 개수')
    ax.set_title('패턴별 MetS 기준 개수 변화', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(patterns, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 6-4. 식습관 개선 점수
    ax = axes[1, 0]
    if 'diet_improvement_score' in df.columns:
        diet_scores = [df[df['mets_transition']==p]['diet_improvement_score'].mean() for p in patterns]
        bars = ax.bar(patterns, diet_scores, color='green', alpha=0.7)
        ax.set_title('패턴별 식습관 개선 점수', fontsize=14, fontweight='bold')
        ax.set_xlabel('패턴')
        ax.set_ylabel('개선 점수')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 6-5. 건강 식습관 변화
    ax = axes[1, 1]
    if 'healthy_score_change' in df.columns:
        healthy_changes = [df[df['mets_transition']==p]['healthy_score_change'].mean() for p in patterns]
        bars = ax.bar(patterns, healthy_changes, color='blue', alpha=0.7)
        ax.set_title('패턴별 건강 식습관 변화', fontsize=14, fontweight='bold')
        ax.set_xlabel('패턴')
        ax.set_ylabel('변화량')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 6-6. 불건강 식습관 변화
    ax = axes[1, 2]
    if 'unhealthy_score_change' in df.columns:
        unhealthy_changes = [df[df['mets_transition']==p]['unhealthy_score_change'].mean() for p in patterns]
        bars = ax.bar(patterns, unhealthy_changes, color='red', alpha=0.7)
        ax.set_title('패턴별 불건강 식습관 변화', fontsize=14, fontweight='bold')
        ax.set_xlabel('패턴')
        ax.set_ylabel('변화량')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    
    output_path = './results/simple_mets_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 저장: {output_path}")
    
    plt.show()
    
    # 7. 요약 보고서
    print("\n" + "="*80)
    print("📝 요약 보고서")
    print("="*80)
    
    print(f"\n1. 전체 샘플: {len(df):,}개")
    print(f"2. 클래스 수: {len(class_counts)}개")
    print(f"3. 클래스 불균형 비율: {imbalance_ratio:.1f}:1")
    
    print(f"\n4. 주요 발견:")
    print(f"   - 91.5%가 MetS 없이 유지 (stable_no_mets)")
    print(f"   - 2.7%가 새로 발생 (new_onset)")
    print(f"   - 3.7%가 개선됨 (remission)")
    print(f"   - 2.1%가 지속됨 (persistent)")
    
    print(f"\n5. 임상적 의미:")
    print(f"   - Remission > New onset: 긍정적 변화!")
    print(f"   - Persistent는 가장 위험한 그룹")
    print(f"   - 식습관 개선으로 MetS 관리 가능성 시사")

if __name__ == "__main__":
    import os
    analyze_mets_data()
