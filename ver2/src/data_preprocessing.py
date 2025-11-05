"""
Ver2: Longitudinal Change Prediction - 데이터 전처리
=======================================================

목적: 개별 방문 데이터 → Paired visits (방문 쌍) 데이터 변환

입력: Ver1 데이터 (각 행 = 1번 방문)
출력: Paired data (각 행 = 2번 방문의 변화)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_data(file_path='../data/total_again.xlsx'):
    """원본 데이터 로드"""
    print("=" * 80)
    print("📂 Step 1: 데이터 로드")
    print("=" * 80)
    
    df = pd.read_excel(file_path, index_col='R-ID')
    
    print(f"   ✅ 총 데이터: {len(df):,}건")
    print(f"   ✅ 참여자 수: {df.index.nunique():,}명")
    print(f"   ✅ 변수 수: {len(df.columns)}개")
    
    return df


def analyze_visit_patterns(df):
    """방문 패턴 분석"""
    print("\n" + "=" * 80)
    print("📊 Step 2: 방문 패턴 분석")
    print("=" * 80)
    
    # 참여자별 방문 횟수
    visit_counts = df.groupby(level=0).size()
    
    print(f"\n📈 방문 횟수 분포:")
    print(f"   평균: {visit_counts.mean():.2f}회")
    print(f"   중앙값: {visit_counts.median():.0f}회")
    print(f"   최소: {visit_counts.min()}회")
    print(f"   최대: {visit_counts.max()}회")
    
    print(f"\n📊 방문 횟수별 참여자 수:")
    for n_visits in sorted(visit_counts.unique()):
        n_people = (visit_counts == n_visits).sum()
        pct = n_people / len(visit_counts) * 100
        print(f"   {n_visits}회: {n_people:,}명 ({pct:.1f}%)")
    
    # Paired 생성 가능한 참여자
    paired_possible = (visit_counts >= 2).sum()
    print(f"\n✅ Paired visits 생성 가능: {paired_possible:,}명")
    print(f"   (전체의 {paired_possible/len(visit_counts)*100:.1f}%)")
    
    return visit_counts


def create_paired_visits(df, 
                         min_time_gap=30,
                         max_time_gap=365,
                         date_column='수진일'):
    """
    방문 쌍(Paired visits) 생성
    
    Parameters:
    -----------
    df : DataFrame
        원본 데이터
    min_time_gap : int
        최소 시간 간격 (일)
    max_time_gap : int
        최대 시간 간격 (일)
    date_column : str
        날짜 컬럼명
    
    Returns:
    --------
    paired_df : DataFrame
        방문 쌍 데이터
    """
    print("\n" + "=" * 80)
    print("🔄 Step 3: Paired Visits 생성")
    print("=" * 80)
    print(f"   설정: {min_time_gap}일 ≤ 간격 ≤ {max_time_gap}일")
    
    paired_data = []
    skipped = {'no_date': 0, 'single_visit': 0, 'time_gap': 0}
    
    # 날짜 컬럼 확인
    if date_column not in df.columns:
        print(f"\n⚠️ 경고: '{date_column}' 컬럼이 없습니다.")
        print(f"   사용 가능한 컬럼: {list(df.columns[:10])}...")
        date_column = None
    
    # 참여자별 처리
    for person_id in df.index.unique():
        person_visits = df.loc[person_id]
        
        # 단일 방문인 경우 (Series)
        if isinstance(person_visits, pd.Series):
            skipped['single_visit'] += 1
            continue
        
        # 날짜 정렬 (날짜 컬럼이 있는 경우)
        if date_column and date_column in person_visits.columns:
            person_visits = person_visits.sort_values(date_column)
        
        # 연속된 방문 쌍 생성
        for i in range(len(person_visits) - 1):
            visit_before = person_visits.iloc[i]
            visit_after = person_visits.iloc[i + 1]
            
            # 시간 간격 계산 (날짜가 있는 경우)
            if date_column and date_column in person_visits.columns:
                try:
                    date_before = pd.to_datetime(visit_before[date_column])
                    date_after = pd.to_datetime(visit_after[date_column])
                    time_gap = (date_after - date_before).days
                    
                    # 시간 간격 필터링
                    if time_gap < min_time_gap or time_gap > max_time_gap:
                        skipped['time_gap'] += 1
                        continue
                except:
                    time_gap = None
            else:
                time_gap = None
            
            # Paired data 생성
            pair = {
                'person_id': person_id,
                'visit_pair': f"{i+1}->{i+2}",
                'time_gap_days': time_gap
            }
            
            # 식습관 변수들
            diet_vars = [
                '간식빈도', '고지방 육류', '단맛', '단백질류', '담배피는데근처있는빈도',
                '곡류', '과일', '너무 빨리 먹는 식습관', '밤늦게 야식', '야채샐러드드레싱',
                '유제품', '음료류', '인스턴트 가공식품', '저녁식사시간', '짠 간', '짠 식습관',
                '채소', '튀김', '아침식사빈도'
            ]
            
            # 식습관 변화량 계산
            for var in diet_vars:
                if var in visit_before.index and var in visit_after.index:
                    before_val = visit_before[var]
                    after_val = visit_after[var]
                    
                    # 변화량
                    pair[f'{var}_before'] = before_val
                    pair[f'{var}_after'] = after_val
                    pair[f'{var}_change'] = after_val - before_val
                    
                    # 변화율 (0으로 나누기 방지)
                    if before_val != 0:
                        pair[f'{var}_change_pct'] = (after_val - before_val) / before_val * 100
            
            # 타겟 바이오마커 (건강지표)
            biomarkers = ['체중', '체질량지수', '허리둘레(WAIST)', 'SBP', 'DBP', 'TG']
            
            for bio in biomarkers:
                if bio in visit_before.index and bio in visit_after.index:
                    before_val = visit_before[bio]
                    after_val = visit_after[bio]
                    
                    # 베이스라인 (시작점)
                    pair[f'{bio}_baseline'] = before_val
                    
                    # 변화량 (타겟)
                    pair[f'{bio}_change'] = after_val - before_val
                    
                    # 변화율
                    if before_val != 0:
                        pair[f'{bio}_change_pct'] = (after_val - before_val) / before_val * 100
                    
                    # 최종값
                    pair[f'{bio}_final'] = after_val
            
            paired_data.append(pair)
    
    # DataFrame 변환
    paired_df = pd.DataFrame(paired_data)
    
    print(f"\n✅ 생성 완료:")
    print(f"   총 Paired visits: {len(paired_df):,}개")
    print(f"   평균 시간 간격: {paired_df['time_gap_days'].mean():.0f}일")
    
    print(f"\n📊 제외된 데이터:")
    print(f"   단일 방문: {skipped['single_visit']:,}명")
    print(f"   시간 간격 불충족: {skipped['time_gap']:,}쌍")
    
    return paired_df


def calculate_derived_features(paired_df):
    """파생 특성 생성"""
    print("\n" + "=" * 80)
    print("🔧 Step 4: 파생 특성 생성")
    print("=" * 80)
    
    df = paired_df.copy()
    
    # 1. 위험 식습관 변화 합계
    risk_habits = [
        '고지방 육류_change', '튀김_change', '인스턴트 가공식품_change',
        '음료류_change', '단맛_change', '야식_change', '짠 식습관_change'
    ]
    
    risk_cols = [col for col in risk_habits if col in df.columns]
    if risk_cols:
        df['risk_habits_total_change'] = df[risk_cols].sum(axis=1)
        print(f"   ✅ 위험 식습관 총 변화량 계산")
    
    # 2. 보호 식습관 변화 합계
    protective_habits = [
        '채소_change', '과일_change', '유제품_change', '아침식사빈도_change'
    ]
    
    protective_cols = [col for col in protective_habits if col in df.columns]
    if protective_cols:
        df['protective_habits_total_change'] = df[protective_cols].sum(axis=1)
        print(f"   ✅ 보호 식습관 총 변화량 계산")
    
    # 3. 순 식습관 개선도
    if 'risk_habits_total_change' in df.columns and 'protective_habits_total_change' in df.columns:
        df['net_diet_improvement'] = df['protective_habits_total_change'] - df['risk_habits_total_change']
        print(f"   ✅ 순 식습관 개선도 계산")
    
    # 4. 시간 정규화 변화율 (월간 변화율)
    # ⚠️ 주의: 타겟 변수(건강지표)의 _change는 제외 (Data Leakage 방지)
    target_biomarkers = ['체중', '체질량지수', '허리둘레(WAIST)', 'SBP', 'DBP', 'TG']
    if 'time_gap_days' in df.columns:
        for col in df.columns:
            if col.endswith('_change') and not col.endswith('_change_pct'):
                # 타겟 변수는 제외
                is_target = any(col.startswith(f'{bio}_change') for bio in target_biomarkers)
                if not is_target:
                    months = df['time_gap_days'] / 30.0
                    df[f'{col}_per_month'] = df[col] / months
        print(f"   ✅ 월간 변화율 계산 (타겟 변수 제외)")
    
    print(f"\n✅ 파생 특성 생성 완료: {len(df.columns) - len(paired_df.columns)}개 추가")
    
    return df


def exploratory_data_analysis(paired_df, output_dir='../result'):
    """탐색적 데이터 분석 및 시각화"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("📊 Step 5: 탐색적 데이터 분석 (EDA)")
    print("=" * 80)
    
    # 1. 시간 간격 분포
    if 'time_gap_days' in paired_df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(paired_df['time_gap_days'], bins=50, edgecolor='black')
        plt.xlabel('시간 간격 (일)', fontsize=12)
        plt.ylabel('빈도', fontsize=12)
        plt.title('방문 간 시간 간격 분포', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(f'{output_dir}/time_gap_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 시간 간격 분포 저장: {output_dir}/time_gap_distribution.png")
    
    # 2. 체중 변화 분포
    if '체중_change' in paired_df.columns:
        plt.figure(figsize=(10, 6))
        weight_change = paired_df['체중_change'].dropna()
        plt.hist(weight_change, bins=50, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label='변화 없음')
        plt.xlabel('체중 변화 (kg)', fontsize=12)
        plt.ylabel('빈도', fontsize=12)
        plt.title('체중 변화 분포', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(f'{output_dir}/weight_change_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 통계
        print(f"\n   📈 체중 변화 통계:")
        print(f"      평균: {weight_change.mean():.2f} kg")
        print(f"      중앙값: {weight_change.median():.2f} kg")
        print(f"      표준편차: {weight_change.std():.2f} kg")
        print(f"      증가: {(weight_change > 0).sum():,}명 ({(weight_change > 0).sum()/len(weight_change)*100:.1f}%)")
        print(f"      감소: {(weight_change < 0).sum():,}명 ({(weight_change < 0).sum()/len(weight_change)*100:.1f}%)")
        print(f"      유지: {(weight_change == 0).sum():,}명 ({(weight_change == 0).sum()/len(weight_change)*100:.1f}%)")
    
    # 3. 식습관 변화와 건강지표 변화 상관관계
    diet_changes = [col for col in paired_df.columns if '_change' in col and not 'baseline' in col and not 'final' in col]
    health_changes = ['체중_change', 'BMI_change', 'SBP_change', 'DBP_change', 'TG_change']
    health_changes = [col for col in health_changes if col in paired_df.columns]
    
    if len(diet_changes) > 0 and len(health_changes) > 0:
        # 주요 식습관만 선택
        key_diet_changes = [
            '고지방 육류_change', '채소_change', '야식_change', 
            '짠 식습관_change', '단맛_change'
        ]
        key_diet_changes = [col for col in key_diet_changes if col in paired_df.columns]
        
        if len(key_diet_changes) > 0:
            correlation_data = paired_df[key_diet_changes + health_changes].corr()
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_data.loc[key_diet_changes, health_changes],
                       annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
                       xticklabels=[col.replace('_change', '') for col in health_changes],
                       yticklabels=[col.replace('_change', '') for col in key_diet_changes])
            plt.title('식습관 변화 vs 건강지표 변화 상관관계', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ 상관관계 히트맵 저장: {output_dir}/correlation_heatmap.png")
    
    print(f"\n✅ EDA 완료! 결과 저장: {output_dir}/")


def save_processed_data(paired_df, output_path='../data/ver2_paired_visits.csv'):
    """처리된 데이터 저장"""
    print("\n" + "=" * 80)
    print("💾 Step 6: 데이터 저장")
    print("=" * 80)
    
    paired_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 저장 완료: {output_path}")
    print(f"   📊 크기: {len(paired_df):,}행 × {len(paired_df.columns)}열")
    
    # 요약 통계 저장
    summary_path = output_path.replace('.csv', '_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Ver2: Longitudinal Change Prediction - 데이터 요약\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"총 Paired visits: {len(paired_df):,}개\n")
        f.write(f"참여자 수: {paired_df['person_id'].nunique():,}명\n")
        f.write(f"특성 수: {len(paired_df.columns)}개\n\n")
        f.write(paired_df.describe().to_string())
    
    print(f"   ✅ 요약 통계 저장: {summary_path}")


def main():
    """메인 실행 함수"""
    import os
    
    print("\n")
    print("=" * 80)
    print("Ver2: Longitudinal Change Prediction - 데이터 전처리")
    print("=" * 80)
    print("\n⏱️ 시작 시간:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # 작업 디렉토리 기준 경로 설정 (run_ver2.py에서 ver2/로 chdir 했음)
    base_dir = os.getcwd()  # ver2/
    data_dir = os.path.join(base_dir, '..', 'data')  # project_root/data/
    result_dir = os.path.join(base_dir, 'result')  # ver2/result/
    
    input_file = os.path.join(data_dir, 'total_again.xlsx')
    output_file = os.path.join(data_dir, 'ver2_paired_visits.csv')
    
    # 1. 데이터 로드
    df = load_data(input_file)
    
    # 2. 방문 패턴 분석
    visit_counts = analyze_visit_patterns(df)
    
    # 3. Paired visits 생성
    paired_df = create_paired_visits(df, min_time_gap=30, max_time_gap=365)
    
    if len(paired_df) == 0:
        print("\n❌ 오류: Paired visits가 생성되지 않았습니다.")
        return
    
    # 4. 파생 특성 생성
    paired_df = calculate_derived_features(paired_df)
    
    # 5. 탐색적 데이터 분석
    exploratory_data_analysis(paired_df, output_dir=result_dir)
    
    # 6. 데이터 저장
    save_processed_data(paired_df, output_path=output_file)
    
    print("\n" + "=" * 80)
    print("✅ 전처리 완료!")
    print("=" * 80)
    print(f"\n⏱️ 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 다음 단계:")
    print(f"   1. 생성된 데이터 확인: data/ver2_paired_visits.csv")
    print(f"   2. EDA 결과 확인: result/ver2_eda/")
    print(f"   3. 모델 학습 시작: python model_training.py")


if __name__ == "__main__":
    main()
