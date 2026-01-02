"""
Ver3: 식습관 변화 기반 건강지표 및 MetS 예측 모델
=======================================================

목적: 두 번 연속 방문자 데이터를 활용하여
      1) 건강지표 변화 예측 (Regression)
      2) MetS 발생/개선 예측 (Classification)
      
핵심 차별점:
- Ver1: 단면 분석 (식습관 → 건강지표)
- Ver2: 변화량만 예측 (식습관 변화 → 건강지표 변화)
- Ver3: 기준선(baseline) + 변화량 통합 예측

저자: SNUH Prediction Team
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class PairedVisitPreprocessor:
    """두 번 연속 방문자 데이터 전처리 클래스"""
    
    def __init__(self, 
                 min_time_gap: int = 90,
                 max_time_gap: int = 365,
                 date_column: str = '수진일'):
        """
        Parameters
        ----------
        min_time_gap : int
            최소 방문 간격 (일), 기본값 90일 (약 3개월)
        max_time_gap : int
            최대 방문 간격 (일), 기본값 365일 (1년)
        date_column : str
            날짜 컬럼명
        """
        self.min_time_gap = min_time_gap
        self.max_time_gap = max_time_gap
        self.date_column = date_column
        
        # 식습관 변수 정의 (19개)
        self.diet_vars = [
            '간식빈도', '고지방 육류', '단맛', '단백질류', '곡류',
            '과일', '유제품', '음료류', '인스턴트 가공식품', '짠 간',
            '짠 식습관', '채소', '튀김'
        ]
        
        # 건강지표 변수 정의
        self.health_vars = [
            '체중', '체질량지수', '허리둘레(WAIST)', 
            'SBP', 'DBP', 'TG', 
            'HDL', 'glucose', 'HbA1c'
        ]
        
        # MetS 기준 (Korean criteria)
        self.mets_criteria = {
            '허리둘레(WAIST)': {'M': 90, 'F': 85},  # cm
            'TG': 150,  # mg/dL
            'HDL': {'M': 40, 'F': 50},  # mg/dL
            'SBP': 130,  # mmHg
            'DBP': 85,   # mmHg
            'glucose': 100  # mg/dL
        }
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """원본 데이터 로드"""
        print("=" * 80)
        print("📂 Ver3: 데이터 로드")
        print("=" * 80)
        
        df = pd.read_excel(file_path, index_col='R-ID')
        
        print(f"✅ 총 레코드: {len(df):,}건")
        print(f"✅ 참여자 수: {df.index.nunique():,}명")
        print(f"✅ 변수 수: {len(df.columns)}개")
        
        return df
    
    def analyze_visit_patterns(self, df: pd.DataFrame) -> pd.Series:
        """방문 패턴 분석"""
        print("\n" + "=" * 80)
        print("📊 방문 패턴 분석")
        print("=" * 80)
        
        visit_counts = df.groupby(level=0).size()
        
        print(f"\n📈 방문 횟수 통계:")
        print(f"   평균: {visit_counts.mean():.2f}회")
        print(f"   중앙값: {visit_counts.median():.0f}회")
        print(f"   최소~최대: {visit_counts.min()}~{visit_counts.max()}회")
        
        print(f"\n📊 방문 횟수별 분포:")
        for n_visits in sorted(visit_counts.unique()):
            n_people = (visit_counts == n_visits).sum()
            pct = n_people / len(visit_counts) * 100
            print(f"   {n_visits}회 방문: {n_people:,}명 ({pct:.1f}%)")
        
        paired_possible = (visit_counts >= 2).sum()
        print(f"\n✅ 2회 이상 방문자: {paired_possible:,}명 ({paired_possible/len(visit_counts)*100:.1f}%)")
        
        return visit_counts
    
    def create_paired_visits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        방문 쌍 데이터 생성
        
        Returns
        -------
        paired_df : DataFrame
            각 행이 연속된 두 번의 방문을 나타내는 데이터
            - baseline 변수: 첫 번째 방문 시점의 값
            - change 변수: 두 번째 방문에서의 변화량
        """
        print("\n" + "=" * 80)
        print("🔄 Paired Visits 생성")
        print("=" * 80)
        print(f"   조건: {self.min_time_gap}일 ≤ 방문 간격 ≤ {self.max_time_gap}일")
        
        paired_data = []
        stats = {'total_pairs': 0, 'valid_pairs': 0, 
                 'time_gap_failed': 0, 'missing_data': 0}
        
        # 참여자별로 반복
        for participant_id in df.index.unique():
            participant_df = df.loc[[participant_id]].copy()
            
            # 1회만 방문한 경우 스킵
            if len(participant_df) < 2:
                continue
            
            # 날짜로 정렬
            if self.date_column in participant_df.columns:
                participant_df = participant_df.sort_values(self.date_column)
            
            # 연속된 방문 쌍 생성
            for i in range(len(participant_df) - 1):
                visit1 = participant_df.iloc[i]
                visit2 = participant_df.iloc[i + 1]
                
                stats['total_pairs'] += 1
                
                # 시간 간격 확인
                if self.date_column in participant_df.columns:
                    try:
                        date1 = pd.to_datetime(visit1[self.date_column])
                        date2 = pd.to_datetime(visit2[self.date_column])
                        time_gap = (date2 - date1).days
                        
                        if time_gap < self.min_time_gap or time_gap > self.max_time_gap:
                            stats['time_gap_failed'] += 1
                            continue
                    except:
                        time_gap = np.nan
                else:
                    time_gap = np.nan
                
                # Paired 데이터 생성
                paired_row = self._create_paired_row(
                    visit1, visit2, participant_id, time_gap
                )
                
                if paired_row is not None:
                    paired_data.append(paired_row)
                    stats['valid_pairs'] += 1
                else:
                    stats['missing_data'] += 1
        
        paired_df = pd.DataFrame(paired_data)
        
        # 통계 출력
        print(f"\n📊 생성 결과:")
        print(f"   전체 방문 쌍: {stats['total_pairs']:,}개")
        print(f"   ✅ 유효한 쌍: {stats['valid_pairs']:,}개")
        print(f"   ❌ 시간 간격 미달: {stats['time_gap_failed']:,}개")
        print(f"   ❌ 결측치 과다: {stats['missing_data']:,}개")
        
        return paired_df
    
    def _create_paired_row(self, 
                          visit1: pd.Series, 
                          visit2: pd.Series,
                          participant_id: str,
                          time_gap: float) -> Dict:
        """단일 방문 쌍에서 특성 추출"""
        
        paired_row = {
            'participant_id': participant_id,
            'time_gap_days': time_gap,
            'sex': visit1.get('성별', np.nan),
            'age_baseline': visit1.get('나이', np.nan)
        }
        
        # 1. 식습관 변수 (baseline + change)
        for var in self.diet_vars:
            if var in visit1 and var in visit2:
                baseline = visit1[var]
                follow_up = visit2[var]
                
                # Baseline 값
                paired_row[f'{var}_baseline'] = baseline
                
                # 절대 변화량
                paired_row[f'{var}_change'] = follow_up - baseline
                
                # 퍼센트 변화율 (baseline이 0이 아닐 때만)
                if baseline != 0:
                    paired_row[f'{var}_change_pct'] = (follow_up - baseline) / baseline * 100
                else:
                    paired_row[f'{var}_change_pct'] = 0
        
        # 2. 건강지표 변수 (baseline + change)
        for var in self.health_vars:
            if var in visit1 and var in visit2:
                baseline = visit1[var]
                follow_up = visit2[var]
                
                paired_row[f'{var}_baseline'] = baseline
                paired_row[f'{var}_change'] = follow_up - baseline
                
                if baseline != 0:
                    paired_row[f'{var}_change_pct'] = (follow_up - baseline) / baseline * 100
                else:
                    paired_row[f'{var}_change_pct'] = 0
        
        # 결측치 확인
        if sum(pd.isna(v) for v in paired_row.values()) > len(paired_row) * 0.3:
            return None
        
        return paired_row
    
    def calculate_mets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MetS (대사증후군) 진단
        
        5개 기준 중 3개 이상 충족 시 MetS
        1. 복부비만 (허리둘레: 남 ≥90cm, 여 ≥85cm)
        2. 고중성지방 (TG ≥150 mg/dL)
        3. 낮은 HDL (남 <40 mg/dL, 여 <50 mg/dL)
        4. 고혈압 (SBP ≥130 or DBP ≥85 mmHg)
        5. 고혈당 (glucose ≥100 mg/dL)
        """
        print("\n" + "=" * 80)
        print("🏥 MetS (대사증후군) 진단")
        print("=" * 80)
        
        df = df.copy()
        
        # Baseline MetS 진단
        df['mets_waist_baseline'] = 0
        df['mets_tg_baseline'] = 0
        df['mets_hdl_baseline'] = 0
        df['mets_bp_baseline'] = 0
        df['mets_glucose_baseline'] = 0
        
        # 1. 복부비만
        male_mask = df['sex'] == 'M'
        female_mask = df['sex'] == 'F'
        
        df.loc[male_mask, 'mets_waist_baseline'] = (
            df.loc[male_mask, '허리둘레(WAIST)_baseline'] >= self.mets_criteria['허리둘레(WAIST)']['M']
        ).astype(int)
        
        df.loc[female_mask, 'mets_waist_baseline'] = (
            df.loc[female_mask, '허리둘레(WAIST)_baseline'] >= self.mets_criteria['허리둘레(WAIST)']['F']
        ).astype(int)
        
        # 2. 고중성지방
        df['mets_tg_baseline'] = (df['TG_baseline'] >= self.mets_criteria['TG']).astype(int)
        
        # 3. 낮은 HDL
        df.loc[male_mask, 'mets_hdl_baseline'] = (
            df.loc[male_mask, 'HDL_baseline'] < self.mets_criteria['HDL']['M']
        ).astype(int)
        
        df.loc[female_mask, 'mets_hdl_baseline'] = (
            df.loc[female_mask, 'HDL_baseline'] < self.mets_criteria['HDL']['F']
        ).astype(int)
        
        # 4. 고혈압
        df['mets_bp_baseline'] = (
            (df['SBP_baseline'] >= self.mets_criteria['SBP']) |
            (df['DBP_baseline'] >= self.mets_criteria['DBP'])
        ).astype(int)
        
        # 5. 고혈당
        df['mets_glucose_baseline'] = (
            df['glucose_baseline'] >= self.mets_criteria['glucose']
        ).astype(int)
        
        # MetS 개수 및 진단
        df['mets_count_baseline'] = (
            df['mets_waist_baseline'] +
            df['mets_tg_baseline'] +
            df['mets_hdl_baseline'] +
            df['mets_bp_baseline'] +
            df['mets_glucose_baseline']
        )
        
        df['mets_diagnosis_baseline'] = (df['mets_count_baseline'] >= 3).astype(int)
        
        # Follow-up MetS 진단 (baseline + change)
        df['허리둘레(WAIST)_followup'] = df['허리둘레(WAIST)_baseline'] + df['허리둘레(WAIST)_change']
        df['TG_followup'] = df['TG_baseline'] + df['TG_change']
        df['HDL_followup'] = df['HDL_baseline'] + df['HDL_change']
        df['SBP_followup'] = df['SBP_baseline'] + df['SBP_change']
        df['DBP_followup'] = df['DBP_baseline'] + df['DBP_change']
        df['glucose_followup'] = df['glucose_baseline'] + df['glucose_change']
        
        df['mets_waist_followup'] = 0
        df['mets_hdl_followup'] = 0
        
        df.loc[male_mask, 'mets_waist_followup'] = (
            df.loc[male_mask, '허리둘레(WAIST)_followup'] >= self.mets_criteria['허리둘레(WAIST)']['M']
        ).astype(int)
        
        df.loc[female_mask, 'mets_waist_followup'] = (
            df.loc[female_mask, '허리둘레(WAIST)_followup'] >= self.mets_criteria['허리둘레(WAIST)']['F']
        ).astype(int)
        
        df['mets_tg_followup'] = (df['TG_followup'] >= self.mets_criteria['TG']).astype(int)
        
        df.loc[male_mask, 'mets_hdl_followup'] = (
            df.loc[male_mask, 'HDL_followup'] < self.mets_criteria['HDL']['M']
        ).astype(int)
        
        df.loc[female_mask, 'mets_hdl_followup'] = (
            df.loc[female_mask, 'HDL_followup'] < self.mets_criteria['HDL']['F']
        ).astype(int)
        
        df['mets_bp_followup'] = (
            (df['SBP_followup'] >= self.mets_criteria['SBP']) |
            (df['DBP_followup'] >= self.mets_criteria['DBP'])
        ).astype(int)
        
        df['mets_glucose_followup'] = (
            df['glucose_followup'] >= self.mets_criteria['glucose']
        ).astype(int)
        
        df['mets_count_followup'] = (
            df['mets_waist_followup'] +
            df['mets_tg_followup'] +
            df['mets_hdl_followup'] +
            df['mets_bp_followup'] +
            df['mets_glucose_followup']
        )
        
        df['mets_diagnosis_followup'] = (df['mets_count_followup'] >= 3).astype(int)
        
        # MetS 변화 분류
        df['mets_transition'] = 'stable_no_mets'
        df.loc[(df['mets_diagnosis_baseline'] == 0) & (df['mets_diagnosis_followup'] == 1), 'mets_transition'] = 'new_onset'
        df.loc[(df['mets_diagnosis_baseline'] == 1) & (df['mets_diagnosis_followup'] == 0), 'mets_transition'] = 'remission'
        df.loc[(df['mets_diagnosis_baseline'] == 1) & (df['mets_diagnosis_followup'] == 1), 'mets_transition'] = 'persistent'
        
        # 통계 출력
        print(f"\n📊 Baseline MetS 유병률:")
        mets_baseline = df['mets_diagnosis_baseline'].sum()
        print(f"   MetS 있음: {mets_baseline:,}명 ({mets_baseline/len(df)*100:.1f}%)")
        print(f"   MetS 없음: {len(df)-mets_baseline:,}명 ({(len(df)-mets_baseline)/len(df)*100:.1f}%)")
        
        print(f"\n📊 Follow-up MetS 유병률:")
        mets_followup = df['mets_diagnosis_followup'].sum()
        print(f"   MetS 있음: {mets_followup:,}명 ({mets_followup/len(df)*100:.1f}%)")
        print(f"   MetS 없음: {len(df)-mets_followup:,}명 ({(len(df)-mets_followup)/len(df)*100:.1f}%)")
        
        print(f"\n📊 MetS 변화 패턴:")
        for transition in ['stable_no_mets', 'new_onset', 'remission', 'persistent']:
            count = (df['mets_transition'] == transition).sum()
            pct = count / len(df) * 100
            print(f"   {transition}: {count:,}명 ({pct:.1f}%)")
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        고급 특성 엔지니어링
        
        1. 종합 식습관 점수 (건강/위험)
        2. 식습관 다양성 지수
        3. 변화 강도 (월별 변화율)
        4. 상호작용 특성
        """
        print("\n" + "=" * 80)
        print("🔧 고급 특성 엔지니어링")
        print("=" * 80)
        
        df = df.copy()
        
        # 건강한 식습관 (높을수록 좋음)
        healthy_foods_baseline = ['채소', '과일', '단백질류', '유제품', '곡류']
        df['healthy_score_baseline'] = df[[f'{food}_baseline' for food in healthy_foods_baseline]].sum(axis=1)
        df['healthy_score_change'] = df[[f'{food}_change' for food in healthy_foods_baseline]].sum(axis=1)
        
        # 위험한 식습관 (낮을수록 좋음)
        unhealthy_foods_baseline = ['튀김', '인스턴트 가공식품', '고지방 육류', '음료류', '단맛', '짠 식습관', '간식빈도']
        df['unhealthy_score_baseline'] = df[[f'{food}_baseline' for food in unhealthy_foods_baseline]].sum(axis=1)
        df['unhealthy_score_change'] = df[[f'{food}_change' for food in unhealthy_foods_baseline]].sum(axis=1)
        
        # 종합 식습관 개선 점수 (증가할수록 식습관 개선)
        df['diet_improvement_score'] = df['healthy_score_change'] - df['unhealthy_score_change']
        
        # 식습관 다양성 (섭취하는 식품군 수)
        all_foods_baseline = healthy_foods_baseline + unhealthy_foods_baseline
        df['diet_diversity_baseline'] = df[[f'{food}_baseline' for food in all_foods_baseline]].apply(
            lambda x: (x > 1).sum(), axis=1
        )
        
        # 월별 변화 강도
        df['monthly_weight_change'] = df['체중_change'] / (df['time_gap_days'] / 30)
        df['monthly_waist_change'] = df['허리둘레(WAIST)_change'] / (df['time_gap_days'] / 30)
        df['monthly_sbp_change'] = df['SBP_change'] / (df['time_gap_days'] / 30)
        df['monthly_dbp_change'] = df['DBP_change'] / (df['time_gap_days'] / 30)
        df['monthly_tg_change'] = df['TG_change'] / (df['time_gap_days'] / 30)
        
        # BMI 카테고리
        df['bmi_category_baseline'] = pd.cut(
            df['체질량지수_baseline'],
            bins=[0, 18.5, 23, 25, 30, 100],
            labels=['저체중', '정상', '과체중', '비만1단계', '비만2단계']
        )
        
        # 상호작용 특성
        df['baseline_risk'] = (
            (df['체질량지수_baseline'] >= 25).astype(int) +
            (df['SBP_baseline'] >= 130).astype(int) +
            (df['DBP_baseline'] >= 85).astype(int) +
            (df['TG_baseline'] >= 150).astype(int)
        )
        
        print(f"✅ 추가된 특성: {len(df.columns) - len(self.diet_vars) - len(self.health_vars)}개")
        
        return df
    
    def preprocess(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        전체 전처리 파이프라인 실행
        
        Returns
        -------
        processed_df : DataFrame
            전처리된 데이터
        info : Dict
            전처리 정보 및 통계
        """
        # 1. 데이터 로드
        df = self.load_data(file_path)
        
        # 2. 방문 패턴 분석
        visit_counts = self.analyze_visit_patterns(df)
        
        # 3. Paired visits 생성
        paired_df = self.create_paired_visits(df)
        
        # 4. MetS 진단
        paired_df = self.calculate_mets(paired_df)
        
        # 5. 고급 특성 엔지니어링
        paired_df = self.create_advanced_features(paired_df)
        
        # 정보 저장
        info = {
            'n_participants': df.index.nunique(),
            'n_total_visits': len(df),
            'n_paired_visits': len(paired_df),
            'n_features': len(paired_df.columns),
            'visit_counts': visit_counts,
            'mets_baseline_prevalence': paired_df['mets_diagnosis_baseline'].mean(),
            'mets_followup_prevalence': paired_df['mets_diagnosis_followup'].mean(),
        }
        
        print("\n" + "=" * 80)
        print("✅ 전처리 완료")
        print("=" * 80)
        print(f"   최종 샘플 수: {len(paired_df):,}개")
        print(f"   최종 특성 수: {len(paired_df.columns)}개")
        
        return paired_df, info


if __name__ == "__main__":
    # 실행 예제
    preprocessor = PairedVisitPreprocessor(
        min_time_gap=90,   # 최소 3개월
        max_time_gap=365   # 최대 1년
    )
    
    processed_df, info = preprocessor.preprocess('../data/total_again.xlsx')
    
    # 결과 저장
    processed_df.to_csv('../data/ver3_paired_data.csv', index=False)
    print(f"\n💾 저장 완료: ver3_paired_data.csv ({len(processed_df):,} rows × {len(processed_df.columns)} cols)")
