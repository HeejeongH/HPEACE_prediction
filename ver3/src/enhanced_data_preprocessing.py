"""
Ver3 Enhanced: 대대적 개선 - 모든 변수 활용 데이터 전처리
====================================================================

개선 사항:
1. HDL, GLUCOSE, HbA1c 등 누락된 건강지표 추가
2. 질병력 (고혈압, 당뇨, 고지혈증 등) 추가
3. 투약 정보 추가
4. 생활습관 (흡연, 음주, 활동량) 추가
5. 복합 위험 점수 생성

저자: SNUH Prediction Team
날짜: 2026-01-03
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class EnhancedPairedVisitPreprocessor:
    """개선된 Paired Visit 전처리 클래스"""
    
    def __init__(self, min_time_gap: int = 90, max_time_gap: int = 365):
        """
        Parameters
        ----------
        min_time_gap : int
            최소 방문 간격 (일)
        max_time_gap : int
            최대 방문 간격 (일)
        """
        self.min_time_gap = min_time_gap
        self.max_time_gap = max_time_gap
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """데이터 로드"""
        print("\n" + "="*80)
        print("📂 Ver3 Enhanced: 데이터 로드")
        print("="*80)
        
        df = pd.read_excel(file_path)
        
        print(f"\n✅ 데이터 로드 완료:")
        print(f"   - 총 레코드: {len(df):,}건")
        print(f"   - 참여자 수: {df['R-ID'].nunique():,}명")
        print(f"   - 변수 수: {len(df.columns):,}개")
        
        # 컬럼 확인
        print(f"\n📋 주요 변수 확인:")
        key_vars = ['HDL CHOL.', 'GLUCOSE', 'HBA1C', 'LDL CHOL.', 'eGFR',
                   '고혈압_통합', '당뇨_통합', '고지혈증_통합',
                   '고혈압_투약여부', '당뇨_투약여부', '고지혈증_투약여부',
                   '일반담배_흡연여부', '음주', '활동량']
        
        for var in key_vars:
            if var in df.columns:
                print(f"   ✅ {var}")
            else:
                print(f"   ⚠️  {var} (없음)")
        
        return df
    
    def create_paired_visits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Paired visits 생성"""
        print("\n" + "="*80)
        print("🔗 Paired Visits 생성")
        print("="*80)
        
        # 날짜 변환
        df['수진일'] = pd.to_datetime(df['수진일'])
        df = df.sort_values(['R-ID', '수진일'])
        
        paired_data = []
        
        for participant_id, group in df.groupby('R-ID'):
            visits = group.sort_values('수진일').reset_index(drop=True)
            
            if len(visits) < 2:
                continue
            
            # 모든 가능한 방문 쌍 생성
            for i in range(len(visits) - 1):
                for j in range(i + 1, len(visits)):
                    baseline = visits.iloc[i]
                    followup = visits.iloc[j]
                    
                    time_gap = (followup['수진일'] - baseline['수진일']).days
                    
                    if self.min_time_gap <= time_gap <= self.max_time_gap:
                        pair = self._create_pair_features(baseline, followup, 
                                                          participant_id, time_gap)
                        if pair is not None:
                            paired_data.append(pair)
        
        paired_df = pd.DataFrame(paired_data)
        
        print(f"\n✅ Paired visits 생성 완료:")
        print(f"   - 총 방문 쌍: {len(paired_df):,}개")
        print(f"   - 특성 수: {len(paired_df.columns):,}개")
        
        info = {
            'n_paired_visits': len(paired_df),
            'n_features': len(paired_df.columns)
        }
        
        return paired_df, info
    
    def _create_pair_features(self, baseline: pd.Series, followup: pd.Series,
                             participant_id: str, time_gap: int) -> Dict:
        """Paired 특성 생성"""
        
        pair = {
            'participant_id': participant_id,
            'time_gap_days': time_gap
        }
        
        # 인구통계학적 변수
        pair['sex'] = baseline.get('성별', np.nan)
        pair['age_baseline'] = baseline.get('나이', np.nan)
        pair['height'] = baseline.get('신장', np.nan)
        
        # ==========================================
        # 1. 식습관 변수 (기존)
        # ==========================================
        diet_vars = ['간식빈도', '고지방 육류', '단맛', '단백질류', '곡류', '과일',
                    '유제품', '음료류', '인스턴트 가공식품', '짠 간', '짠 식습관',
                    '채소', '튀김', '물', '밥 양', '식사 빈도', '식사량', 
                    '외식빈도', '커피']
        
        for var in diet_vars:
            if var in baseline.index:
                baseline_val = baseline.get(var, np.nan)
                followup_val = followup.get(var, np.nan)
                
                pair[f'{var}_baseline'] = baseline_val
                
                if pd.notna(baseline_val) and pd.notna(followup_val):
                    change = followup_val - baseline_val
                    pair[f'{var}_change'] = change
                    
                    if baseline_val != 0:
                        pair[f'{var}_change_pct'] = (change / baseline_val) * 100
                    else:
                        pair[f'{var}_change_pct'] = 0
                else:
                    pair[f'{var}_change'] = np.nan
                    pair[f'{var}_change_pct'] = np.nan
        
        # ==========================================
        # 2. 건강지표 (확장!) - HDL, GLUCOSE, HbA1c 추가
        # ==========================================
        health_vars = ['체중', '체질량지수', '허리둘레(WAIST)', 
                      'SBP', 'DBP', 'TG',
                      'HDL CHOL.', 'GLUCOSE', 'HBA1C',  # 추가!
                      'LDL CHOL.', 'CHOL.', 'eGFR', 'nonHDLC']  # 추가!
        
        for var in health_vars:
            if var in baseline.index:
                baseline_val = baseline.get(var, np.nan)
                followup_val = followup.get(var, np.nan)
                
                # 컬럼명 정리 (공백 제거)
                clean_var = var.replace(' ', '_').replace('.', '')
                
                pair[f'{clean_var}_baseline'] = baseline_val
                pair[f'{clean_var}_followup'] = followup_val
                
                if pd.notna(baseline_val) and pd.notna(followup_val):
                    change = followup_val - baseline_val
                    pair[f'{clean_var}_change'] = change
                    
                    if baseline_val != 0:
                        pair[f'{clean_var}_change_pct'] = (change / baseline_val) * 100
                    else:
                        pair[f'{clean_var}_change_pct'] = 0
                else:
                    pair[f'{clean_var}_change'] = np.nan
                    pair[f'{clean_var}_change_pct'] = np.nan
        
        # ==========================================
        # 3. 질병력 (새로 추가!)
        # ==========================================
        disease_vars = ['고혈압_통합', '당뇨_통합', '고지혈증_통합',
                       '협심증/심근경색증_통합', '뇌졸중(중풍)_통합']
        
        for var in disease_vars:
            if var in baseline.index:
                pair[f'{var}_baseline'] = baseline.get(var, np.nan)
                pair[f'{var}_followup'] = followup.get(var, np.nan)
        
        # ==========================================
        # 4. 투약 정보 (새로 추가!)
        # ==========================================
        medication_vars = ['고혈압_투약여부', '당뇨_투약여부', '고지혈증_투약여부']
        
        for var in medication_vars:
            if var in baseline.index:
                pair[f'{var}_baseline'] = baseline.get(var, np.nan)
                pair[f'{var}_followup'] = followup.get(var, np.nan)
        
        # ==========================================
        # 5. 생활습관 (새로 추가!)
        # ==========================================
        lifestyle_vars = ['일반담배_흡연여부', '음주', '활동량']
        
        for var in lifestyle_vars:
            if var in baseline.index:
                baseline_val = baseline.get(var, np.nan)
                followup_val = followup.get(var, np.nan)
                
                pair[f'{var}_baseline'] = baseline_val
                pair[f'{var}_followup'] = followup_val
                
                # 변화 (범주형이므로 변화 여부만)
                if pd.notna(baseline_val) and pd.notna(followup_val):
                    pair[f'{var}_changed'] = int(baseline_val != followup_val)
                else:
                    pair[f'{var}_changed'] = np.nan
        
        return pair
    
    def calculate_mets_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """개선된 MetS 진단 (HDL, GLUCOSE 포함!)"""
        print("\n" + "="*80)
        print("🏥 MetS 진단 (Enhanced)")
        print("="*80)
        
        # MetS 기준 (Korean criteria)
        # 1. 허리둘레: M >= 90cm, F >= 85cm
        # 2. TG >= 150 mg/dL
        # 3. HDL: M < 40, F < 50 mg/dL  ← 이제 가능!
        # 4. SBP >= 130 or DBP >= 85 mmHg
        # 5. Glucose >= 100 mg/dL  ← 이제 가능!
        
        # Baseline MetS
        df['mets_waist_baseline'] = 0
        if 'sex' in df.columns and '허리둘레WAIST_baseline' in df.columns:
            male_mask = df['sex'] == 'M'
            female_mask = df['sex'] == 'F'
            df.loc[male_mask, 'mets_waist_baseline'] = (
                df.loc[male_mask, '허리둘레WAIST_baseline'] >= 90
            ).astype(int)
            df.loc[female_mask, 'mets_waist_baseline'] = (
                df.loc[female_mask, '허리둘레WAIST_baseline'] >= 85
            ).astype(int)
        
        # TG
        if 'TG_baseline' in df.columns:
            df['mets_tg_baseline'] = (df['TG_baseline'] >= 150).astype(int)
        
        # HDL (새로 추가!)
        df['mets_hdl_baseline'] = 0
        if 'sex' in df.columns and 'HDL_CHOL_baseline' in df.columns:
            male_mask = df['sex'] == 'M'
            female_mask = df['sex'] == 'F'
            df.loc[male_mask, 'mets_hdl_baseline'] = (
                df.loc[male_mask, 'HDL_CHOL_baseline'] < 40
            ).astype(int)
            df.loc[female_mask, 'mets_hdl_baseline'] = (
                df.loc[female_mask, 'HDL_CHOL_baseline'] < 50
            ).astype(int)
        
        # BP
        if 'SBP_baseline' in df.columns and 'DBP_baseline' in df.columns:
            df['mets_bp_baseline'] = (
                (df['SBP_baseline'] >= 130) | (df['DBP_baseline'] >= 85)
            ).astype(int)
        
        # Glucose (새로 추가!)
        if 'GLUCOSE_baseline' in df.columns:
            df['mets_glucose_baseline'] = (df['GLUCOSE_baseline'] >= 100).astype(int)
        
        # MetS count & diagnosis (baseline)
        mets_cols_baseline = [col for col in df.columns if col.startswith('mets_') 
                             and col.endswith('_baseline') and 'count' not in col 
                             and 'diagnosis' not in col]
        
        df['mets_count_baseline'] = df[mets_cols_baseline].sum(axis=1)
        df['mets_diagnosis_baseline'] = (df['mets_count_baseline'] >= 3).astype(int)
        
        # Follow-up MetS (동일 로직)
        df['mets_waist_followup'] = 0
        if 'sex' in df.columns and '허리둘레WAIST_followup' in df.columns:
            male_mask = df['sex'] == 'M'
            female_mask = df['sex'] == 'F'
            df.loc[male_mask, 'mets_waist_followup'] = (
                df.loc[male_mask, '허리둘레WAIST_followup'] >= 90
            ).astype(int)
            df.loc[female_mask, 'mets_waist_followup'] = (
                df.loc[female_mask, '허리둘레WAIST_followup'] >= 85
            ).astype(int)
        
        if 'TG_followup' in df.columns:
            df['mets_tg_followup'] = (df['TG_followup'] >= 150).astype(int)
        
        df['mets_hdl_followup'] = 0
        if 'sex' in df.columns and 'HDL_CHOL_followup' in df.columns:
            male_mask = df['sex'] == 'M'
            female_mask = df['sex'] == 'F'
            df.loc[male_mask, 'mets_hdl_followup'] = (
                df.loc[male_mask, 'HDL_CHOL_followup'] < 40
            ).astype(int)
            df.loc[female_mask, 'mets_hdl_followup'] = (
                df.loc[female_mask, 'HDL_CHOL_followup'] < 50
            ).astype(int)
        
        if 'SBP_followup' in df.columns and 'DBP_followup' in df.columns:
            df['mets_bp_followup'] = (
                (df['SBP_followup'] >= 130) | (df['DBP_followup'] >= 85)
            ).astype(int)
        
        if 'GLUCOSE_followup' in df.columns:
            df['mets_glucose_followup'] = (df['GLUCOSE_followup'] >= 100).astype(int)
        
        mets_cols_followup = [col for col in df.columns if col.startswith('mets_') 
                             and col.endswith('_followup') and 'count' not in col 
                             and 'diagnosis' not in col]
        
        df['mets_count_followup'] = df[mets_cols_followup].sum(axis=1)
        df['mets_diagnosis_followup'] = (df['mets_count_followup'] >= 3).astype(int)
        
        # MetS transition
        conditions = [
            (df['mets_diagnosis_baseline'] == 0) & (df['mets_diagnosis_followup'] == 0),
            (df['mets_diagnosis_baseline'] == 0) & (df['mets_diagnosis_followup'] == 1),
            (df['mets_diagnosis_baseline'] == 1) & (df['mets_diagnosis_followup'] == 0),
            (df['mets_diagnosis_baseline'] == 1) & (df['mets_diagnosis_followup'] == 1)
        ]
        choices = ['stable_no_mets', 'new_onset', 'remission', 'persistent']
        df['mets_transition'] = np.select(conditions, choices, default='unknown')
        
        print(f"\n✅ MetS 진단 완료:")
        print(f"   - Baseline MetS 있음: {df['mets_diagnosis_baseline'].sum():,}명 "
              f"({df['mets_diagnosis_baseline'].mean()*100:.1f}%)")
        print(f"   - Follow-up MetS 있음: {df['mets_diagnosis_followup'].sum():,}명 "
              f"({df['mets_diagnosis_followup'].mean()*100:.1f}%)")
        
        print(f"\n   MetS 변화 패턴:")
        for pattern in ['stable_no_mets', 'new_onset', 'remission', 'persistent']:
            count = (df['mets_transition'] == pattern).sum()
            pct = count / len(df) * 100
            print(f"   - {pattern}: {count:,}명 ({pct:.1f}%)")
        
        return df
    
    def create_advanced_features_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """개선된 고급 특성 엔지니어링"""
        print("\n" + "="*80)
        print("⚙️  고급 특성 엔지니어링 (Enhanced)")
        print("="*80)
        
        n_features_before = len(df.columns)
        
        # ==========================================
        # 1. 질병 위험 점수 (새로 추가!)
        # ==========================================
        print("\n1. 질병 위험 점수 생성...")
        
        # Cardiovascular disease risk score
        cv_risk_components = []
        
        if '고혈압_통합_baseline' in df.columns:
            cv_risk_components.append(df['고혈압_통합_baseline'].fillna(0))
        if '고지혈증_통합_baseline' in df.columns:
            cv_risk_components.append(df['고지혈증_통합_baseline'].fillna(0))
        if '협심증/심근경색증_통합_baseline' in df.columns:
            cv_risk_components.append(df['협심증/심근경색증_통합_baseline'].fillna(0))
        if '뇌졸중(중풍)_통합_baseline' in df.columns:
            cv_risk_components.append(df['뇌졸중(중풍)_통합_baseline'].fillna(0))
        
        if cv_risk_components:
            df['cv_disease_risk_baseline'] = sum(cv_risk_components)
        
        # Diabetes risk score
        diabetes_risk_components = []
        
        if '당뇨_통합_baseline' in df.columns:
            diabetes_risk_components.append(df['당뇨_통합_baseline'].fillna(0) * 2)  # 가중치
        if 'HBA1C_baseline' in df.columns:
            df['hba1c_risk'] = (df['HBA1C_baseline'] >= 5.7).astype(int)  # Prediabetes
            diabetes_risk_components.append(df['hba1c_risk'])
        if 'GLUCOSE_baseline' in df.columns:
            df['glucose_risk'] = (df['GLUCOSE_baseline'] >= 100).astype(int)
            diabetes_risk_components.append(df['glucose_risk'])
        
        if diabetes_risk_components:
            df['diabetes_risk_baseline'] = sum(diabetes_risk_components)
        
        # ==========================================
        # 2. 투약 점수 (치료 강도)
        # ==========================================
        print("2. 투약 점수 생성...")
        
        medication_components = []
        if '고혈압_투약여부_baseline' in df.columns:
            medication_components.append(df['고혈압_투약여부_baseline'].fillna(0))
        if '당뇨_투약여부_baseline' in df.columns:
            medication_components.append(df['당뇨_투약여부_baseline'].fillna(0))
        if '고지혈증_투약여부_baseline' in df.columns:
            medication_components.append(df['고지혈증_투약여부_baseline'].fillna(0))
        
        if medication_components:
            df['medication_count_baseline'] = sum(medication_components)
            df['medication_count_followup'] = 0
            
            # Follow-up 투약
            if '고혈압_투약여부_followup' in df.columns:
                df['medication_count_followup'] += df['고혈압_투약여부_followup'].fillna(0)
            if '당뇨_투약여부_followup' in df.columns:
                df['medication_count_followup'] += df['당뇨_투약여부_followup'].fillna(0)
            if '고지혈증_투약여부_followup' in df.columns:
                df['medication_count_followup'] += df['고지혈증_투약여부_followup'].fillna(0)
            
            df['medication_change'] = (df['medication_count_followup'] - 
                                       df['medication_count_baseline'])
        
        # ==========================================
        # 3. 생활습관 위험 점수
        # ==========================================
        print("3. 생활습관 위험 점수 생성...")
        
        lifestyle_risk = []
        
        # 흡연
        if '일반담배_흡연여부_baseline' in df.columns:
            df['smoking_risk'] = (df['일반담배_흡연여부_baseline'] == 1).astype(int)
            lifestyle_risk.append(df['smoking_risk'])
        
        # 음주 (높은 값 = 위험)
        if '음주_baseline' in df.columns:
            df['alcohol_risk'] = (df['음주_baseline'] >= 3).astype(int)  # 임계값 조정 가능
            lifestyle_risk.append(df['alcohol_risk'])
        
        # 활동량 (낮은 값 = 위험)
        if '활동량_baseline' in df.columns:
            df['low_activity_risk'] = (df['활동량_baseline'] <= 2).astype(int)
            lifestyle_risk.append(df['low_activity_risk'])
        
        if lifestyle_risk:
            df['lifestyle_risk_score'] = sum(lifestyle_risk)
        
        # ==========================================
        # 4. 식습관 점수 (기존 + 확장)
        # ==========================================
        print("4. 식습관 점수 생성...")
        
        # 건강한 식습관
        healthy_foods = ['과일_baseline', '채소_baseline', '유제품_baseline', 
                        '곡류_baseline', '단백질류_baseline', '물_baseline']
        healthy_cols = [col for col in healthy_foods if col in df.columns]
        
        if healthy_cols:
            df['healthy_score_baseline'] = df[healthy_cols].fillna(0).sum(axis=1)
            
            # 변화
            healthy_changes = [col.replace('_baseline', '_change') for col in healthy_cols 
                             if col.replace('_baseline', '_change') in df.columns]
            if healthy_changes:
                df['healthy_score_change'] = df[healthy_changes].fillna(0).sum(axis=1)
        
        # 불건강한 식습관
        unhealthy_foods = ['인스턴트 가공식품_baseline', '튀김_baseline', 
                          '고지방 육류_baseline', '단맛_baseline', '짠 간_baseline',
                          '짠 식습관_baseline', '음료류_baseline', '외식빈도_baseline']
        unhealthy_cols = [col for col in unhealthy_foods if col in df.columns]
        
        if unhealthy_cols:
            df['unhealthy_score_baseline'] = df[unhealthy_cols].fillna(0).sum(axis=1)
            
            unhealthy_changes = [col.replace('_baseline', '_change') for col in unhealthy_cols 
                                if col.replace('_baseline', '_change') in df.columns]
            if unhealthy_changes:
                df['unhealthy_score_change'] = df[unhealthy_changes].fillna(0).sum(axis=1)
        
        # 식습관 개선 점수
        if 'healthy_score_change' in df.columns and 'unhealthy_score_change' in df.columns:
            df['diet_improvement_score'] = (df['healthy_score_change'] - 
                                            df['unhealthy_score_change'])
        
        # ==========================================
        # 5. 복합 위험 점수 (종합)
        # ==========================================
        print("5. 복합 위험 점수 생성...")
        
        risk_components = []
        
        if 'mets_count_baseline' in df.columns:
            risk_components.append(df['mets_count_baseline'] * 2)  # MetS 가중치 높임
        if 'cv_disease_risk_baseline' in df.columns:
            risk_components.append(df['cv_disease_risk_baseline'] * 1.5)
        if 'diabetes_risk_baseline' in df.columns:
            risk_components.append(df['diabetes_risk_baseline'])
        if 'lifestyle_risk_score' in df.columns:
            risk_components.append(df['lifestyle_risk_score'])
        if '체질량지수_baseline' in df.columns:
            df['obesity_risk'] = ((df['체질량지수_baseline'] >= 25).astype(int) +
                                 (df['체질량지수_baseline'] >= 30).astype(int))
            risk_components.append(df['obesity_risk'])
        
        if risk_components:
            df['comprehensive_risk_score'] = sum(risk_components)
        
        # ==========================================
        # 6. 기타 유용한 특성
        # ==========================================
        print("6. 기타 특성 생성...")
        
        # 신장 기능 저하
        if 'eGFR_baseline' in df.columns:
            df['ckd_risk'] = (df['eGFR_baseline'] < 60).astype(int)
        
        # LDL 고위험
        if 'LDL_CHOL_baseline' in df.columns:
            df['ldl_high_risk'] = (df['LDL_CHOL_baseline'] >= 160).astype(int)
        
        # 나이 그룹
        if 'age_baseline' in df.columns:
            df['age_group'] = pd.cut(df['age_baseline'], 
                                     bins=[0, 40, 50, 60, 100],
                                     labels=['<40', '40-50', '50-60', '60+'])
        
        n_features_after = len(df.columns)
        n_added = n_features_after - n_features_before
        
        print(f"\n✅ 고급 특성 생성 완료:")
        print(f"   - 추가된 특성: {n_added}개")
        print(f"   - 최종 특성 수: {n_features_after}개")
        
        return df
    
    def preprocess(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """전체 전처리 파이프라인"""
        
        # 1. 데이터 로드
        df = self.load_data(file_path)
        
        # 2. Paired visits 생성
        paired_df, info = self.create_paired_visits(df)
        
        # 3. MetS 진단
        paired_df = self.calculate_mets_enhanced(paired_df)
        
        # 4. 고급 특성 생성
        paired_df = self.create_advanced_features_enhanced(paired_df)
        
        # 5. 최종 정보
        info['n_features_final'] = len(paired_df.columns)
        info['n_samples_final'] = len(paired_df)
        
        print("\n" + "="*80)
        print("✅ 전처리 완료!")
        print("="*80)
        print(f"   - 최종 샘플 수: {info['n_samples_final']:,}개")
        print(f"   - 최종 특성 수: {info['n_features_final']:,}개")
        
        return paired_df, info


if __name__ == "__main__":
    # 실행 예제
    preprocessor = EnhancedPairedVisitPreprocessor(min_time_gap=90, max_time_gap=365)
    
    df, info = preprocessor.preprocess('../data/total_again.xlsx')
    
    # 저장
    output_path = '../data/ver3_enhanced_paired_data.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장 완료: {output_path}")
