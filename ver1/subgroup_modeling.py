"""
Subgroup-Specific Modeling - 세부 그룹별 모델 학습
===================================================

각 세부 그룹(나이/성별/BMI)별로 전용 모델을 학습하여
더 높은 정확도와 개인맞춤형 예측 제공

Author: Research Team
Date: 2025-11-06
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('./advanced_results/subgroup_models')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SubgroupModeling:
    """세부 그룹별 전용 모델"""
    
    def __init__(self, data_path='./advanced_results/data_with_subgroups.csv'):
        """
        Args:
            data_path: 그룹 정보가 포함된 데이터 경로
        """
        print("\n" + "="*80)
        print("🎯 Subgroup-Specific Modeling 초기화")
        print("="*80)
        
        self.df = pd.read_csv(data_path)
        print(f"\n✅ 데이터 로드: {len(self.df):,}개 샘플")
        
        self.health_indicators = [
            '체중', '체질량지수', '허리둘레(WAIST)', 'SBP', 'DBP', 'TG'
        ]
        
        # 식습관 특성 (19개)
        self.diet_features = [
            '간식빈도', '고지방 육류', '단맛', '단백질류', '곡류',
            '과일', '유제품', '음료류', '인스턴트 가공식품',
            '짠 간', '짠 식습관', '채소', '튀김',
            '식사 빈도', '식사량', '외식빈도',
            '나이', '성별'  # 추가 정보
        ]
        
        # 성별을 숫자로 인코딩
        self.df['성별_encoded'] = self.df['성별'].map({'M': 1, 'F': 0})
        
        print(f"✅ 건강지표: {len(self.health_indicators)}개")
        print(f"✅ 식습관 특성: {len(self.diet_features)}개")
        
    def get_available_features(self):
        """실제 데이터에 있는 특성 확인"""
        available = []
        for feat in self.diet_features:
            if feat == '성별':
                # 성별은 인코딩된 버전 사용
                if '성별_encoded' in self.df.columns:
                    available.append('성별_encoded')
            elif feat in self.df.columns:
                available.append(feat)
        
        print(f"\n사용 가능한 특성: {len(available)}개")
        return available
    
    def train_subgroup_model(self, df_subset, target, group_name):
        """
        특정 그룹의 데이터로 모델 학습
        
        Args:
            df_subset: 해당 그룹 데이터
            target: 예측 타겟 (건강지표)
            group_name: 그룹 이름
        
        Returns:
            dict: 모델 및 성능 정보
        """
        # 특성 준비
        available_features = self.get_available_features()
        X = df_subset[available_features].fillna(df_subset[available_features].median())
        y = df_subset[target].values
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 모델 학습 (RandomForest + GradientBoosting 앙상블)
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        rf.fit(X_train_scaled, y_train)
        gb.fit(X_train_scaled, y_train)
        
        # 앙상블 예측
        y_pred_train = (rf.predict(X_train_scaled) + gb.predict(X_train_scaled)) / 2
        y_pred_test = (rf.predict(X_test_scaled) + gb.predict(X_test_scaled)) / 2
        
        # 성능 평가
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            rf, X_train_scaled, y_train, 
            cv=5, scoring='r2', n_jobs=-1
        )
        
        # Feature importance (RandomForest 기준)
        importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        result = {
            'group': group_name,
            'target': target,
            'n_samples': len(df_subset),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'models': {'rf': rf, 'gb': gb},
            'scaler': scaler,
            'feature_importance': importance,
            'features': available_features
        }
        
        return result
    
    def train_all_subgroups(self):
        """모든 세부 그룹에 대해 모델 학습"""
        print("\n" + "="*80)
        print("🚀 세부 그룹별 모델 학습 시작")
        print("="*80)
        
        all_results = []
        
        # 1. 나이 그룹별
        print("\n[1. 나이 그룹별 모델]")
        for age_group in self.df['나이그룹'].dropna().unique():
            df_age = self.df[self.df['나이그룹'] == age_group]
            
            for target in self.health_indicators:
                if target in df_age.columns:
                    print(f"\n   학습 중: {age_group} - {target}")
                    result = self.train_subgroup_model(
                        df_age, target, f"나이_{age_group}"
                    )
                    all_results.append(result)
                    
                    print(f"      샘플: {result['n_samples']:,}개")
                    print(f"      Test R²: {result['test_r2']:.4f}")
                    print(f"      Test RMSE: {result['test_rmse']:.4f}")
        
        # 2. 성별 그룹별
        print("\n[2. 성별 그룹별 모델]")
        for sex in self.df['성별'].dropna().unique():
            df_sex = self.df[self.df['성별'] == sex]
            
            for target in self.health_indicators:
                if target in df_sex.columns:
                    print(f"\n   학습 중: {sex} - {target}")
                    result = self.train_subgroup_model(
                        df_sex, target, f"성별_{sex}"
                    )
                    all_results.append(result)
                    
                    print(f"      샘플: {result['n_samples']:,}개")
                    print(f"      Test R²: {result['test_r2']:.4f}")
                    print(f"      Test RMSE: {result['test_rmse']:.4f}")
        
        # 3. BMI 그룹별
        print("\n[3. BMI 그룹별 모델]")
        for bmi_group in self.df['BMI그룹'].dropna().unique():
            df_bmi = self.df[self.df['BMI그룹'] == bmi_group]
            
            for target in self.health_indicators:
                if target in df_bmi.columns and target != '체질량지수':
                    print(f"\n   학습 중: {bmi_group} - {target}")
                    result = self.train_subgroup_model(
                        df_bmi, target, f"BMI_{bmi_group}"
                    )
                    all_results.append(result)
                    
                    print(f"      샘플: {result['n_samples']:,}개")
                    print(f"      Test R²: {result['test_r2']:.4f}")
                    print(f"      Test RMSE: {result['test_rmse']:.4f}")
        
        self.all_results = all_results
        
        return all_results
    
    def save_results(self):
        """결과 저장"""
        print("\n" + "="*80)
        print("💾 결과 저장")
        print("="*80)
        
        # 1. 성능 요약 CSV
        summary_data = []
        for result in self.all_results:
            summary_data.append({
                '그룹': result['group'],
                '지표': result['target'],
                '샘플수': result['n_samples'],
                'Train_R²': result['train_r2'],
                'Test_R²': result['test_r2'],
                'CV_R²_mean': result['cv_mean'],
                'CV_R²_std': result['cv_std'],
                'Test_RMSE': result['test_rmse'],
                'Test_MAE': result['test_mae']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = OUTPUT_DIR / 'subgroup_model_performance.csv'
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 성능 요약 저장: {summary_path}")
        
        # 2. 각 모델 저장
        for idx, result in enumerate(self.all_results):
            model_dir = OUTPUT_DIR / f"{result['group']}_{result['target']}"
            model_dir.mkdir(exist_ok=True)
            
            # 모델 파일
            joblib.dump(result['models'], model_dir / 'models.pkl')
            joblib.dump(result['scaler'], model_dir / 'scaler.pkl')
            
            # Feature importance
            result['feature_importance'].to_csv(
                model_dir / 'feature_importance.csv',
                index=False,
                encoding='utf-8-sig'
            )
        
        print(f"✅ 모델 파일 저장: {len(self.all_results)}개")
        
        # 3. Top performers
        summary_df_sorted = summary_df.sort_values('Test_R²', ascending=False)
        print("\n[Top 10 모델 성능]")
        print(summary_df_sorted.head(10)[['그룹', '지표', 'Test_R²', 'Test_RMSE']].to_string(index=False))
        
        return summary_df
    
    def compare_with_overall(self, overall_r2_dict):
        """
        세부 그룹 모델 vs 전체 모델 비교
        
        Args:
            overall_r2_dict: 전체 모델의 R² 딕셔너리
                            예: {'체중': 0.9986, '체질량지수': 0.9988, ...}
        """
        print("\n" + "="*80)
        print("📊 세부 그룹 모델 vs 전체 모델 비교")
        print("="*80)
        
        comparison_data = []
        
        for target in self.health_indicators:
            overall_r2 = overall_r2_dict.get(target, np.nan)
            
            # 해당 지표의 모든 세부 그룹 모델 R²
            subgroup_r2_list = [
                r['test_r2'] for r in self.all_results 
                if r['target'] == target
            ]
            
            if len(subgroup_r2_list) > 0:
                mean_r2 = np.mean(subgroup_r2_list)
                max_r2 = np.max(subgroup_r2_list)
                min_r2 = np.min(subgroup_r2_list)
                
                comparison_data.append({
                    '지표': target,
                    '전체모델_R²': overall_r2,
                    '세부그룹_평균_R²': mean_r2,
                    '세부그룹_최대_R²': max_r2,
                    '세부그룹_최소_R²': min_r2,
                    '평균_개선도': mean_r2 - overall_r2,
                    '최대_개선도': max_r2 - overall_r2
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 저장
        output_path = OUTPUT_DIR / 'comparison_overall_vs_subgroup.csv'
        comparison_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 비교 결과 저장: {output_path}")
        
        # 출력
        print("\n", comparison_df.to_string(index=False))
        
        return comparison_df


def main():
    """메인 실행 함수"""
    print("\n" + "="*80)
    print("🚀 Subgroup-Specific Modeling 시작")
    print("="*80)
    
    # Initialize
    modeler = SubgroupModeling()
    
    # Train all subgroups
    results = modeler.train_all_subgroups()
    
    # Save results
    summary_df = modeler.save_results()
    
    # Compare with overall model (Ver1 결과)
    # 실제 Ver1 R² 값 (README 참고)
    overall_r2 = {
        '체질량지수': 0.9988,
        '체중': 0.9986,
        '허리둘레(WAIST)': 0.9651,
        'DBP': 0.8164,
        'TG': 0.8093,
        'SBP': 0.8068
    }
    
    comparison_df = modeler.compare_with_overall(overall_r2)
    
    print("\n" + "="*80)
    print("✅ Phase 2 완료: 세부 그룹별 모델 학습")
    print("="*80)
    print(f"\n결과 저장 위치: {OUTPUT_DIR.absolute()}")
    print(f"총 {len(results)}개 모델 학습 완료")


if __name__ == '__main__':
    main()
