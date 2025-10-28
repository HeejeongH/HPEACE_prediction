"""
실제 모델 기반 식습관 최적화 시스템
Real Model-based Diet Optimization System
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

class DietOptimizer:
    """바이오마커 개선을 위한 식습관 최적화 클래스"""

    def __init__(self, trained_models, diet_features, feature_structure):
        self.models = trained_models  # 학습된 모델들
        self.diet_features = diet_features  # 식습관 특성 목록
        self.feature_structure = feature_structure  # 모델별 특성 구조
        self.diet_bounds = [(1, 5) for _ in diet_features]  # 식습관 점수 범위 (1-5)

    def predict_biomarker_change(self, current_diet, target_diet, model_info, biomarker_name, other_features):
        """식습관 변화에 따른 바이오마커 변화 예측"""

        # 해당 모델의 특성 구조 가져오기
        required_features = self.feature_structure[biomarker_name]

        # 현재 상태 특성 벡터 생성
        current_features = self._create_feature_vector(current_diet, other_features, required_features)
        current_scaled = model_info['scaler'].transform(model_info['selector'].transform([current_features]))
        current_prediction = self.make_prediction(model_info, current_scaled[0])

        # 목표 상태 특성 벡터 생성
        target_features = self._create_feature_vector(target_diet, other_features, required_features)
        target_scaled = model_info['scaler'].transform(model_info['selector'].transform([target_features]))
        target_prediction = self.make_prediction(model_info, target_scaled[0])

        return target_prediction - current_prediction

    def _create_feature_vector(self, diet_values, other_features, required_features):
        """식습관 값들과 기타 특성으로 전체 특성 벡터 생성"""

        # 기본 식습관 값들
        feature_dict = dict(zip(self.diet_features, diet_values))

        # 기타 특성들 추가 (나이, 성별, 신장 등)
        feature_dict.update(other_features)

        # 특성 공학 적용 (모델 훈련 시와 동일)
        healthy_weights = {'채소': 2.0, '과일': 1.8, '단백질류': 1.5, '곡류': 1.2, '유제품': 1.3}
        unhealthy_weights = {'인스턴트 가공식품': 2.2, '튀김': 2.0, '단맛': 1.8, '고지방 육류': 1.6, '음료류': 1.4}

        # 건강 식품 점수
        healthy_score = sum(feature_dict.get(food, 0) * weight for food, weight in healthy_weights.items())
        unhealthy_score = sum(feature_dict.get(food, 0) * weight for food, weight in unhealthy_weights.items())

        feature_dict['weighted_healthy_score'] = healthy_score
        feature_dict['weighted_unhealthy_score'] = unhealthy_score
        feature_dict['advanced_diet_ratio'] = healthy_score / (unhealthy_score + 1)
        feature_dict['diet_quality_score'] = healthy_score - unhealthy_score

        # 나이-식습관 상호작용
        if '나이' in feature_dict:
            feature_dict['age_healthy_interaction'] = feature_dict['나이'] * healthy_score / 100
            feature_dict['age_unhealthy_interaction'] = feature_dict['나이'] * unhealthy_score / 100

        # 나트륨 위험 점수
        sodium_foods = ['짠 식습관', '짠 간', '인스턴트 가공식품']
        sodium_score = sum(feature_dict.get(food, 0) * weight
                          for food, weight in zip(sodium_foods, [2.0, 1.8, 1.5]))
        feature_dict['sodium_risk_score'] = sodium_score

        # 식습관 다양성
        diet_variety = sum(1 for food in self.diet_features if feature_dict.get(food, 0) > 0)
        feature_dict['diet_variety_count'] = diet_variety

        # 극단 패턴
        if '채소' in feature_dict and '과일' in feature_dict:
            feature_dict['super_healthy_pattern'] = int(feature_dict['채소'] >= 4 and feature_dict['과일'] >= 3)

        if '단맛' in feature_dict and '튀김' in feature_dict:
            feature_dict['junk_food_pattern'] = int(feature_dict['단맛'] >= 4 and feature_dict['튀김'] >= 3)

        # 생활습관 조합
        if '일반담배_흡연여부' in feature_dict:
            feature_dict['smoking_diet_risk'] = feature_dict['일반담배_흡연여부'] * unhealthy_score

        if '활동량' in feature_dict:
            feature_dict['activity_diet_balance'] = feature_dict['활동량'] * feature_dict['advanced_diet_ratio']

        # 모델이 기대하는 특성 순서대로 벡터 생성
        feature_vector = []
        for feature_name in required_features:
            if feature_name in feature_dict:
                feature_vector.append(feature_dict[feature_name])
            else:
                feature_vector.append(0)  # 누락된 특성은 0으로

        return feature_vector

    def make_prediction(self, model_info, scaled_features):
        """모델 타입에 따라 예측 수행"""
        if model_info['type'] == 'ensemble':
            # 앙상블 모델
            predictions = []
            for model in model_info['models']:
                pred = model.predict([scaled_features])[0]
                predictions.append(pred)

            # 가중 평균
            weighted_pred = sum(w * p for w, p in zip(model_info['weights'], predictions))
            return weighted_pred
        else:
            # 단일 모델
            return model_info['model'].predict([scaled_features])[0]

    def optimize_diet_for_biomarker(self, current_diet, target_improvement, biomarker_name, other_features):
        """특정 바이오마커 개선을 위한 최적 식습관 찾기"""

        if biomarker_name not in self.models:
            return {
                'success': False,
                'message': f"{biomarker_name} 모델을 찾을 수 없습니다.",
                'biomarker': biomarker_name
            }

        model_info = self.models[biomarker_name]

        def objective(diet_values):
            """최적화 목적 함수: 목표 개선량과의 차이를 최소화"""
            try:
                predicted_change = self.predict_biomarker_change(
                    current_diet, diet_values, model_info, biomarker_name, other_features
                )
                # 목표와의 차이 + 큰 변화에 대한 페널티
                change_penalty = sum(abs(new - old) for new, old in zip(diet_values, current_diet)) * 0.1
                return abs(predicted_change - target_improvement) + change_penalty
            except Exception as e:
                print(f"   최적화 오류: {str(e)[:50]}")
                return 1e6  # 오류 시 큰 값 반환

        def realistic_constraint(diet_values):
            """현실적인 식습관 변화 제약: 한 번에 1.5점 이상 변화하지 않음"""
            max_change = max(abs(new - old) for new, old in zip(diet_values, current_diet))
            return 1.5 - max_change  # >= 0이어야 함

        # 최적화 실행
        constraints = [{'type': 'ineq', 'fun': realistic_constraint}]

        try:
            # 여러 시작점에서 최적화 시도 (더 좋은 해 찾기)
            best_result = None
            best_objective = float('inf')

            for i in range(3):  # 3번 시도
                # 시작점을 약간씩 다르게 설정
                x0 = np.array(current_diet) + np.random.normal(0, 0.2, len(current_diet))
                x0 = np.clip(x0, 1, 5)  # 범위 내로 클리핑

                result = minimize(
                    objective,
                    x0=x0,
                    bounds=self.diet_bounds,
                    constraints=constraints,
                    method='SLSQP',
                    options={'maxiter': 100}
                )

                if result.success and result.fun < best_objective:
                    best_result = result
                    best_objective = result.fun

            if best_result and best_result.success:
                optimal_diet = best_result.x
                predicted_change = self.predict_biomarker_change(
                    current_diet, optimal_diet, model_info, biomarker_name, other_features
                )

                # 변화가 있는 식습관만 추출
                diet_changes = {}
                for i, (food, old_val, new_val) in enumerate(zip(self.diet_features, current_diet, optimal_diet)):
                    if abs(new_val - old_val) > 0.1:  # 0.1 이상 변화한 것만
                        diet_changes[food] = {
                            'current': old_val,
                            'optimal': new_val,
                            'change': new_val - old_val
                        }

                return {
                    'success': True,
                    'optimal_diet': optimal_diet,
                    'predicted_change': predicted_change,
                    'target_improvement': target_improvement,
                    'diet_changes': diet_changes,
                    'biomarker': biomarker_name,
                    'optimization_score': best_objective
                }
            else:
                return {
                    'success': False,
                    'message': f"최적화 실패: 적절한 해를 찾을 수 없습니다",
                    'biomarker': biomarker_name
                }

        except Exception as e:
            return {
                'success': False,
                'message': f"최적화 오류: {str(e)[:50]}",
                'biomarker': biomarker_name
            }

def run_optimization_example(model_results, diet_features, analysis_df):
    """최적화 예제 실행"""

    if not model_results:
        print("❌ 학습된 모델이 없습니다.")
        return

    # 최적화 시스템 초기화
    trained_models = {}
    feature_structure = {}

    for result in model_results:
        biomarker = result['Biomarker_KR']
        trained_models[biomarker] = result['Model']
        feature_structure[biomarker] = result['Features']

    optimizer = DietOptimizer(
        trained_models=trained_models,
        diet_features=[col for col in diet_features if col in analysis_df.columns],
        feature_structure=feature_structure
    )

    # 가상 환자 프로필
    sample_patient = {
        'demographics': {'나이': 45, '성별': 1, '신장': 170},
        'lifestyle': {'일반담배_흡연여부': 0, '활동량': 2, '음주': 2},
        'current_diet': [3.5, 2.8, 2.2, 1.8, 2.5, 3.8, 3.2, 3.0, 3.5, 4.0, 3.3, 3.8]  # 식습관 점수
    }

    other_features = {**sample_patient['demographics'], **sample_patient['lifestyle']}

    print("\n🎯 실제 모델 기반 식습관 최적화 결과")
    print("="*60)

    # 최적화 가능한 바이오마커들 (R² > 0.3)
    optimizable_biomarkers = [
        {'name': '체중', 'target': -5.0, 'unit': 'kg'},
        {'name': 'SBP', 'target': -10.0, 'unit': 'mmHg'},
        {'name': '허리둘레(WAIST)', 'target': -5.0, 'unit': 'cm'}
    ]

    for biomarker_info in optimizable_biomarkers:
        biomarker = biomarker_info['name']
        target_change = biomarker_info['target']
        unit = biomarker_info['unit']

        # 해당 바이오마커 모델이 있는지 확인
        if biomarker not in trained_models:
            continue

        # 모델 성능 확인
        model_r2 = next((r['R_squared'] for r in model_results if r['Biomarker_KR'] == biomarker), 0)
        if model_r2 < 0.3:
            continue

        print(f"\n📊 {biomarker} 개선 목표: {abs(target_change)}{unit} 감소 (모델 R²={model_r2:.3f})")

        # 최적화 실행
        optimization_result = optimizer.optimize_diet_for_biomarker(
            current_diet=sample_patient['current_diet'],
            target_improvement=target_change,
            biomarker_name=biomarker,
            other_features=other_features
        )

        if optimization_result['success']:
            predicted_change = optimization_result['predicted_change']
            diet_changes = optimization_result['diet_changes']

            print(f"   ✅ 최적화 성공!")
            print(f"   📈 예측 개선량: {predicted_change:.2f}{unit}")
            print(f"   🎯 목표 달성도: {abs(predicted_change/target_change)*100:.1f}%")
            print(f"   🍽️ 권장 식습관 변화:")

            for food, change_info in diet_changes.items():
                change_val = change_info['change']
                direction = "🔼" if change_val > 0 else "🔽"
                print(f"      {direction} {food}: {change_info['current']:.1f} → {change_info['optimal']:.1f} ({change_val:+.1f})")
        else:
            print(f"   ❌ 최적화 실패: {optimization_result['message']}")

    return optimizer

if __name__ == "__main__":
    print("🎯 실제 모델 기반 식습관 최적화 시스템")
    print("   - scipy.optimize 사용한 실제 역산 알고리즘")
    print("   - 학습된 모델을 통한 정확한 예측")
    print("   - 현실적 제약조건 적용")