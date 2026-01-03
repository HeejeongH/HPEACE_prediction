import torch
import numpy as np
from scipy.optimize import minimize, differential_evolution
from utils import demo_cols, life_cols, bio_cols, diet_cols, interaction_cols, mets_cols
import os

class DietaryRecommendationSystem:
    def __init__(self, trained_model, device, feature_scaler=None):
        self.model = trained_model
        self.device = device
        self.scaler = feature_scaler
        self.model.eval()
        
        self.diet_names = [
            '간식빈도', '고지방 육류', '곡류', '과일', '단맛', '단백질류', 
            '물', '밥 양', '식사 빈도', '식사량', '외식빈도', '유제품', 
            '음료류', '인스턴트 가공식품', '짠 간', '짠 식습관', '채소', 
            '커피', '튀김'
        ]
        
        # 동적으로 차원 계산
        sample_input = torch.zeros(1, 75).to(device)
        try:
            test_disease = mets_cols[0]
            with torch.no_grad():
                _ = self.model(
                    sample_input[:, :19], sample_input[:, 19:23], sample_input[:, 23:26],
                    sample_input[:, 26:37], sample_input[:, 37:59], sample_input[:, 59:65],
                    sample_input[:, 65:75], test_disease
                )
            self.dims = {
                'diet': 19, 'demo': 4, 'life': 3, 'bio': 11,
                'delta': 22, 'interaction': 6, 'pca': 10
            }
        except:
            print("모델 차원 자동 감지 실패, 기본값 사용")
            self.dims = {
                'diet': 19, 'demo': 4, 'life': 3, 'bio': 11,
                'delta': 22, 'interaction': 6, 'pca': 10
            }
        
        self.diet_indices = list(range(self.dims['diet']))        
        print(f"추천 시스템 초기화 완료 - 총 피처 차원: {sum(self.dims.values())}")
    
    def predict_disease_probabilities(self, features, disease_name):
        """질병 확률 예측"""
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            start_idx = 0
            diet_data = features_tensor[:, start_idx:start_idx+self.dims['diet']]
            start_idx += self.dims['diet']
            
            demo_data = features_tensor[:, start_idx:start_idx+self.dims['demo']]
            start_idx += self.dims['demo']
            
            life_data = features_tensor[:, start_idx:start_idx+self.dims['life']]
            start_idx += self.dims['life']
            
            bio_data = features_tensor[:, start_idx:start_idx+self.dims['bio']]
            start_idx += self.dims['bio']
            
            delta_data = features_tensor[:, start_idx:start_idx+self.dims['delta']]
            start_idx += self.dims['delta']
            
            interaction_data = features_tensor[:, start_idx:start_idx+self.dims['interaction']]
            start_idx += self.dims['interaction']
            
            pca_data = features_tensor[:, start_idx:start_idx+self.dims['pca']]
            
            output = self.model(diet_data, demo_data, life_data, bio_data, 
                              delta_data, interaction_data, pca_data, disease_name)
            probabilities = torch.softmax(output['disease_logits'], dim=1)
            
            return probabilities.cpu().numpy()[0]
    
    def _manual_search(self, patient_features, targets, max_change):
        """수동 그리드 탐색 - 최후 수단"""
        print("수동 탐색으로 최적 조합 찾는 중...")
        
        best_changes = np.zeros(self.dims['diet'])
        best_score = self.objective_function(best_changes, patient_features, targets)
        
        # 각 식품별로 순차 탐색
        search_values = [-1.0, -0.5, 0.5, 1.0]
        
        for i in range(min(10, self.dims['diet'])):  # 상위 10개 식품만
            for value in search_values:
                test_changes = best_changes.copy()
                test_changes[i] = value
                
                score = self.objective_function(test_changes, patient_features, targets)
                
                if score < best_score:
                    best_score = score
                    best_changes = test_changes.copy()
                    print(f"{self.diet_names[i]} {value:+.1f} 적용: 점수 {score:.4f}")
        
        # 조합 최적화 (상위 3개 식품)
        top_foods = np.argsort(np.abs(best_changes))[-3:]
        
        for combo in [(-0.5, 0.5, 0), (0.5, -0.5, 0), (1.0, -1.0, 0)]:
            test_changes = best_changes.copy()
            for idx, food_idx in enumerate(top_foods):
                if idx < len(combo):
                    test_changes[food_idx] = combo[idx]
            
            score = self.objective_function(test_changes, patient_features, targets)
            if score < best_score:
                best_score = score
                best_changes = test_changes
        
        print(f"수동 탐색 완료: 최종 점수 {best_score:.4f}")
        
        if best_score < 0:  # 개선이 있었다면
            return self._format_recommendations(best_changes, patient_features, targets)
        else:
            return {"error": "개선 방안을 찾을 수 없습니다", "tried_manual_search": True}

    def objective_function(self, diet_changes, base_features, targets, weights=None):
        """더 안정적인 목적함수"""
        try:
            modified_features = base_features.copy()
            modified_features[self.diet_indices] += diet_changes
            
            # 값 범위 클리핑
            modified_features[self.diet_indices] = np.clip(
                modified_features[self.diet_indices], -3, 3
            )
            
            total_score = 0
            
            for disease, target_direction in targets.items():
                try:
                    current_probs = self.predict_disease_probabilities(base_features, disease)
                    new_probs = self.predict_disease_probabilities(modified_features, disease)
                    
                    if target_direction == 'improve':
                        # 더 안정적인 점수 계산
                        improvement_gain = new_probs[0] - current_probs[0]
                        score = improvement_gain * 50  # 적당한 증폭
                        
                        # 절대값 보너스
                        if new_probs[0] > 0.5:
                            score += (new_probs[0] - 0.5) * 20
                            
                    elif target_direction == 'maintain':
                        maintain_gain = new_probs[1] - current_probs[1] 
                        score = maintain_gain * 30
                        
                    elif target_direction == 'prevent_worsening':
                        worsen_reduction = current_probs[2] - new_probs[2]
                        score = worsen_reduction * 40
                    else:
                        score = 0
                    
                    # NaN 체크
                    if np.isnan(score) or np.isinf(score):
                        score = 0
                        
                    weight = weights.get(disease, 1.0) if weights else 1.0
                    total_score += weight * score
                    
                except Exception as e:
                    print(f"질병 {disease} 예측 오류: {e}")
                    total_score += -10  # 패널티
            
            # 변화량 페널티
            change_penalty = np.sum(np.abs(diet_changes)) * 0.01
            
            final_score = -(total_score - change_penalty)
            
            # NaN/inf 체크
            if np.isnan(final_score) or np.isinf(final_score):
                return 1000
                
            return final_score
            
        except Exception as e:
            print(f"목적함수 전체 오류: {e}")
            return 1000
    
    def generate_recommendations(self, patient_features, targets, max_change=2.0, method='optimization'):
        """식습관 추천 생성"""
        if len(patient_features) != sum(self.dims.values()):
            print(f"피처 차원 불일치: 예상 {sum(self.dims.values())}, 실제 {len(patient_features)}")
            return {"error": "피처 차원 불일치"}
        
        if method == 'optimization':
            return self._scipy_optimization(patient_features, targets, max_change)
        elif method == 'genetic':
            return self._genetic_algorithm(patient_features, targets, max_change)
    
    def _scipy_optimization(self, patient_features, targets, max_change):
        """더 적극적인 최적화 - 다중 시작점"""
        bounds = [(-max_change, max_change) for _ in range(self.dims['diet'])]
        
        # 여러 시작점에서 최적화 시도
        best_result = None
        best_score = float('inf')
        
        start_points = [
            np.zeros(self.dims['diet']),
            np.random.uniform(-0.5, 0.5, self.dims['diet']),
            np.random.uniform(-1.0, 1.0, self.dims['diet']),
        ]
        
        for x0 in start_points:
            try:
                result = minimize(
                    self.objective_function,
                    x0,
                    args=(patient_features, targets),
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result
                    
            except Exception as e:
                continue
        
        if best_result and best_result.success:
            return self._format_recommendations(best_result.x, patient_features, targets)
        else:
            return {"error": "모든 최적화 시도 실패"}
    
    def _genetic_algorithm(self, patient_features, targets, max_change):
        """유전 알고리즘"""
        bounds = [(-max_change, max_change) for _ in range(self.dims['diet'])]
        
        result = differential_evolution(
            self.objective_function,
            bounds,
            args=(patient_features, targets),
            maxiter=100,
            seed=42,
            popsize=15
        )
        
        if result.success:
            return self._format_recommendations(result.x, patient_features, targets)
        else:
            return {"error": "유전 알고리즘 실패"}
    
    def _format_recommendations(self, diet_changes, patient_features, targets):
        """매우 낮은 임계값으로 모든 변화 포착"""
        
        # 현재와 예상 예측
        current_predictions = {}
        predicted_predictions = {}
        
        for disease in targets.keys():
            current_predictions[disease] = self.predict_disease_probabilities(patient_features, disease)
            
            modified_features = patient_features.copy()
            modified_features[self.diet_indices] += diet_changes
            predicted_predictions[disease] = self.predict_disease_probabilities(modified_features, disease)
        
        # 모든 변화량을 기록하되, 임계값은 0.01로 설정
        all_changes = []
        for i, change in enumerate(diet_changes):
            if abs(change) > 0.01:  # 매우 낮은 임계값
                all_changes.append((i, change, abs(change)))
        
        # 변화량 기준으로 정렬
        all_changes.sort(key=lambda x: x[2], reverse=True)
        
        # 상위 변화들만 추천으로 선택
        recommendations = {}
        for i, change, abs_change in all_changes[:10]:  # 상위 10개
            direction = "증가" if change > 0 else "감소"
            
            if abs_change > 0.5:
                magnitude = "크게"
            elif abs_change > 0.2:
                magnitude = "적당히" 
            else:
                magnitude = "조금"
                
            food_name = self.diet_names[i]
            recommendations[food_name] = {
                'change_amount': round(change, 3),
                'direction': direction,
                'magnitude': magnitude,
                'priority': len(recommendations) + 1,
                'recommendation': f"{food_name} 섭취를 {magnitude} {direction}시키세요"
            }
        
        # 예상 효과 계산
        effects = {}
        for disease in targets.keys():
            current_prob = current_predictions[disease]
            predicted_prob = predicted_predictions[disease]
            
            improvement_change = predicted_prob[0] - current_prob[0]
            maintain_change = predicted_prob[1] - current_prob[1] 
            worsen_change = predicted_prob[2] - current_prob[2]
            
            # 효과 크기 분류
            if improvement_change > 0.05:
                outcome = "상당한 개선 기대"
            elif improvement_change > 0.02:
                outcome = "적당한 개선 기대"
            elif improvement_change > 0.005:
                outcome = "약간의 개선 기대"
            elif worsen_change < -0.01:
                outcome = "악화 방지 효과"
            else:
                outcome = "미미한 변화"
            
            effects[disease] = {
                'current_probs': [round(p, 4) for p in current_prob],
                'predicted_probs': [round(p, 4) for p in predicted_prob],
                'improvement_change': round(improvement_change, 4),
                'maintain_change': round(maintain_change, 4),
                'worsen_change': round(worsen_change, 4),
                'expected_outcome': outcome
            }
        
        return {
            'recommendations': recommendations,
            'predicted_effects': effects,
            'total_changes': len(recommendations),
            'max_improvement': max([effects[d]['improvement_change'] for d in effects.keys()]),
            'feasibility_score': self._calculate_feasibility(diet_changes)
        }
    
    def _calculate_feasibility(self, diet_changes):
        """실행 가능성 점수"""
        significant_changes = sum(1 for change in diet_changes if abs(change) > 0.1)
        avg_change = np.mean(np.abs(diet_changes))
        
        count_penalty = min(significant_changes / 10, 0.4)
        magnitude_penalty = min(avg_change / 2, 0.3)
        
        base_score = 1.0
        final_score = base_score - count_penalty - magnitude_penalty
        
        return max(final_score, 0.1)

def run_recommendation_example(final_results, test_df_global, device):
    """추천 시스템 실행 예제"""
    try:
        final_test = test_df_global.copy()
        
        if not final_results:
            print("final_results가 비어있습니다.")
            return
            
        best_disease = max(final_results.keys(), 
                          key=lambda k: final_results[k].get('f1_score', 0))
        print(f"선택된 질병: {best_disease} (F1: {final_results[best_disease]['f1_score']:.3f})")
        
        # 모델 로딩
        model_path = '../result/final_optimization/final_best_models.pth'
        if not os.path.exists(model_path):
            print("저장된 모델을 찾을 수 없습니다.")
            return
            
        saved_models = torch.load(model_path, map_location=device)
        if best_disease not in saved_models:
            print(f"{best_disease}에 대한 모델을 찾을 수 없습니다.")
            return
            
        trained_model = saved_models[best_disease]
        recommender = DietaryRecommendationSystem(trained_model=trained_model, device=device)
        print("추천 시스템 준비 완료")
        
        # 샘플 환자 선택
        sample_index = 0
        sample_patient = final_test.iloc[sample_index]
        
        # disease delta 컬럼들 제외하고 피처 추출
        disease_delta_cols = [f'{disease}_delta' for disease in mets_cols]
        exclude_cols = [col for col in sample_patient.index if col in disease_delta_cols]
        patient_features = sample_patient.drop(exclude_cols).values
        
        print(f"\n=== {sample_index+1}번째 환자 추천 ===")
        print(f"환자 피처 차원: {len(patient_features)}")
        
        # 개선 목표 설정
        targets = {best_disease: 'improve'}
        
        # 추천 생성 - 더 적극적인 설정
        recommendations = recommender.generate_recommendations(
            patient_features=patient_features,
            targets=targets,
            max_change=3.0,  # 더 큰 변화 허용
            method='optimization'
        )
        
        # 결과 출력
        if 'error' in recommendations:
            print(f"오류: {recommendations['error']}")
            return
            
        print(f"총 {recommendations['total_changes']}개 식습관 변경 권장")
        print(f"최대 개선 기대치: {recommendations.get('max_improvement', 0):+.4f}")
        print(f"실행 가능성: {recommendations['feasibility_score']:.1%}")
        
        if recommendations['recommendations']:
            print("\n우선순위별 추천사항:")
            sorted_recs = sorted(recommendations['recommendations'].items(), 
                               key=lambda x: x[1]['priority'])
            
            for food, rec in sorted_recs[:5]:  # 상위 5개
                print(f"{rec['priority']}. {rec['recommendation']} (변화량: {rec['change_amount']:+.3f})")
        
        # 효과 요약
        for disease, effect in recommendations['predicted_effects'].items():
            print(f"\n{disease} 예상 효과:")
            print(f"  개선 확률: {effect['current_probs'][0]:.4f} → {effect['predicted_probs'][0]:.4f} "
                  f"({effect['improvement_change']:+.4f})")
            print(f"  결과: {effect['expected_outcome']}")
        
        # 다른 목표로도 시도
        print("\n=== 악화 방지 목표 ===")
        other_recommendations = recommender.generate_recommendations(
            patient_features=patient_features,
            targets={best_disease: 'prevent_worsening'},
            max_change=2.5
        )
        
        print(f"추천 개수: {other_recommendations.get('total_changes', 0)}")
        if other_recommendations.get('recommendations'):
            sorted_other = sorted(other_recommendations['recommendations'].items(),
                                key=lambda x: abs(x[1]['change_amount']), reverse=True)
            for food, rec in sorted_other[:3]:
                print(f"  • {rec['recommendation']}")
        
    except Exception as e:
        print(f"추천 시스템 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

# 디버깅 함수
def debug_recommendation_system(recommender, patient_features, disease_name):
    """추천 시스템 디버깅"""
    print("=== 디버깅 정보 ===")
    
    current_probs = recommender.predict_disease_probabilities(patient_features, disease_name)
    print(f"현재 예측: 개선={current_probs[0]:.3f}, 유지={current_probs[1]:.3f}, 악화={current_probs[2]:.3f}")
    
    # 수동으로 작은 변화 테스트
    test_changes = [0.5, -0.5, 1.0, -1.0]
    
    for i, change in enumerate(test_changes):
        if i >= len(recommender.diet_names):
            break
            
        modified_features = patient_features.copy()
        modified_features[i] += change
        
        new_probs = recommender.predict_disease_probabilities(modified_features, disease_name)
        improvement_diff = new_probs[0] - current_probs[0]
        
        print(f"{recommender.diet_names[i]} {change:+.1f} 변화: "
              f"개선확률 {improvement_diff:+.3f} 변화")