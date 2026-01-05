"""
Optuna Study 로드 문제 해결
"""
import pickle
import os
from pathlib import Path

def load_optuna_results_safe():
    """안전한 Optuna 결과 로드"""
    study_dir = Path('../result/optuna_study')
    
    if not study_dir.exists():
        print("❌ Optuna 결과 디렉토리가 없습니다.")
        return None
    
    study_files = list(study_dir.glob('optuna_study_*.pkl'))
    
    if not study_files:
        print("❌ Optuna 결과 파일이 없습니다.")
        return None
    
    # 가장 최근 파일
    latest_file = max(study_files, key=os.path.getmtime)
    print(f"📂 파일: {latest_file.name}")
    
    try:
        with open(latest_file, 'rb') as f:
            study = pickle.load(f)
        print(f"✅ Optuna 결과 로드 성공")
        print(f"   최고 점수: {study.best_value:.4f}")
        print(f"   최적 파라미터: {study.best_params}")
        return study
    except (TypeError, AttributeError, pickle.UnpicklingError) as e:
        print(f"⚠️  Pickle 로드 실패: {e}")
        print("\n대안: 최적 하이퍼파라미터를 직접 사용하세요.")
        print("\n# 최적 하이퍼파라미터 (Optuna 최고 성능)")
        print("best_params = {")
        print("    'dropout_rate': 0.357,")
        print("    'l1_lambda': 0.000366,")
        print("    'l2_lambda': 0.000596")
        print("}")
        print("\n# 모델 생성")
        print("model = MultiDiseasePredictor(")
        print("    ...,")
        print("    dropout_rate=0.357,")
        print("    l1_lambda=0.000366,")
        print("    l2_lambda=0.000596")
        print(")")
        return None

if __name__ == '__main__':
    load_optuna_results_safe()
