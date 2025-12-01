"""
Ver2 실행 스크립트
=================

Ver2 (종단 분석) 전체 파이프라인 실행
"""

import os
import sys

# 작업 디렉토리 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, os.path.join(script_dir, 'src'))


def print_menu():
    """메뉴 출력"""
    print("\n" + "="*80)
    print("🔬 Ver2: 종단 분석 (Longitudinal Change Prediction)")
    print("="*80)
    print("\n메뉴:")
    print("  1. 데이터 전처리 (Paired Visits + 분류 타겟 생성)")
    print("  2. TabNet 모델 학습 (Attention-based)")
    print("  3. XGBoost 모델 학습 (Baseline)")
    print("  4. LSTM 모델 학습 (Deep Learning)")
    print("  5. 🚀 앙상블 분류 모델 학습 (RF + XGB + LGB) ⭐ 추천")
    print("  6. 전체 실행 (회귀 모델: 1→2→3→4)")
    print("  7. 결과 비교 (TabNet vs XGBoost vs LSTM)")
    print("  8. 전체 실행 (분류 모델: 1→5) 🚀 최고 성능")
    print("  0. 종료")
    print("="*80)


def step1_preprocessing():
    """Step 1: 데이터 전처리"""
    print("\n" + "="*80)
    print("📂 Step 1: 데이터 전처리 시작")
    print("="*80)
    
    try:
        from data_preprocessing import main as preprocess_main
        preprocess_main()
        print("\n✅ 데이터 전처리 완료!")
        return True
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        return False


def step2_tabnet():
    """Step 2: TabNet 학습"""
    print("\n" + "="*80)
    print("🎯 Step 2: TabNet 모델 학습")
    print("="*80)
    
    # 전처리 데이터 확인 (프로젝트 루트/data 폴더)
    data_path = os.path.join(script_dir, '..', 'data', 'ver2_paired_visits.csv')
    data_path = os.path.abspath(data_path)
    
    if not os.path.exists(data_path):
        print(f"\n⚠️  전처리 데이터가 없습니다: {data_path}")
        print("먼저 '1. 데이터 전처리'를 실행하세요.")
        return False
    
    print(f"📂 데이터 로드: {data_path}")
    
    try:
        from tabnet_model import train_all_targets
        results = train_all_targets(data_path)
        print("\n✅ TabNet 학습 완료!")
        return True
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def step3_xgboost():
    """Step 3: XGBoost 학습"""
    print("\n" + "="*80)
    print("🎯 Step 3: XGBoost 모델 학습")
    print("="*80)
    
    # 전처리 데이터 확인 (프로젝트 루트/data 폴더)
    data_path = os.path.join(script_dir, '..', 'data', 'ver2_paired_visits.csv')
    data_path = os.path.abspath(data_path)
    
    if not os.path.exists(data_path):
        print(f"\n⚠️  전처리 데이터가 없습니다: {data_path}")
        print("먼저 '1. 데이터 전처리'를 실행하세요.")
        return False
    
    print(f"📂 데이터 로드: {data_path}")
    
    try:
        from xgboost_model import train_all_targets
        results = train_all_targets(data_path)
        print("\n✅ XGBoost 학습 완료!")
        return True
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def step4_lstm():
    """Step 4: LSTM 학습"""
    print("\n" + "="*80)
    print("🎯 Step 4: LSTM 모델 학습")
    print("="*80)
    
    # 전처리 데이터 확인 (프로젝트 루트/data 폴더)
    data_path = os.path.join(script_dir, '..', 'data', 'ver2_paired_visits.csv')
    data_path = os.path.abspath(data_path)
    
    if not os.path.exists(data_path):
        print(f"\n⚠️  전처리 데이터가 없습니다: {data_path}")
        print("먼저 '1. 데이터 전처리'를 실행하세요.")
        return False
    
    print(f"📂 데이터 로드: {data_path}")
    
    try:
        from lstm_model import train_all_targets
        results = train_all_targets(data_path)
        print("\n✅ LSTM 학습 완료!")
        return True
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def step5_ensemble_classifier():
    """Step 5: 앙상블 분류 모델 학습 (최고 성능!)"""
    print("\n" + "="*80)
    print("🎯 Step 5: 앙상블 분류 모델 학습")
    print("="*80)
    
    # 전처리 데이터 확인
    data_path = os.path.join(script_dir, '..', 'data', 'ver2_paired_visits.csv')
    data_path = os.path.abspath(data_path)
    
    if not os.path.exists(data_path):
        print(f"\n⚠️  전처리 데이터가 없습니다: {data_path}")
        print("먼저 '1. 데이터 전처리'를 실행하세요.")
        return False
    
    print(f"📂 데이터 로드: {data_path}")
    
    try:
        from ensemble_classifier import train_all_targets
        results = train_all_targets(data_path)
        print("\n✅ 앙상블 분류 모델 학습 완료!")
        return True
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def step6_full_pipeline():
    """Step 6: 전체 파이프라인 (회귀 모델)"""
    print("\n" + "="*80)
    print("🚀 전체 파이프라인 실행 (회귀 모델)")
    print("="*80)
    
    # Step 1: 전처리
    if not step1_preprocessing():
        return False
    
    # Step 2: TabNet
    if not step2_tabnet():
        return False
    
    # Step 3: XGBoost
    if not step3_xgboost():
        return False
    
    # Step 4: LSTM
    if not step4_lstm():
        return False
    
    print("\n" + "="*80)
    print("✅ 전체 파이프라인 완료!")
    print("="*80)
    return True


def step8_classification_pipeline():
    """Step 8: 분류 모델 파이프라인 (최고 성능!)"""
    print("\n" + "="*80)
    print("🚀 분류 모델 파이프라인 실행 (최고 성능 목표!)")
    print("="*80)
    
    # Step 1: 전처리
    if not step1_preprocessing():
        return False
    
    # Step 5: 앙상블 분류 모델
    if not step5_ensemble_classifier():
        return False
    
    print("\n" + "="*80)
    print("✅ 분류 모델 파이프라인 완료!")
    print("="*80)
    return True


def step7_compare_results():
    """Step 7: 결과 비교"""
    print("\n" + "="*80)
    print("📊 TabNet vs XGBoost vs LSTM 결과 비교")
    print("="*80)
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 결과 파일 확인
    tabnet_path = '../result/tabnet_all_results.csv'
    xgb_path = '../result/xgboost_all_results.csv'
    lstm_path = '../result/lstm_all_results.csv'
    
    available_models = {}
    
    if os.path.exists(tabnet_path):
        available_models['TabNet'] = pd.read_csv(tabnet_path, index_col=0)
    
    if os.path.exists(xgb_path):
        available_models['XGBoost'] = pd.read_csv(xgb_path, index_col=0)
    
    if os.path.exists(lstm_path):
        available_models['LSTM'] = pd.read_csv(lstm_path, index_col=0)
    
    if len(available_models) == 0:
        print("⚠️  학습 결과가 없습니다. 먼저 모델을 학습하세요.")
        return False
    
    if len(available_models) == 1:
        print(f"⚠️  모델이 1개만 있습니다. 비교를 위해 2개 이상 학습하세요.")
        model_name = list(available_models.keys())[0]
        print(f"\n📊 {model_name} 결과:")
        print(available_models[model_name].round(4))
        return False
    
    # 결과 출력
    for model_name, results in available_models.items():
        print(f"\n📊 {model_name} 결과:")
        print(results.round(4))
    
    # 비교 시각화
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    metrics = ['R²', 'RMSE', 'MAE', 'Direction_Accuracy']
    model_names = list(available_models.keys())
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        indicators = available_models[model_names[0]].index
        x = range(len(indicators))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            values = available_models[model_name][metric].values
            offset = (i - len(model_names)/2 + 0.5) * width
            ax.bar([j + offset for j in x], values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('건강지표', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} 비교', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(indicators, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = '../result/model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 비교 그래프 저장: {output_path}")
    plt.close()
    
    # 모델 간 평균 성능
    print("\n" + "="*80)
    print("📈 모델별 평균 성능")
    print("="*80)
    
    for model_name, results in available_models.items():
        print(f"\n{model_name}:")
        for metric in metrics:
            avg_value = results[metric].mean()
            print(f"  평균 {metric}: {avg_value:.4f}")
    
    # 지표별 최고 모델
    print("\n" + "="*80)
    print("🏆 지표별 최고 모델")
    print("="*80)
    
    for indicator in indicators:
        print(f"\n{indicator}:")
        for metric in metrics:
            best_model = None
            best_value = None
            
            for model_name, results in available_models.items():
                value = results.loc[indicator, metric]
                
                # R²와 Direction_Accuracy는 높을수록 좋음
                if metric in ['R²', 'Direction_Accuracy']:
                    if best_value is None or value > best_value:
                        best_value = value
                        best_model = model_name
                # RMSE, MAE는 낮을수록 좋음
                else:
                    if best_value is None or value < best_value:
                        best_value = value
                        best_model = model_name
            
            print(f"  {metric}: {best_model} ({best_value:.4f})")
    
    return True


def main():
    """메인 실행"""
    while True:
        print_menu()
        
        try:
            choice = input("\n선택 (0-8): ").strip()
            
            if choice == '0':
                print("\n👋 프로그램을 종료합니다.")
                break
            
            elif choice == '1':
                step1_preprocessing()
            
            elif choice == '2':
                step2_tabnet()
            
            elif choice == '3':
                step3_xgboost()
            
            elif choice == '4':
                step4_lstm()
            
            elif choice == '5':
                step5_ensemble_classifier()
            
            elif choice == '6':
                step6_full_pipeline()
            
            elif choice == '7':
                step7_compare_results()
            
            elif choice == '8':
                step8_classification_pipeline()
            
            else:
                print("\n⚠️  잘못된 선택입니다. 0-8 사이의 숫자를 입력하세요.")
            
            input("\n▶️  Enter를 눌러 계속...")
        
        except KeyboardInterrupt:
            print("\n\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            input("\n▶️  Enter를 눌러 계속...")


if __name__ == '__main__':
    main()
