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
    print("  1. 데이터 전처리 (Paired Visits 생성)")
    print("  2. XGBoost 모델 학습 (Baseline)")
    print("  3. LSTM 모델 학습 (Advanced)")
    print("  4. 전체 실행 (1→2→3)")
    print("  5. 결과 비교 (XGBoost vs LSTM)")
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


def step2_xgboost():
    """Step 2: XGBoost 학습"""
    print("\n" + "="*80)
    print("🎯 Step 2: XGBoost 모델 학습")
    print("="*80)
    
    # 전처리 데이터 확인
    data_path = '../data/ver2_paired_visits.csv'
    if not os.path.exists(data_path):
        print(f"\n⚠️  전처리 데이터가 없습니다: {data_path}")
        print("먼저 '1. 데이터 전처리'를 실행하세요.")
        return False
    
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


def step3_lstm():
    """Step 3: LSTM 학습"""
    print("\n" + "="*80)
    print("🎯 Step 3: LSTM 모델 학습")
    print("="*80)
    
    # 전처리 데이터 확인
    data_path = '../data/ver2_paired_visits.csv'
    if not os.path.exists(data_path):
        print(f"\n⚠️  전처리 데이터가 없습니다: {data_path}")
        print("먼저 '1. 데이터 전처리'를 실행하세요.")
        return False
    
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


def step4_full_pipeline():
    """Step 4: 전체 파이프라인"""
    print("\n" + "="*80)
    print("🚀 전체 파이프라인 실행")
    print("="*80)
    
    # Step 1: 전처리
    if not step1_preprocessing():
        return False
    
    # Step 2: XGBoost
    if not step2_xgboost():
        return False
    
    # Step 3: LSTM
    if not step3_lstm():
        return False
    
    print("\n" + "="*80)
    print("✅ 전체 파이프라인 완료!")
    print("="*80)
    return True


def step5_compare_results():
    """Step 5: 결과 비교"""
    print("\n" + "="*80)
    print("📊 XGBoost vs LSTM 결과 비교")
    print("="*80)
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 결과 파일 확인
    xgb_path = '../result/xgboost_all_results.csv'
    lstm_path = '../result/lstm_all_results.csv'
    
    if not os.path.exists(xgb_path):
        print(f"⚠️  XGBoost 결과 없음: {xgb_path}")
        return False
    
    if not os.path.exists(lstm_path):
        print(f"⚠️  LSTM 결과 없음: {lstm_path}")
        return False
    
    # 결과 로드
    xgb_results = pd.read_csv(xgb_path, index_col=0)
    lstm_results = pd.read_csv(lstm_path, index_col=0)
    
    print("\n📊 XGBoost 결과:")
    print(xgb_results.round(4))
    
    print("\n📊 LSTM 결과:")
    print(lstm_results.round(4))
    
    # 비교 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['R²', 'RMSE', 'MAE', 'Direction_Accuracy']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        indicators = xgb_results.index
        x = range(len(indicators))
        width = 0.35
        
        xgb_values = xgb_results[metric].values
        lstm_values = lstm_results[metric].values
        
        ax.bar([i - width/2 for i in x], xgb_values, width, label='XGBoost', alpha=0.8)
        ax.bar([i + width/2 for i in x], lstm_values, width, label='LSTM', alpha=0.8)
        
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
    
    # 차이 계산
    print("\n" + "="*80)
    print("📈 모델 간 성능 차이 (LSTM - XGBoost)")
    print("="*80)
    
    diff = lstm_results - xgb_results
    print("\n", diff.round(4))
    
    # 승자 카운트
    print("\n" + "="*80)
    print("🏆 지표별 우수 모델")
    print("="*80)
    
    for metric in metrics:
        if metric in ['R²', 'Direction_Accuracy']:  # 높을수록 좋음
            better = (lstm_results[metric] > xgb_results[metric]).sum()
        else:  # 낮을수록 좋음 (RMSE, MAE)
            better = (lstm_results[metric] < xgb_results[metric]).sum()
        
        print(f"\n{metric}:")
        print(f"  LSTM 우세: {better}/{len(indicators)} 지표")
        print(f"  XGBoost 우세: {len(indicators) - better}/{len(indicators)} 지표")
    
    return True


def main():
    """메인 실행"""
    while True:
        print_menu()
        
        try:
            choice = input("\n선택 (0-5): ").strip()
            
            if choice == '0':
                print("\n👋 프로그램을 종료합니다.")
                break
            
            elif choice == '1':
                step1_preprocessing()
            
            elif choice == '2':
                step2_xgboost()
            
            elif choice == '3':
                step3_lstm()
            
            elif choice == '4':
                step4_full_pipeline()
            
            elif choice == '5':
                step5_compare_results()
            
            else:
                print("\n⚠️  잘못된 선택입니다. 0-5 사이의 숫자를 입력하세요.")
            
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
