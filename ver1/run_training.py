"""
TabNet Enhanced Model - Python 실행 스크립트
===========================================
여러 가지 실행 모드를 제공합니다.
"""

import sys
import os

# 스크립트 파일의 디렉토리를 기준으로 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))

# src 폴더를 path에 추가
sys.path.insert(0, os.path.join(script_dir, 'src'))

# 작업 디렉토리를 프로젝트 루트로 변경 (상대 경로 문제 해결)
os.chdir(script_dir)

from TABNET_ENHANCED_MODEL import main


def print_menu():
    """실행 모드 선택 메뉴"""
    print("\n" + "=" * 80)
    print("🚀 TabNet Enhanced Model - 실행 모드 선택")
    print("=" * 80)
    print("\n실행 모드를 선택하세요:")
    print()
    print("  1. 안전 모드 (추천) - TabNet + Stacking, Optuna 없음")
    print("     → 빠르고 안정적, Optuna segfault 문제 없음")
    print()
    print("  2. 전체 최적화 모드 - TabNet + Stacking + Optuna")
    print("     → 최고 성능, 하지만 Optuna segfault 발생 가능")
    print()
    print("  3. TabNet만 사용 - Stacking 없음, Optuna 없음")
    print("     → 순수 TabNet 딥러닝 모델만 사용")
    print()
    print("  4. 빠른 테스트 모드 - Optuna 5회만 (빠른 최적화)")
    print("     → 전체 최적화보다 빠르지만 성능은 약간 낮음")
    print()
    print("=" * 80)
    

def run_safe_mode():
    """안전 모드: TabNet + Stacking, Optuna 없음"""
    print("\n✅ 안전 모드로 실행합니다...")
    print("   - TabNet + Stacking Ensemble: 사용")
    print("   - Optuna 최적화: 미사용")
    print()
    main(use_tabnet_stacking=True, use_optuna=False)


def run_full_optimization():
    """전체 최적화 모드: TabNet + Stacking + Optuna"""
    print("\n⚡ 전체 최적화 모드로 실행합니다...")
    print("   - TabNet + Stacking Ensemble: 사용")
    print("   - Optuna 최적화: 사용 (20회 시도)")
    print("   ⚠️ 주의: Optuna segfault 발생 가능")
    print()
    main(use_tabnet_stacking=True, use_optuna=True, optuna_trials=20)


def run_tabnet_only():
    """TabNet만 사용: Stacking 없음"""
    print("\n🧠 TabNet 전용 모드로 실행합니다...")
    print("   - TabNet 딥러닝 모델만 사용")
    print("   - Stacking Ensemble: 미사용")
    print("   - Optuna 최적화: 미사용")
    print()
    main(use_tabnet_stacking=False, use_optuna=False)


def run_quick_test():
    """빠른 테스트 모드: Optuna 5회만"""
    print("\n⚡ 빠른 테스트 모드로 실행합니다...")
    print("   - TabNet + Stacking Ensemble: 사용")
    print("   - Optuna 최적화: 사용 (5회 시도만)")
    print("   ⚠️ 주의: Optuna segfault 발생 가능")
    print()
    main(use_tabnet_stacking=True, use_optuna=True, optuna_trials=5)


if __name__ == "__main__":
    # 명령줄 인자가 있으면 직접 실행
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "safe" or mode == "1":
            run_safe_mode()
        elif mode == "full" or mode == "2":
            run_full_optimization()
        elif mode == "tabnet" or mode == "3":
            run_tabnet_only()
        elif mode == "quick" or mode == "4":
            run_quick_test()
        else:
            print(f"❌ 알 수 없는 모드: {mode}")
            print("\n사용법:")
            print("  python run_training.py [safe|full|tabnet|quick]")
            print("  python run_training.py [1|2|3|4]")
            sys.exit(1)
    else:
        # 명령줄 인자가 없으면 메뉴 표시
        print_menu()
        
        try:
            choice = input("모드 선택 (1-4): ").strip()
            
            if choice == "1":
                run_safe_mode()
            elif choice == "2":
                run_full_optimization()
            elif choice == "3":
                run_tabnet_only()
            elif choice == "4":
                run_quick_test()
            else:
                print(f"\n❌ 잘못된 선택: {choice}")
                print("1, 2, 3, 4 중 하나를 선택하세요.")
                sys.exit(1)
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 사용자가 실행을 취소했습니다.")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            sys.exit(1)
