"""
Complete Analysis Pipeline - 전체 분석 자동 실행
===============================================

Phase 1: Ver1 모델 학습 (TabNet + Stacking)
Phase 2: 임계값 분석 (완료됨)
Phase 3: 서브그룹 모델링
Phase 4: SHAP 해석성 분석

Author: Research Team
Date: 2025-11-06
"""

import os
import sys
import time
from pathlib import Path

# 경로 설정
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / 'src'))
os.chdir(script_dir)


def print_header(title):
    """섹션 헤더 출력"""
    print("\n" + "=" * 80)
    print(f"🚀 {title}")
    print("=" * 80 + "\n")


def phase1_train_model():
    """Phase 1: Ver1 모델 학습"""
    print_header("Phase 1: Ver1 모델 학습 (TabNet + Stacking)")
    print("⏱️  예상 시간: 10-20분")
    print("📊 안전 모드로 실행 (Optuna 없음)\n")
    
    start = time.time()
    
    try:
        from TABNET_ENHANCED_MODEL import main
        main(use_tabnet_stacking=True, use_optuna=False)
        
        elapsed = time.time() - start
        print(f"\n✅ Phase 1 완료 ({elapsed/60:.1f}분 소요)")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 1 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def phase2_threshold_analysis():
    """Phase 2: 임계값 분석 (이미 완료됨)"""
    print_header("Phase 2: 임계값 분석")
    
    threshold_dir = Path('./advanced_results/threshold_analysis')
    
    if threshold_dir.exists() and len(list(threshold_dir.glob('*.csv'))) >= 4:
        print("✅ Phase 2 이미 완료됨")
        print(f"   결과 위치: {threshold_dir}")
        return True
    else:
        print("⚠️  Phase 2 결과 없음, 재실행 필요")
        return False


def phase3_subgroup_modeling():
    """Phase 3: 서브그룹 모델링"""
    print_header("Phase 3: 서브그룹별 모델 학습")
    print("⏱️  예상 시간: 30-60분")
    print("📊 연령/성별/BMI별 전용 모델 생성\n")
    
    start = time.time()
    
    try:
        from subgroup_modeling import SubgroupModeling
        
        modeler = SubgroupModeling()
        results = modeler.train_all_subgroups()
        modeler.save_results()
        
        elapsed = time.time() - start
        print(f"\n✅ Phase 3 완료 ({elapsed/60:.1f}분 소요)")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 3 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def phase4_shap_analysis():
    """Phase 4: SHAP 해석성 분석"""
    print_header("Phase 4: SHAP 해석성 분석")
    print("⏱️  예상 시간: 30-60분")
    print("📊 특성 중요도 및 상호작용 분석\n")
    
    start = time.time()
    
    try:
        from shap_analysis import SHAPAnalyzer
        
        analyzer = SHAPAnalyzer()
        results = analyzer.analyze_all_indicators()
        analyzer.generate_summary_report()
        
        elapsed = time.time() - start
        print(f"\n✅ Phase 4 완료 ({elapsed/60:.1f}분 소요)")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 4 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """전체 파이프라인 실행"""
    print("\n" + "=" * 80)
    print("🎯 Complete Analysis Pipeline")
    print("=" * 80)
    print("\n전체 분석을 순차적으로 실행합니다.")
    print("예상 총 시간: 1-2시간\n")
    
    input("Enter 키를 눌러 시작하세요...")
    
    pipeline_start = time.time()
    results = {}
    
    # Phase 1: 모델 학습
    results['phase1'] = phase1_train_model()
    
    if not results['phase1']:
        print("\n⚠️  Phase 1 실패로 인해 파이프라인 중단")
        return
    
    # Phase 2: 임계값 분석 (이미 완료됨)
    results['phase2'] = phase2_threshold_analysis()
    
    # Phase 3: 서브그룹 모델링
    results['phase3'] = phase3_subgroup_modeling()
    
    # Phase 4: SHAP 분석 (Phase 1 성공 시에만)
    if results['phase1']:
        results['phase4'] = phase4_shap_analysis()
    else:
        print("\n⚠️  Phase 1 실패로 인해 Phase 4 건너뜀")
        results['phase4'] = False
    
    # 최종 요약
    total_elapsed = time.time() - pipeline_start
    
    print("\n" + "=" * 80)
    print("📊 전체 파이프라인 완료")
    print("=" * 80)
    print(f"\n⏱️  총 소요 시간: {total_elapsed/60:.1f}분\n")
    
    print("결과 요약:")
    print(f"  Phase 1 (모델 학습):    {'✅ 성공' if results['phase1'] else '❌ 실패'}")
    print(f"  Phase 2 (임계값 분석):  {'✅ 완료' if results['phase2'] else '⚠️  없음'}")
    print(f"  Phase 3 (서브그룹):     {'✅ 성공' if results['phase3'] else '❌ 실패'}")
    print(f"  Phase 4 (SHAP 분석):    {'✅ 성공' if results['phase4'] else '❌ 실패'}")
    
    print("\n결과 위치:")
    print("  - 모델: ./result/models/")
    print("  - 임계값: ./advanced_results/threshold_analysis/")
    print("  - 서브그룹: ./advanced_results/subgroup_models/")
    print("  - SHAP: ./advanced_results/shap_analysis/")
    print("  - 논문: ./advanced_results/FINAL_PAPER_REPORT.md")
    
    success_count = sum(results.values())
    print(f"\n🎯 전체 성공률: {success_count}/4 ({success_count/4*100:.0f}%)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 실행을 중단했습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
