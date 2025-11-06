"""
전체 분석 파이프라인 실행
========================

모든 고급 분석을 순차적으로 실행:
1. 기초 분석 (세부 그룹 통계)
2. 임계값 분석 (빠름)
3. 세부 그룹 모델 학습 (시간 소요)
4. SHAP 분석 (시간 소요)
5. 최종 보고서 업데이트

Author: Research Team
Date: 2025-11-06
"""

import sys
from pathlib import Path
import time

def run_phase(phase_name, script_name, estimated_time):
    """단계별 실행"""
    print("\n" + "="*80)
    print(f"🚀 {phase_name}")
    print(f"예상 시간: {estimated_time}")
    print("="*80)
    
    response = input("\n실행하시겠습니까? (y/n): ").strip().lower()
    
    if response != 'y':
        print(f"⏭️  {phase_name} 건너뜀")
        return False
    
    print(f"\n▶️  {script_name} 실행 중...")
    start_time = time.time()
    
    try:
        exec(open(script_name).read())
        elapsed = time.time() - start_time
        print(f"\n✅ {phase_name} 완료 (소요 시간: {elapsed/60:.1f}분)")
        return True
    except Exception as e:
        print(f"\n❌ {phase_name} 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 실행"""
    print("\n" + "="*80)
    print("📊 Ver1 Advanced Analysis - 전체 파이프라인")
    print("="*80)
    print("")
    print("분석 단계:")
    print("   1. 기초 분석 (세부 그룹 통계) - 약 2분")
    print("   2. 임계값 분석 (건강 위험 기준) - 약 5분")
    print("   3. 세부 그룹 모델 학습 - 약 30-60분 ⏰")
    print("   4. SHAP 해석력 분석 - 약 30-60분 ⏰")
    print("   5. 최종 보고서 생성 - 즉시")
    print("")
    print("전체 예상 시간: 약 1-2시간")
    print("")
    print("="*80)
    
    response = input("\n계속 진행하시겠습니까? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\n종료합니다.")
        return
    
    # Phase 1: 기초 분석
    run_phase(
        "Phase 1: 기초 분석 (세부 그룹 통계)",
        "advanced_analysis.py",
        "약 2분"
    )
    
    # Phase 2: 임계값 분석
    run_phase(
        "Phase 2: 임계값 분석 (건강 위험 기준)",
        "threshold_analysis.py",
        "약 5분"
    )
    
    # Phase 3: 세부 그룹 모델 (시간 많이 소요)
    print("\n" + "="*80)
    print("⚠️  Phase 3: 세부 그룹 모델 학습")
    print("="*80)
    print("\n이 단계는 시간이 오래 걸립니다 (30-60분).")
    print("지금 실행하지 않고 나중에 별도로 실행할 수도 있습니다:")
    print("   python subgroup_modeling.py")
    print("")
    
    run_phase(
        "Phase 3: 세부 그룹 모델 학습",
        "subgroup_modeling.py",
        "약 30-60분"
    )
    
    # Phase 4: SHAP 분석 (시간 많이 소요)
    print("\n" + "="*80)
    print("⚠️  Phase 4: SHAP 해석력 분석")
    print("="*80)
    print("\n이 단계도 시간이 오래 걸립니다 (30-60분).")
    print("SHAP library가 설치되어 있어야 합니다:")
    print("   pip install shap")
    print("")
    print("지금 실행하지 않고 나중에 별도로 실행할 수도 있습니다:")
    print("   python shap_analysis.py")
    print("")
    
    run_phase(
        "Phase 4: SHAP 해석력 분석",
        "shap_analysis.py",
        "약 30-60분"
    )
    
    # Phase 5: 최종 보고서
    print("\n" + "="*80)
    print("📄 Phase 5: 최종 보고서 생성")
    print("="*80)
    
    try:
        exec(open("generate_paper_report.py").read())
        print("\n✅ 최종 보고서 생성 완료!")
    except Exception as e:
        print(f"\n❌ 보고서 생성 오류: {str(e)}")
    
    # 완료
    print("\n" + "="*80)
    print("🎉 전체 분석 완료!")
    print("="*80)
    print("")
    print("결과 위치:")
    print("   - 기초 분석: advanced_results/")
    print("   - 세부 그룹 모델: advanced_results/subgroup_models/")
    print("   - SHAP 분석: advanced_results/shap_analysis/")
    print("   - 임계값 분석: advanced_results/threshold_analysis/")
    print("   - 최종 보고서: advanced_results/FINAL_PAPER_REPORT.md")
    print("")
    print("다음 단계:")
    print("   1. 보고서 검토 및 편집")
    print("   2. 그래프 추가 및 포맷팅")
    print("   3. 학술지 투고 준비")
    print("")


if __name__ == '__main__':
    main()
