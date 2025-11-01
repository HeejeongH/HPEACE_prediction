#!/bin/bash

# 고성능 6개 바이오마커 TabNet 포함 최고 성능 학습
# 예상 시간: 2~3시간 (기존 4~6시간에서 단축)
# TabNet + Stacking + Optuna 포함

echo "=================================="
echo "🚀 고성능 바이오마커 학습 시작"
echo "=================================="
echo "시작 시간: $(date)"
echo ""
echo "설정:"
echo "  - 바이오마커: 6개 고성능 모델"
echo "    (체중, BMI, 허리둘레, SBP, DBP, TG)"
echo "  - 제외: 저성능 바이오마커 5개"
echo "    (GLUCOSE, HBA1C, HDL, LDL, eGFR)"
echo "  - 모델: TabNet + Stacking Ensemble"
echo "  - Optuna 최적화: trials=20"
echo "  - 예상 시간: 2~3시간"
echo "  - 예상 평균 R²: 0.70~0.75"
echo ""
echo "로그 파일: /home/user/webapp/training.log"
echo "진행 상황은 로그 파일에서 확인 가능합니다."
echo ""

cd /home/user/webapp/src

# TabNet 포함 전체 학습 실행
python -c "
from TABNET_ENHANCED_MODEL import main
import sys
import time

start_time = time.time()

print('='*80)
print('🧠 TabNet 딥러닝 포함 전체 학습 시작')
print('='*80)
print()

try:
    results, summary = main(
        use_tabnet_stacking=True,   # TabNet + 기존 모델 Stacking
        use_optuna=True,             # Optuna 최적화
        optuna_trials=20             # 각 모델당 20회 최적화
    )
    
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    
    print()
    print('='*80)
    print('✅ 학습 완료!')
    print('='*80)
    print(f'총 소요 시간: {hours}시간 {minutes}분')
    print()
    print('📊 최종 결과:')
    print(summary.to_string(index=False))
    print()
    print('결과가 저장되었습니다.')
    
except Exception as e:
    print()
    print('='*80)
    print('❌ 오류 발생')
    print('='*80)
    print(f'오류 내용: {str(e)}')
    sys.exit(1)
" 2>&1 | tee /home/user/webapp/training.log

echo ""
echo "=================================="
echo "✅ 학습 프로세스 완료"
echo "=================================="
echo "종료 시간: $(date)"
echo "결과 로그: /home/user/webapp/training.log"
