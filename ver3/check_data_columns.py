"""
Ver3: 데이터 컬럼 확인 스크립트
"""

import pandas as pd
import glob
import os

def check_columns():
    """결과 파일의 컬럼 확인"""
    
    # 최신 결과 파일 찾기
    paired_files = glob.glob('./results/paired_data_*.csv')
    
    if not paired_files:
        print("❌ 결과 파일을 찾을 수 없습니다!")
        print("   먼저 파이프라인을 실행하세요:")
        print("   python run_ver3_pipeline.py --data ../data/total_again.xlsx")
        return
    
    latest_file = max(paired_files, key=os.path.getmtime)
    print(f"✅ 데이터 파일: {latest_file}")
    
    # 데이터 로드
    df = pd.read_csv(latest_file)
    
    print(f"\n📊 데이터 정보:")
    print(f"   - 샘플 수: {len(df):,}개")
    print(f"   - 컬럼 수: {len(df.columns):,}개")
    
    # 모든 컬럼 출력
    print(f"\n📋 전체 컬럼 리스트 ({len(df.columns)}개):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:3d}. {col}")
    
    # MetS 관련 컬럼 찾기
    print(f"\n🔍 MetS 관련 컬럼:")
    mets_cols = [col for col in df.columns if 'mets' in col.lower()]
    
    if mets_cols:
        for col in mets_cols:
            print(f"   - {col}")
            print(f"     유니크 값: {df[col].nunique()}개")
            print(f"     값 분포:")
            print(df[col].value_counts().to_string().replace('\n', '\n     '))
            print()
    else:
        print("   ⚠️  'mets' 관련 컬럼을 찾을 수 없습니다!")
    
    # 기타 중요 컬럼 확인
    print(f"\n🔍 기타 중요 컬럼:")
    
    important_patterns = ['change', 'baseline', 'followup', 'pattern', 'label', 'class']
    
    for pattern in important_patterns:
        matching_cols = [col for col in df.columns if pattern in col.lower()]
        if matching_cols:
            print(f"\n   [{pattern}] 관련 컬럼:")
            for col in matching_cols:
                print(f"   - {col}")

if __name__ == "__main__":
    check_columns()
