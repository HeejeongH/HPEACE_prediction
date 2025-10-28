# 수정된 cell-4 코드
# 노트북에서 이 코드를 복사해서 cell-4를 교체하세요

# 방문 쌍 생성 함수들
def create_first_last_pairs(df, id_col, date_col, min_interval_days=30, max_interval_days=2190):
    """
    첫 방문 - 마지막 방문 쌍 생성
    """
    print(f"🔄 첫-마지막 방문 쌍 생성 (간격: {min_interval_days}~{max_interval_days}일)")

    pairs = []

    for patient_id in df[id_col].unique():
        patient_data = df[df[id_col] == patient_id].sort_values(date_col)

        if len(patient_data) >= 2:
            first_visit = patient_data.iloc[0]
            last_visit = patient_data.iloc[-1]

            # 방문 간격 계산
            days_interval = (last_visit[date_col] - first_visit[date_col]).days

            if min_interval_days <= days_interval <= max_interval_days:
                pairs.append({
                    'patient_id': patient_id,
                    'first_visit': first_visit,
                    'second_visit': last_visit,
                    'days_interval': days_interval,
                    'visit_gap': len(patient_data) - 1,  # 중간에 몇 번의 방문이 있었는지
                    'strategy': 'first_last'
                })

    return pairs

def create_max_change_pairs(df, id_col, date_col, biomarkers, min_interval_days=30, max_interval_days=2190):
    """
    바이오마커 변화가 가장 큰 방문 쌍 생성
    """
    print(f"🔄 최대 변화 방문 쌍 생성 (간격: {min_interval_days}~{max_interval_days}일)")

    pairs = []

    for patient_id in df[id_col].unique():
        patient_data = df[df[id_col] == patient_id].sort_values(date_col)

        if len(patient_data) >= 2:
            max_change_score = 0
            best_pair = None

            # 모든 방문 쌍 조합 확인
            for i in range(len(patient_data)):
                for j in range(i+1, len(patient_data)):
                    visit1 = patient_data.iloc[i]
                    visit2 = patient_data.iloc[j]

                    days_interval = (visit2[date_col] - visit1[date_col]).days

                    if min_interval_days <= days_interval <= max_interval_days:
                        # 바이오마커 변화량 계산
                        change_score = 0
                        valid_changes = 0

                        for biomarker in biomarkers:
                            if biomarker in visit1.index and biomarker in visit2.index:
                                val1 = pd.to_numeric(visit1[biomarker], errors='coerce')
                                val2 = pd.to_numeric(visit2[biomarker], errors='coerce')

                                if pd.notna(val1) and pd.notna(val2) and val1 > 0:
                                    # 상대적 변화량 사용
                                    relative_change = abs(val2 - val1) / val1
                                    change_score += relative_change
                                    valid_changes += 1

                        if valid_changes > 0:
                            avg_change_score = change_score / valid_changes

                            if avg_change_score > max_change_score:
                                max_change_score = avg_change_score
                                best_pair = {
                                    'patient_id': patient_id,
                                    'first_visit': visit1,
                                    'second_visit': visit2,
                                    'days_interval': days_interval,
                                    'change_score': avg_change_score,
                                    'valid_biomarkers': valid_changes,
                                    'strategy': 'max_change'
                                }

            if best_pair is not None:
                pairs.append(best_pair)

    return pairs

# ========== 여기가 중요! 함수 호출 시 인자 수정 ==========
# 두 전략으로 방문 쌍 생성
first_last_pairs = create_first_last_pairs(df_multi, id_col, date_col)  # ← available_biomarkers 제거!
max_change_pairs = create_max_change_pairs(df_multi, id_col, date_col, available_biomarkers)

print(f"\n📊 방문 쌍 생성 결과:")
print(f"  첫-마지막 전략: {len(first_last_pairs):,}개 쌍")
print(f"  최대변화 전략: {len(max_change_pairs):,}개 쌍")

if first_last_pairs:
    fl_intervals = [p['days_interval'] for p in first_last_pairs]
    print(f"  첫-마지막 평균 간격: {np.mean(fl_intervals):.1f}일")

if max_change_pairs:
    mc_intervals = [p['days_interval'] for p in max_change_pairs]
    mc_changes = [p['change_score'] for p in max_change_pairs]
    print(f"  최대변화 평균 간격: {np.mean(mc_intervals):.1f}일")
    print(f"  평균 변화점수: {np.mean(mc_changes):.4f}")
