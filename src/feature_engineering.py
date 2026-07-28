import numpy as np
from sklearn.decomposition import PCA
from constants import mets_cols

def create_medical_interactions(df):
    df_enhanced = df.copy()
    
    # 1. 핵심 의료 상호작용
    # BMI-허리둘레 (복부비만 지표)
    if 'BMI_T0' in df.columns and 'WAIST_T0' in df.columns:
        df_enhanced['bmi_waist_risk'] = df['BMI_T0'] * df['WAIST_T0'] / 100
    
    # 혈압-나이 (고령 고혈압 위험)
    if 'SBP_T0' in df.columns and '나이_T0' in df.columns:
        df_enhanced['bp_age_risk'] = df['SBP_T0'] * df['나이_T0'] / 100
    
    # TG/HDL 비율 (심혈관 위험도)
    if 'TG_T0' in df.columns and 'HDL CHOL_T0' in df.columns:
        df_enhanced['tg_hdl_ratio'] = df['TG_T0'] / (df['HDL CHOL_T0'] + 1)
    
    # 2. 식습관 조합 (2가지만)
    # 나쁜 식습관 조합
    bad_diet_cols = ['고지방 육류_T0', '튀김_T0', '단맛_T0']
    available_bad = [col for col in bad_diet_cols if col in df.columns]
    if available_bad:
        df_enhanced['unhealthy_diet_score'] = df[available_bad].mean(axis=1)
    
    # 좋은 식습관 조합
    good_diet_cols = ['과일_T0', '채소_T0', '물_T0']
    available_good = [col for col in good_diet_cols if col in df.columns]
    if available_good:
        df_enhanced['healthy_diet_score'] = df[available_good].mean(axis=1)
    
    return df_enhanced

def create_change_features(df):
    df_temporal = df.copy()
    
    # 1. 변화 속도 (3가지만)
    if 'days_between' in df.columns:
        # 체중 변화 속도
        if '체중_delta' in df.columns:
            df_temporal['weight_change_rate'] = df['체중_delta'] / (df['days_between'] + 1)
        
        # 전체 식습관 변화 속도
        diet_delta_cols = [col for col in df.columns if '_delta' in col and 
                          any(food in col for food in ['과일', '채소', '튀김', '단맛'])]
        if diet_delta_cols:
            df_temporal['diet_change_rate'] = df[diet_delta_cols].abs().sum(axis=1) / (df['days_between'] + 1)
    
    # 2. 상대적 변화 (2가지만)
    # 개인 기준 대비 체중 변화 비율
    if '체중_delta' in df.columns and '체중_T0' in df.columns:
        df_temporal['relative_weight_change'] = df['체중_delta'] / (df['체중_T0'] + 1)
    
    # 3. 변화 방향성 (1가지만)
    # 체중 변화가 의미있는 수준인지
    if '체중_delta' in df.columns:
        df_temporal['significant_weight_change'] = (abs(df['체중_delta']) > 2).astype(int)
    
    return df_temporal

def apply_to_all_data(train_df, val_df, test_df):
    original_features = len(train_df.columns)
    original_columns = train_df.columns.tolist()
    
    print("=== 피처 엔지니어링 시작 ===")
    
    # 1. 상호작용 + 변화 피처 적용
    enhanced_train = create_medical_interactions(train_df)
    enhanced_train = create_change_features(enhanced_train)
    
    enhanced_val = create_medical_interactions(val_df)
    enhanced_val = create_change_features(enhanced_val)
    
    enhanced_test = create_medical_interactions(test_df)
    enhanced_test = create_change_features(enhanced_test)
    
    # 2. PCA 적용
    numeric_cols = enhanced_train.select_dtypes(include=[np.number]).columns
    disease_delta_cols = [col for col in original_columns if '_delta' in col and 
                         any(disease in col for disease in mets_cols)]
    numeric_cols = [col for col in numeric_cols if col not in disease_delta_cols]
    
    X_train = enhanced_train[numeric_cols].fillna(0)
    
    pca = PCA(n_components=10, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    
    X_val_pca = pca.transform(enhanced_val[numeric_cols].fillna(0))
    X_test_pca = pca.transform(enhanced_test[numeric_cols].fillna(0))
    
    for i in range(10):  # PCA 컴포넌트 추가
        col_name = f'pca_component_{i}'
        enhanced_train[col_name] = X_train_pca[:, i]
        enhanced_val[col_name] = X_val_pca[:, i]
        enhanced_test[col_name] = X_test_pca[:, i]
    
    print(f"✅ PCA 설명 분산비: {pca.explained_variance_ratio_.sum():.3f}")
    
    # 3. 컬럼 순서 정리
    disease_delta_cols = [col for col in original_columns if '_delta' in col and 
                         any(disease in col for disease in mets_cols)]
    base_cols = [col for col in original_columns if col not in disease_delta_cols]
    new_cols = [col for col in enhanced_train.columns if col not in original_columns]
    final_column_order = base_cols + new_cols + disease_delta_cols
    
    enhanced_train = enhanced_train[final_column_order]
    enhanced_val = enhanced_val[final_column_order]
    enhanced_test = enhanced_test[final_column_order]
    
    # 4. 정확한 change_dim 계산
    interaction_features = [col for col in new_cols if any(pattern in col for pattern in 
                           ['_risk', '_ratio', '_score', '_rate', '_change'])]
    pca_features = [col for col in new_cols if 'pca_component' in col]
    
    actual_change_dim = len(interaction_features)  # PCA는 별도로 처리
    
    print(f"=== 피처 엔지니어링 완료 ===")
    print(f"상호작용 피처: {len(interaction_features)}개")
    print(f"PCA 피처: {len(pca_features)}개") 
    print(f"실제 change_dim: {actual_change_dim}")

    return enhanced_train, enhanced_val, enhanced_test, actual_change_dim, pca
