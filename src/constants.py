"""
컬럼 이름 상수. utils.py(torch 등 무거운 의존성 포함)와 분리해서,
feature_engineering.py처럼 상수만 필요한 가벼운 소비자(예: 추론 서비스)가
torch를 끌고 오지 않도록 함.
"""
demo_cols = ['days_between', '나이_T0', '신장_T0', '성별_T0']
life_cols = ['흡연_T0', '활동량_T0', '음주_T0']
bio_cols = ['SBP_T0', 'DBP_T0', 'GLUCOSE_T0', 'HBA1C_T0', 'TG_T0', 'HDL CHOL_T0', 'LDL CHOL_T0', 'eGFR_T0', 'WAIST_T0', '체중_T0', 'BMI_T0']
diet_cols = ['간식빈도_T0', '고지방 육류_T0', '곡류_T0', '과일_T0', '단맛_T0', '단백질류_T0', '물_T0', '밥 양_T0',
             '식사 빈도_T0', '식사량_T0', '외식빈도_T0', '유제품_T0', '음료류_T0', '인스턴트 가공식품_T0', '짠 간_T0', '짠 식습관_T0', '채소_T0', '커피_T0', '튀김_T0']
interaction_cols = ['bmi_waist_risk', 'bp_age_risk', 'tg_hdl_ratio', 'unhealthy_diet_score', 'healthy_diet_score', 'diet_change_rate']
mets_cols = ['Increased waist circumference', 'Elevated blood pressure', 'Impaired fasting glucose', 'Elevated triglycerides', 'Decreased HDL-C']
disease_delta_cols = [f'{disease}_delta' for disease in mets_cols]
