import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import stats
import statsmodels.api as sm


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def normalize_missing(df: pd.DataFrame, markers: Optional[List[str]] = None) -> pd.DataFrame:
    out = df.copy()
    out[['일반담배_흡연여부', '음주']] = out[['일반담배_흡연여부', '음주']].apply(pd.to_numeric, errors='coerce')

    if markers is None:
        markers = ["Missing value", "missing value", "MISSING VALUE","Missing", "missing", "N/A", "NA", "na", "NaN", "nan"]
    obj_cols = out.select_dtypes(include=["object"]).columns
    out[obj_cols] = out[obj_cols].apply(lambda s: s.str.strip())
    out = out.replace(markers, np.nan)
    return out

def subset_multi_visit(df: pd.DataFrame, idcol: str) -> pd.DataFrame:
    mask = df.groupby(idcol)[idcol].transform("size") >= 2
    return df.loc[mask].copy()

def add_visit_order_and_intervals(df: pd.DataFrame, idcol: str, datecol: str) -> pd.DataFrame:
    out = df.copy()
    out[datecol] = pd.to_datetime(out[datecol], errors="coerce")
    out = out.sort_values([idcol, datecol]).reset_index(drop=True)
    out["visit_index"] = out.groupby(idcol).cumcount()
    out["days_since_prev"] = out.groupby(idcol)[datecol].diff().dt.days
    out["interval_years"] = out["days_since_prev"] / 365.25
    return out

def add_med_change(df: pd.DataFrame, idcol: str, medications: List[str]) -> pd.DataFrame:
    out = df.copy()
    med_cols = [c for c in medications if c in out.columns]
    if len(med_cols) == 0:
        out["med_any_change"] = 0
        return out
    out["med_any"] = (out[med_cols].fillna(0).sum(axis=1) > 0).astype(int)
    out["med_any_change"] = (out.groupby(idcol)["med_any"].diff().fillna(0).ne(0).astype(int))
    return out

def map_gender(df: pd.DataFrame,col: str = "성별",mapping: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        return out
    s = out[col]
    if np.issubdtype(s.dtype, np.number):
        return out
    mapping = mapping or {
        "M": 1, "F": 0,
    }
    try:
        s2 = s.astype(str).str.strip().map(mapping)
        if s2.notna().sum() >= max(1, int(0.5 * s.notna().sum())):
            out[col] = s2.astype(float)
    except Exception:
        pass
    return out

def prepare_panel(
    df: pd.DataFrame,
    idcol: str,
    datecol: str,
    ffqs: List[str],
    biomarkers: List[str],
    lifestyles: Optional[List[str]] = None,
    medications: Optional[List[str]] = None,
    extra_numeric: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    
    lifestyles = lifestyles or []
    medications = medications or []
    extra_numeric = extra_numeric or []
    exclude = set(exclude or [])

    ffqs = [c for c in ffqs if c not in exclude]
    biomarkers = [c for c in biomarkers if c not in exclude]
    lifestyles = [c for c in lifestyles if c not in exclude]
    medications = [c for c in medications if c not in exclude]
    extra_numeric = [c for c in extra_numeric if c not in exclude]

    cols_to_num = [c for c in (ffqs + biomarkers + lifestyles + medications + extra_numeric) if c in df.columns]
    out = normalize_missing(df)

    if '성별' not in exclude:
        out = map_gender(out, col='성별')
    
    out = coerce_numeric(out, cols_to_num)
    out = add_visit_order_and_intervals(out, idcol=idcol, datecol=datecol)
    out = subset_multi_visit(out, idcol=idcol)
    out = add_med_change(out, idcol=idcol, medications=medications)
    
    # 파생변수 생성
    out['Increased waist circumference'] = (
        ((out['성별'] == 'M') & (out['허리둘레(WAIST)'].astype(float) >= 90)) | 
        ((out['성별'] == 'F') & (out['허리둘레(WAIST)'].astype(float) >= 85))
    )
    out['Elevated blood pressure'] = (
        ((out['SBP'].astype(float) >= 130) | 
            (out['DBP'].astype(float) >= 85)) | 
        (out['고혈압_투약여부'] == 1)
    )
    out['Impaired fasting glucose'] = (
        (out['GLUCOSE'].astype(float) >= 100) | 
        (out['당뇨_투약여부'] == 1)
    )
    out['Elevated triglycerides'] = (
        (out['TG'].astype(float) >= 150) | 
        (out['고지혈증_투약여부'] == 1)
    )
    out['Decreased HDL-C'] = (
        ((out['성별'] == 'M') & (out['HDL CHOL.'].astype(float) < 40)) | 
        ((out['성별'] == 'F') & (out['HDL CHOL.'].astype(float) < 50))
    )
    
    # 카테고리 컬럼들
    category_cols = ['성별', '일반담배_흡연여부', '음주', '간식빈도', '고지방 육류', '곡류', '과일', 
                     '단맛', '단백질류', '물', '밥 양', '식사 빈도', '식사량', '외식빈도', '유제품', 
                     '음료류', '인스턴트 가공식품', '짠 간', '짠 식습관', '채소', '커피', '튀김']

    # 여/부 컬럼들  
    binary_cols = ['고혈압_투약여부', '당뇨_투약여부', '고지혈증_투약여부', '고혈압_통합', '당뇨_통합', 
                   '고지혈증_통합', '협심증/심근경색증_통합', '뇌졸중(중풍)_통합', '비만', 
                   'Chronic kidney disease (eGFR<60)', 'Increased waist circumference', 
                   'Elevated blood pressure', 'Impaired fasting glucose', 
                   'Elevated triglycerides', 'Decreased HDL-C']

    # exclude 필터링 후 dtype 변경
    category_cols = [c for c in category_cols if c in out.columns and c not in exclude]
    binary_cols = [c for c in binary_cols if c in out.columns and c not in exclude]
    
    out[category_cols] = out[category_cols].astype('category')
    out[binary_cols] = out[binary_cols].astype('category')
    
    if '생년월' in out.columns:
        out['생년월'] = pd.to_datetime(out['생년월'])
    
    return out


def ewma_exposure(
    df: pd.DataFrame,
    idcol: str,
    datecol: str,
    cols: List[str],
    halflife_days: float = 365.25 * 2,
    suffix: str = "_ewma",
) -> pd.DataFrame:
    """
    Exponentially weighted moving average per person for irregular intervals.
    Uses decay = exp(-ln(2) * delta_days / halflife_days).
    """
    out = df.copy()
    out[datecol] = pd.to_datetime(out[datecol], errors="coerce")
    out = out.sort_values([idcol, datecol]).reset_index(drop=True)
    for c in cols:
        if c not in out.columns:
            continue
        ewma_vals = []
        last = None
        last_date = None
        for _id, g in out.groupby(idcol, sort=False):
            prev = np.nan
            prev_date = None
            series = []
            for _, row in g.iterrows():
                x = row[c]
                d = row[datecol]
                if prev_date is None or pd.isna(prev):
                    prev = x
                else:
                    delta_days = (d - prev_date).days if pd.notna(d) and pd.notna(prev_date) else np.nan
                    if pd.isna(delta_days) or delta_days <= 0:
                        decay = 0.5  # fallback
                    else:
                        decay = float(np.exp(-np.log(2) * (delta_days / halflife_days)))
                    prev = decay * prev + (1 - decay) * x
                series.append(prev)
                prev_date = d
            ewma_vals.extend(series)
        out[f"{c}{suffix}"] = ewma_vals
    return out


def ensure_time_index(df: pd.DataFrame, idcol: str, datecol: str) -> pd.DataFrame:
    out = df.copy()
    if datecol in out.columns:
        out[datecol] = pd.to_datetime(out[datecol], errors="coerce")
        out = out.sort_values([idcol, datecol]).reset_index(drop=True)
    if "visit_index" not in out.columns:
        out["visit_index"] = out.groupby(idcol).cumcount()
    return out


def fit_gee_gaussian_ar1(
    df: pd.DataFrame,
    idcol: str,
    outcome: str,
    predictors: List[str],
    datecol: Optional[str] = None,
    add_constant: bool = True,
    drop_zero_variance_ids: bool = True,
    min_obs_per_id: int = 2,
    fallback_on_ar1_failure: bool = True,
    fallback_cov: str = "exchangeable",
):
    data = df.copy()
    data = ensure_time_index(data, idcol=idcol, datecol=datecol)
    cols = [c for c in predictors + [outcome, idcol, "visit_index"] if c in data.columns]
    data = data[cols].dropna()

    if not data.empty and min_obs_per_id > 1:
        counts = data.groupby(idcol)[idcol].transform("size")
        data = data.loc[counts >= min_obs_per_id].copy()

    if drop_zero_variance_ids and not data.empty:
        var_by_id = data.groupby(idcol)[outcome].var().fillna(0)
        keep_ids = var_by_id.index[var_by_id > 0]
        before, after = data[idcol].nunique(), keep_ids.size
        if after == 0:
            raise ValueError("All clusters have zero within-person variance in outcome; cannot fit AR(1) GEE.")
        if after < before:
            data = data[data[idcol].isin(keep_ids)].copy()

    if data.shape[0] == 0:
        raise ValueError("No rows remain after NA drop, min-obs, and zero-variance filtering for AR(1) GEE.")

    y = data[outcome].astype(float).to_numpy()
    X_df = data[[c for c in predictors if c in data.columns]]
    if add_constant:
        X_df = sm.add_constant(X_df, has_constant="add")
    
    variable_names = X_df.columns.tolist()  # 변수명 저장
    
    X = X_df.to_numpy(dtype=float, copy=False)
    groups = data[idcol].to_numpy()
    time = data["visit_index"].astype(int).to_numpy()
    
    model = sm.GEE(y, X, groups=groups, time=time, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Autoregressive())
    
    try:
        res = model.fit()
    except Exception as e:
        if fallback_on_ar1_failure:
            if fallback_cov.lower().startswith("exch"):
                covs = sm.cov_struct.Exchangeable()
            else:
                covs = sm.cov_struct.Independence()
            model2 = sm.GEE(y, X, groups=groups, time=time, family=sm.families.Gaussian(), cov_struct=covs)
            res = model2.fit()
        else:
            raise
    
    return res, variable_names


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def fit_gee_binomial(
    df: pd.DataFrame,
    idcol: str,
    outcome: str,
    predictors: List[str],
    add_constant: bool = True,
    auto_dummies: bool = True,
):
    present_preds = _ensure_columns(df, predictors)
    
    # 완전분리를 일으키는 변수들 자동 제외
    if outcome == '비만':
        present_preds = [p for p in present_preds if p not in ['체질량지수', '허리둘레(WAIST)']]
    elif outcome == 'Chronic kidney disease (eGFR<60)':
        present_preds = [p for p in present_preds if p != 'eGFR']
    
    # outcome과 idcol 포함하여 데이터 선택
    cols_needed = present_preds + [outcome, idcol]
    data = df[cols_needed].copy()

    cat_cols = ['성별', '일반담배_흡연여부', '음주', '간식빈도', '고지방 육류', '곡류', '과일', 
                '단맛', '단백질류', '물', '밥 양', '식사 빈도', '식사량', '외식빈도', '유제품', 
                '음료류', '인스턴트 가공식품', '짠 간', '짠 식습관', '채소', '커피', '튀김']

    # 존재하는 카테고리 컬럼만 선택 (outcome, idcol 제외)
    cat_cols = [c for c in cat_cols if c in present_preds]
    
    if cat_cols:
        dummies = pd.get_dummies(data[cat_cols], drop_first=True, dtype=float)
        data = pd.concat([data.drop(columns=cat_cols), dummies], axis=1)
    
    # predictors 업데이트 (outcome, idcol 제외)
    present_preds = [c for c in data.columns if c not in [outcome, idcol]]
    
    # 결측값 제거
    data = data.dropna()
    
    if data.shape[0] == 0:
        raise ValueError(f"No rows remaining after dropping NAs for {outcome}")

    # Build arrays
    y = data[outcome].astype(int).to_numpy()
    X_cols = [c for c in present_preds if c in data.columns]
    
    if len(X_cols) == 0:
        raise ValueError("Design matrix has zero predictors after preprocessing.")
    
    X_df = data[X_cols]

    # 상수형 변수 제거
    const_like = [c for c in X_df.columns if pd.to_numeric(X_df[c], errors='coerce').nunique(dropna=True) <= 1]
    if const_like:
        X_df = X_df.drop(columns=const_like)
        X_cols = [c for c in X_cols if c not in const_like]
    
    if len(X_cols) == 0:
        raise ValueError("All predictors are constant or invalid after filtering; cannot fit GEE.")

    if np.unique(y).size < 2:
        raise ValueError("Outcome has a single class after filtering; cannot fit binomial model.")
    
    if add_constant:
        X_df = sm.add_constant(X_df, has_constant="add")
    
    variable_names = X_df.columns.tolist()
    
    X = X_df.to_numpy(dtype=float, copy=False)
    if X.size == 0 or y.size == 0:
        raise ValueError("Empty design or response array. Check predictors and missingness.")

    groups = data[idcol].to_numpy()
    fam = sm.families.Binomial()
    cov_struct = sm.cov_struct.Exchangeable()
    model = sm.GEE(y, X, groups=groups, family=fam, cov_struct=cov_struct)
    res = model.fit()
    
    return res, variable_names

def extract_gee_results(model_result, outcome_name, model_type='gaussian'):
    try:
        # 계수, 표준오차, p값 추출
        coef = model_result.params
        se = model_result.bse
        pvalues = model_result.pvalues
        conf_int = model_result.conf_int()
        
        # 실제 변수명 가져오기 (모델에서 직접)
        if hasattr(model_result.model, 'exog_names') and model_result.model.exog_names is not None:
            var_names = model_result.model.exog_names
        elif hasattr(coef, 'index'):
            var_names = coef.index.tolist()
        else:
            var_names = [f'x{i}' for i in range(len(coef))]
        
        # 결과 DataFrame 생성
        results_df = pd.DataFrame({
            'Variable': var_names,
            'Coefficient': coef if isinstance(coef, np.ndarray) else coef.values,
            'Std_Error': se if isinstance(se, np.ndarray) else se.values,
            'P_value': pvalues if isinstance(pvalues, np.ndarray) else pvalues.values,
            'CI_Lower': conf_int.iloc[:, 0].values if hasattr(conf_int, 'iloc') else conf_int[:, 0],
            'CI_Upper': conf_int.iloc[:, 1].values if hasattr(conf_int, 'iloc') else conf_int[:, 1],
            'Outcome': outcome_name,
            'Model_Type': model_type
        })
        
        # 유의성 표시
        results_df['Significance'] = results_df['P_value'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        )
        
        return results_df
    except Exception as e:
        print(f"Error extracting results for {outcome_name}: {e}")
        return pd.DataFrame()