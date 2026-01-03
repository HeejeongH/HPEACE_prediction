import pandas as pd
import warnings
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, file_path, seed=42, normalize=False):
        self.file_path = file_path
        self.seed = seed
        self.normalize = normalize
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 컬럼 정의
        self.demo_cols = ['수진일', '성별', '나이', '신장']  # 기본 인구통계학적 변수
        self.lifestyle_cols = ['흡연', '활동량', '음주']  # 생활습관 변수
        self.ffq_cols = ['간식빈도', '고지방 육류', '곡류', '과일', '단맛', '단백질류', '물', '밥 양','커피', '튀김',
                        '식사 빈도', '식사량', '외식빈도', '유제품', '음료류', '인스턴트 가공식품', '짠 간', '짠 식습관', '채소']
        self.biomarker_cols = ['SBP', 'DBP', 'WAIST', '체중', 'BMI', 'GLUCOSE', 'HBA1C', 'TG', 'HDL CHOL', 'LDL CHOL', 'eGFR']
        self.mets_cols = ['Increased waist circumference', 'Elevated blood pressure', 'Impaired fasting glucose', 'Elevated triglycerides', 'Decreased HDL-C']
        
        self._set_seed()
        print(f"Device: {self.device}")
    
    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def load_and_preprocess_data(self):
        df = pd.read_excel(self.file_path, index_col='R-ID').drop('Unnamed: 0', axis=1)
        df['수진일'] = pd.to_datetime(df['수진일'])
        df = df.sort_values(['R-ID', '수진일'])
        
        df = self._rename_columns(df)
        df = self._create_mets_criteria(df)
        df = self._handle_missing_values(df)
        
        return df
    
    def _rename_columns(self, df):
        df.rename(columns={
            '고혈압_통합':'Hypertension',
            '당뇨_통합':'Diabetes',
            '고지혈증_통합':'Dyslipidemia',
            '협심증/심근경색증_통합':'Cardiovascular',
            '뇌졸중(중풍)_통합':'Stroke',
            'Chronic kidney disease (eGFR<60)':'CKD',
            '일반담배_흡연여부':'흡연',
            '허리둘레(WAIST)':'WAIST',
            '체질량지수': 'BMI',
            'LDL CHOL.': 'LDL CHOL',
            'HDL CHOL.': 'HDL CHOL'
        }, inplace=True)
        
        df['Obesity'] = np.where(df['비만'] > 1, 1, 0).astype(int)
        return df
    
    def _create_mets_criteria(self, df):
        df['Increased waist circumference'] = (
            ((df['성별'] == 'M') & (df['WAIST'].astype(float) >= 90)) | 
            ((df['성별'] == 'F') & (df['WAIST'].astype(float) >= 85))
        )
        df['Elevated blood pressure'] = (
            ((df['SBP'].astype(float) >= 130) | 
             (df['DBP'].astype(float) >= 85)) | 
            (df['고혈압_투약여부'] == 1)
        )
        df['Impaired fasting glucose'] = (
            (df['GLUCOSE'].astype(float) >= 100) | 
            (df['당뇨_투약여부'] == 1)
        )
        df['Elevated triglycerides'] = (
            (df['TG'].astype(float) >= 150) | 
            (df['고지혈증_투약여부'] == 1)
        )
        df['Decreased HDL-C'] = (
            ((df['성별'] == 'M') & (df['HDL CHOL'].astype(float) < 40)) | 
            ((df['성별'] == 'F') & (df['HDL CHOL'].astype(float) < 50))
        )
        
        df[self.mets_cols] = df[self.mets_cols].astype(int)
        return df
    
    def _handle_missing_values(self, df):
        df['흡연'] = df['흡연'].replace('Missing value', 0)
        df['활동량'] = df['활동량'].replace('Missing value', 1)
        df['음주'] = df['음주'].replace('Missing value', 1)
        df['성별'] = df['성별'].replace('M', 0).replace('F', 1)
        return df
    
    def create_optimal_delta_features(self, df, selection_strategy='max_time_span'):
        optimal_data = []
        
        for patient_id, patient_data in df.groupby('R-ID'):
            patient_data = patient_data.sort_values('수진일')
            
            if len(patient_data) < 2:
                continue
                
            if selection_strategy == 'max_disease_change':
                best_pair = self._select_max_disease_change_pair(patient_data)
            else:
                best_pair = self._select_first_last_pair(patient_data)
            
            if best_pair:
                optimal_data.append(self._create_delta_row(best_pair, patient_id))
        
        return pd.DataFrame(optimal_data)

    def _select_max_disease_change_pair(self, patient_data):
        max_disease_change = 0
        best_pair = None
        
        for i in range(len(patient_data)):
            for j in range(i+1, len(patient_data)):
                visit1 = patient_data.iloc[i]
                visit2 = patient_data.iloc[j]
                
                disease_changes = 0
                for col in self.mets_cols:
                    if col in patient_data.columns:
                        disease_changes += abs(visit2[col] - visit1[col])
                
                if disease_changes > max_disease_change:
                    max_disease_change = disease_changes
                    best_pair = (visit1, visit2)
        
        return best_pair if best_pair else self._select_first_last_pair(patient_data)

    def _select_first_last_pair(self, patient_data):
        first_visit = patient_data.iloc[0]
        last_visit = patient_data.iloc[-1]
        return (first_visit, last_visit)

    def _create_delta_row(self, pair, patient_id):
        visit1, visit2 = pair
        days_between = (visit2['수진일'] - visit1['수진일']).days
        
        row = {'patient_id': patient_id, 'days_between': days_between}
        
        # 기준시점 데이터 추가
        for col in self.demo_cols[1:] + self.lifestyle_cols + self.ffq_cols + self.biomarker_cols:
            if col in visit1.index:
                row[f'{col}_T0'] = visit1[col]
        
        # 델타 데이터 추가
        for col in self.ffq_cols + self.lifestyle_cols:
            if col in visit1.index and col in visit2.index:
                row[f'{col}_delta'] = visit2[col] - visit1[col]
        
        # 질병 상태 변화 (3클래스)
        for col in self.mets_cols:
            if col in visit1.index and col in visit2.index:
                change = visit2[col] - visit1[col]
                if change == -1:    # 개선
                    target_class = 0
                elif change == 0:   # 유지
                    target_class = 1
                else:               # 악화
                    target_class = 2
                row[f'{col}_delta'] = target_class
        
        return row

    def split_data(self, df, test_size=0.2, val_size=0.5):
        print(f"Total unique patients: {df['patient_id'].nunique()}")
        unique_patient_ids = df['patient_id'].unique()
        
        train_ids, temp_ids = train_test_split(unique_patient_ids, test_size=test_size, random_state=self.seed)
        val_ids, test_ids = train_test_split(temp_ids, test_size=val_size, random_state=self.seed)
        
        train_df = df[df['patient_id'].isin(train_ids)].reset_index(drop=True).drop('patient_id', axis=1)
        val_df = df[df['patient_id'].isin(val_ids)].reset_index(drop=True).drop('patient_id', axis=1)
        test_df = df[df['patient_id'].isin(test_ids)].reset_index(drop=True).drop('patient_id', axis=1)
        
        return train_df, val_df, test_df
    
    def normalize_and_encode(self, train_df, val_df, test_df, cont_cols=True):
        if cont_cols == True:
            continuous_candidates = ['days_between', '나이_T0', '신장_T0'] + [f'{col}_T0' for col in self.biomarker_cols] + [f'{col}_delta' for col in self.ffq_cols]
            continuous_cols = [col for col in continuous_candidates if col in train_df.columns]
            
            if continuous_cols:  # 연속형 컬럼이 있을 때만 정규화
                train_df[continuous_cols] = self.scaler.fit_transform(train_df[continuous_cols])
                val_df[continuous_cols] = self.scaler.transform(val_df[continuous_cols])
                test_df[continuous_cols] = self.scaler.transform(test_df[continuous_cols])
        
        ordinal_candidates = [f'{col}_T0' for col in self.ffq_cols] + ['흡연_T0', '활동량_T0', '음주_T0']
        ordinal_cols = [col for col in ordinal_candidates if col in train_df.columns]
        
        for col in ordinal_cols:
            if col in train_df.columns:
                le = LabelEncoder()
                train_df[col] = le.fit_transform(train_df[col].astype(str))
                self.label_encoders[col] = le
                
                for df_split in [val_df, test_df]:
                    try:
                        if col in df_split.columns:
                            df_split[col] = le.transform(df_split[col].astype(str))
                    except ValueError:
                        if len(train_df[col]) > 0: 
                            mode_val = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 0
                            df_split[col] = df_split[col].astype(str)
                            df_split[col] = df_split[col].apply(lambda x: mode_val if x not in le.classes_ else le.transform([x])[0])
                        else:
                            df_split[col] = 0  # 기본값
        
        all_columns = list(set(train_df.columns) | set(val_df.columns) | set(test_df.columns))
        for df_split in [train_df, val_df, test_df]:
            for col in all_columns:
                if col not in df_split.columns:
                    df_split[col] = 0
        
        train_df = train_df[all_columns]
        val_df = val_df[all_columns]
        test_df = test_df[all_columns]
        
        return train_df, val_df, test_df
    
    def get_feature_columns(self, df):
        available_cols = df.columns.tolist()
        ffq_baseline_cols = [col for col in available_cols if '_T0' in col and any(ffq in col for ffq in self.ffq_cols)]
        demo_baseline_cols = ['days_between'] + [col for col in available_cols if any(demo in col for demo in self.demo_cols + self.lifestyle_cols)]
        bio_baseline_cols = [col for col in available_cols if '_T0' in col and any(bio in col for bio in self.biomarker_cols)]
        delta_cols = [col for col in available_cols if '_delta' in col and any(item in col for item in self.ffq_cols + self.lifestyle_cols)]
        
        return {
            'ffq_baseline': ffq_baseline_cols,
            'demo_baseline': demo_baseline_cols,
            'bio_baseline': bio_baseline_cols,
            'delta': delta_cols,
            'all_features': ffq_baseline_cols + demo_baseline_cols + bio_baseline_cols + delta_cols
        }
    
    def process_all(self, normalize=True, selection_strategy='first_last'):
        df = self.load_and_preprocess_data()
        print(f"Initial data shape: {df.shape}")
        
        if selection_strategy == 'max_disease_change':
            delta_df = self.create_optimal_delta_features(df, 'max_disease_change')
            print(f"Data shape after max disease change selection: {delta_df.shape}")
        else:
            delta_df = self.create_optimal_delta_features(df, 'max_time_span')
        
        train_df, val_df, test_df = self.split_data(delta_df)
        if normalize:
            train_df, val_df, test_df = self.normalize_and_encode(train_df, val_df, test_df)
        else:
            train_df, val_df, test_df = self.normalize_and_encode(train_df, val_df, test_df, cont_cols=False)
        
        feature_cols = self.get_feature_columns(train_df)
        
        return train_df, val_df, test_df, feature_cols

class SingleDietImpactDataset(Dataset):
    def __init__(self, df):
        self.df = df.copy()
        
        # utils에서 정의된 순서대로 컬럼 정렬 및 검증
        from utils import demo_cols, life_cols, bio_cols, diet_cols, disease_delta_cols, interaction_cols
        
        # 1단계: 존재하는 컬럼만 필터링 (순서 유지)
        self.demo_cols = [col for col in demo_cols if col in df.columns]
        self.life_cols = [col for col in life_cols if col in df.columns]
        self.bio_cols = [col for col in bio_cols if col in df.columns]
        self.diet_cols = [col for col in diet_cols if col in df.columns]
        self.disease_delta_cols = [col for col in disease_delta_cols if col in df.columns]
        self.interaction_cols = [col for col in interaction_cols if col in df.columns]
        
        # PCA와 delta 컬럼들은 동적으로 찾되 정렬해서 순서 고정
        self.pca_cols = sorted([col for col in df.columns if 'pca' in col])
        self.delta_cols = sorted([col for col in df.columns if '_delta' in col and col not in self.disease_delta_cols])
        
        # 차원 정보 저장 (나중에 인덱싱할 때 사용)
        self.dims = {
            'diet': len(self.diet_cols),
            'demo': len(self.demo_cols), 
            'life': len(self.life_cols),
            'bio': len(self.bio_cols),
            'delta': len(self.delta_cols),
            'interaction': len(self.interaction_cols),
            'pca': len(self.pca_cols)
        }
        
        print(f"컬럼 순서 고정 완료 - diet:{self.dims['diet']}, demo:{self.dims['demo']}, life:{self.dims['life']}, bio:{self.dims['bio']}")
        
        self.pairs = self._create_pairs()
    
    def _create_pairs(self):
        pairs = []
        for idx, row in self.df.iterrows():
            for disease_col in self.disease_delta_cols:
                disease_name = disease_col.replace('_delta', '')
                target_class = row[disease_col]
                
                pairs.append({
                    'diet': row[self.diet_cols].values.astype(np.float32),
                    'demo': row[self.demo_cols].values.astype(np.float32),
                    'life': row[self.life_cols].values.astype(np.float32),
                    'bio': row[self.bio_cols].values.astype(np.float32),
                    'delta': row[self.delta_cols].values.astype(np.float32),
                    'interaction': row[self.interaction_cols].values.astype(np.float32),
                    'pca': row[self.pca_cols].values.astype(np.float32),
                    'target': np.array([target_class]).astype(np.int64),
                    'disease_name': disease_name
                })
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        return {k: torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in item.items()}

class DataLoaderManager:
    @staticmethod
    def split_dataset_by_disease(dataset):
        disease_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            disease = dataset[idx]['disease_name']
            disease_to_indices[disease].append(idx)

        disease_datasets = {
            disease: Subset(dataset, indices)
            for disease, indices in disease_to_indices.items()
        }
        return disease_datasets
    
    @staticmethod
    def create_disease_loaders(train_df, val_df, test_df, batch_size=32):
        train_dataset = SingleDietImpactDataset(train_df)
        val_dataset = SingleDietImpactDataset(val_df)
        test_dataset = SingleDietImpactDataset(test_df)

        train_disease_datasets = DataLoaderManager.split_dataset_by_disease(train_dataset)
        val_disease_datasets = DataLoaderManager.split_dataset_by_disease(val_dataset)
        test_disease_datasets = DataLoaderManager.split_dataset_by_disease(test_dataset)

        train_disease_loaders = {
            disease: DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
            for disease, ds in train_disease_datasets.items()
        }

        val_disease_loaders = {
            disease: DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)
            for disease, ds in val_disease_datasets.items()
        }

        test_disease_loaders = {
            disease: DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)
            for disease, ds in test_disease_datasets.items()
        }        
        
        return train_disease_loaders, val_disease_loaders, test_disease_loaders
    