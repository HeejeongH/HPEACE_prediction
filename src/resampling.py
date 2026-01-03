from data_import import SingleDietImpactDataset, DataLoaderManager
from torch.utils.data import Dataset
import numpy as np
import torch

def create_resampled_dataset(df, imbmodel='none'):
    base_dataset = SingleDietImpactDataset(df)    
    if imbmodel == 'none':
        return base_dataset
    
    dims = base_dataset.dims
    print(f"리샘플링에 사용할 차원: {dims}")
    
    resampled_pairs = []
    disease_datasets = DataLoaderManager.split_dataset_by_disease(base_dataset)
    
    for disease_name, disease_subset in disease_datasets.items():
        X_list = []
        y_list = []
        
        for idx in range(len(disease_subset)):
            item = disease_subset[idx]
            features = np.concatenate([
                item['diet'].numpy(),
                item['demo'].numpy(),
                item['life'].numpy(),
                item['bio'].numpy(),
                item['delta'].numpy(),
                item['interaction'].numpy(),
                item['pca'].numpy()
            ])
            X_list.append(features)
            y_list.append(item['target'].numpy()[0])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 리샘플링 적용
        if imbmodel == 'smote':
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=42, k_neighbors=3)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        for i in range(len(X_resampled)):
            start_idx = 0
            
            diet_data = X_resampled[i][start_idx:start_idx+dims['diet']]
            start_idx += dims['diet']
            
            demo_data = X_resampled[i][start_idx:start_idx+dims['demo']]
            start_idx += dims['demo']
            
            life_data = X_resampled[i][start_idx:start_idx+dims['life']]
            start_idx += dims['life']
            
            bio_data = X_resampled[i][start_idx:start_idx+dims['bio']]
            start_idx += dims['bio']
            
            delta_data = X_resampled[i][start_idx:start_idx+dims['delta']]
            start_idx += dims['delta']
            
            interaction_data = X_resampled[i][start_idx:start_idx+dims['interaction']]
            start_idx += dims['interaction']
            
            pca_data = X_resampled[i][start_idx:start_idx+dims['pca']]
            
            resampled_pairs.append({
                'diet': diet_data.astype(np.float32),
                'demo': demo_data.astype(np.float32),
                'life': life_data.astype(np.float32),
                'bio': bio_data.astype(np.float32),
                'delta': delta_data.astype(np.float32),
                'interaction': interaction_data.astype(np.float32),
                'pca': pca_data.astype(np.float32),
                'target': np.array([y_resampled[i]]).astype(np.int64),
                'disease_name': disease_name
            })
    
    class ResampledDataset(Dataset):
        def __init__(self, pairs, dims):
            self.pairs = pairs
            self.dims = dims
            
        def __len__(self):
            return len(self.pairs)
            
        def __getitem__(self, idx):
            item = self.pairs[idx]
            return {k: torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in item.items()}
    
    return ResampledDataset(resampled_pairs, dims)