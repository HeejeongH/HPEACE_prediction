import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from data_import import DataPreprocessor, DataLoaderManager
from feature_engineering import apply_to_all_data
from resampling import create_resampled_dataset
import pickle
import glob
from data_import import SingleDietImpactDataset
from utils import create_model_with_dynamic_dims, mets_cols

plt.rcParams['font.family'] = 'DejaVu Sans'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_existing_shap_results():
    result_pattern = '../result/shap_analysis/*/shap_results.pkl'
    existing_files = glob.glob(result_pattern)
    
    if existing_files:
        latest_file = max(existing_files, key=os.path.getctime)
        print(f"기존 SHAP 결과 발견: {latest_file}")
        
        try:
            with open(latest_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            print("기존 SHAP 결과를 성공적으로 로드했습니다.")
            return cached_data['results'], cached_data['category_importance'], cached_data['feature_names']
        
        except Exception as e:
            print(f"기존 결과 로드 실패: {e}")
            print("새로 SHAP 분석을 수행합니다.")
            return None, None, None
    
    else:
        print("기존 SHAP 결과 없음. 새로 분석을 수행합니다.")
        return None, None, None

def load_trained_model_and_data(best_combo, best_params, actual_change_dim):
    sel_strategy, mod_method, resample_method, loss_method = best_combo

    data_processor = DataPreprocessor('../data/total_again.xlsx')
    train_df, val_df, test_df, _ = data_processor.process_all(normalize=True, selection_strategy=sel_strategy)    

    train_df, val_df, test_df, actual_change_dim, _pca = apply_to_all_data(train_df, val_df, test_df)
    
    batch_size = 16
    dropout_rate = best_params['dropout_rate']    
    l1_lambda = best_params['l1_lambda']
    l2_lambda = best_params['l2_lambda']
    
    if resample_method == 'none':
        train_loaders, val_loaders, test_loaders = DataLoaderManager.create_disease_loaders(
            train_df, val_df, test_df, batch_size=batch_size)
    else:
        resampled_train = create_resampled_dataset(train_df, resample_method)
        train_loaders = {disease: torch.utils.data.DataLoader(
            torch.utils.data.Subset(resampled_train, 
                [i for i, item in enumerate(resampled_train) if item['disease_name'] == disease]), 
            batch_size=batch_size, shuffle=True, drop_last=True) for disease in mets_cols}
        
        _, val_loaders, test_loaders = DataLoaderManager.create_disease_loaders(
            train_df, val_df, test_df, batch_size=batch_size)    

    dataset = SingleDietImpactDataset(train_df)
    model = create_model_with_dynamic_dims(dataset, mets_cols, dropout_rate, l1_lambda, l2_lambda).to(device)     


    return model, train_df, val_df, test_df, train_loaders, val_loaders, test_loaders, mets_cols

def extract_features_and_targets(dataset, disease_name):
    features = []
    targets = []
    
    for batch in dataset:
        combined = np.concatenate([
            batch['diet'].numpy(),
            batch['demo'].numpy(),
            batch['life'].numpy(), 
            batch['bio'].numpy(),
            batch['delta'].numpy(),
            batch['interaction'].numpy(),
            batch['pca'].numpy()
        ], axis=1)
        
        disease_mask = [item == disease_name for item in batch['disease_name']]
        if any(disease_mask):
            features.append(combined[disease_mask])
            targets.append(batch['target'].numpy()[disease_mask])
    
    if features:
        return np.vstack(features), np.concatenate(targets)
    return np.empty((0, 80)), np.array([])

def create_model_wrapper(model, disease_name, train_dataset):
    if hasattr(train_dataset, 'dims'):
        dims = train_dataset.dims
    else:
        sample_item = train_dataset[0]
        dims = {
            'diet': len(sample_item['diet']),
            'demo': len(sample_item['demo']),
            'life': len(sample_item['life']),
            'bio': len(sample_item['bio']),
            'delta': len(sample_item['delta']),
            'interaction': len(sample_item['interaction']),
            'pca': len(sample_item['pca'])
        }
        
    def predict_proba(X):
        model.eval()
        probabilities = []
        
        with torch.no_grad():
            for i in range(len(X)):
                features = torch.FloatTensor(X[i:i+1]).to(device)
                
                start_idx = 0
                
                diet_data = features[:, start_idx:start_idx+dims['diet']]
                start_idx += dims['diet']
                
                demo_data = features[:, start_idx:start_idx+dims['demo']]
                start_idx += dims['demo']
                
                life_data = features[:, start_idx:start_idx+dims['life']]
                start_idx += dims['life']
                
                bio_data = features[:, start_idx:start_idx+dims['bio']]
                start_idx += dims['bio']
                
                delta_data = features[:, start_idx:start_idx+dims['delta']]
                start_idx += dims['delta']
                
                interaction_data = features[:, start_idx:start_idx+dims['interaction']]
                start_idx += dims['interaction']
                
                pca_data = features[:, start_idx:start_idx+dims['pca']]
                
                output = model(diet_data, demo_data, life_data, bio_data, delta_data, 
                             interaction_data, pca_data, disease_name)
                prob = torch.softmax(output['disease_logits'], dim=1)
                probabilities.append(prob.cpu().numpy()[0])
        
        return np.array(probabilities)
    
    return predict_proba

