import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from data_import import DataPreprocessor, SingleDietImpactDataset, DataLoaderManager
import warnings
import random
from utils import *

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

class MultiDiseasePredictor(nn.Module):
    def __init__(self, diet_dim, demo_dim, life_dim, bio_dim, change_dim, inter_dim, pca_dim, 
                 disease_names, dropout_rate=0.4, l1_lambda=0, l2_lambda=0):
        super(MultiDiseasePredictor, self).__init__()
        
        self.disease_names = disease_names
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        hidden_dim = 8
        
        self.diet_encoders = nn.ModuleDict({
            disease: nn.Sequential(
                nn.Linear(diet_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for disease in disease_names
        })
        
        self.demo_encoders = nn.ModuleDict({
            disease: nn.Sequential(
                nn.Linear(demo_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for disease in disease_names
        })
        
        self.life_encoders = nn.ModuleDict({
            disease: nn.Sequential(
                nn.Linear(life_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for disease in disease_names
        })
        
        self.bio_encoders = nn.ModuleDict({
            disease: nn.Sequential(
                nn.Linear(bio_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for disease in disease_names
        })
        
        self.change_encoders = nn.ModuleDict({
            disease: nn.Sequential(
                nn.Linear(change_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for disease in disease_names
        })

        self.inter_encoders = nn.ModuleDict({
            disease: nn.Sequential(
                nn.Linear(inter_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for disease in disease_names
        })
        
        self.pca_encoders = nn.ModuleDict({
            disease: nn.Sequential(
                nn.Linear(pca_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for disease in disease_names
        })
        
        combined_dim = hidden_dim * 7  # 7개 컴포넌트
        self.heads = nn.ModuleDict({
            disease: nn.Sequential(
                nn.Linear(combined_dim, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(16, 3)
            ) for disease in disease_names
        })
        
    def forward(self, diet_data, demo_data, life_data, bio_data, change_data, inter_data, pca_data, disease_name):
        d = self.diet_encoders[disease_name](diet_data)
        demo = self.demo_encoders[disease_name](demo_data)
        life = self.life_encoders[disease_name](life_data)
        b = self.bio_encoders[disease_name](bio_data)
        c = self.change_encoders[disease_name](change_data)
        inter = self.inter_encoders[disease_name](inter_data)
        pca = self.pca_encoders[disease_name](pca_data)
        
        concat = torch.cat([d, demo, life, b, c, inter, pca], dim=1)
        disease_logits = self.heads[disease_name](concat)
        
        return {'disease_logits': disease_logits}
    
    def regularization_loss(self):
        l1_loss = sum(p.abs().sum() for p in self.parameters())
        l2_loss = sum(p.pow(2).sum() for p in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss
  
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

def train_sklearn_ensemble(train_df, val_df, test_df, disease_name):
    target_col = f'{disease_name}_delta'
    
    feature_cols = [col for col in train_df.columns if col not in target_col]
    full_train = pd.concat([train_df, val_df], ignore_index=True)
    
    X_train = full_train[feature_cols].fillna(0).values
    y_train = full_train[target_col].values
    
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df[target_col].values
    
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    print(f"    라벨: {unique_labels}")
    
    # 앙상블 모델들
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1, max_depth=10)
    lr = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')
    dt = DecisionTreeClassifier(random_state=42, max_depth=10)
    
    voting_ensemble = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr),  ('dt', dt)], 
        voting='soft'
    )
    voting_ensemble.fit(X_train, y_train)
    
    y_pred = voting_ensemble.predict(X_test)
    y_prob = voting_ensemble.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    roc_aucs = {}
    pr_aucs = {}
    class_names = ['개선', '유지', '악화']
    
    n_classes = y_prob.shape[1]    
    for i in range(n_classes):
        class_name = class_names[i]
        y_true_binary = (y_test == i).astype(int)
        
        y_prob_class = y_prob[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob_class)
        roc_auc = auc(fpr, tpr)
        roc_aucs[class_name] = roc_auc
        
        precision, recall, _ = precision_recall_curve(y_true_binary, y_prob_class)
        pr_auc = auc(recall, precision)
        pr_aucs[class_name] = pr_auc
    
    roc_aucs['Macro'] = np.mean(list(roc_aucs.values()))
    pr_aucs['Macro'] = np.mean(list(pr_aucs.values()))
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_aucs': roc_aucs,
        'pr_aucs': pr_aucs,
        'method': 'sklearn_ensemble'
    }
