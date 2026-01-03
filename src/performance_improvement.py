"""
성능 개선 실험 스크립트
- Loss 함수별 성능 비교
- 클래스 불균형 해결 전략
- 앙상블 기법 강화
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 기존 모듈 import
from data_import import DataPreprocessor, DataLoaderManager
from feature_engineering import FeatureEngineer
from MetS_prediction_model import MultiDiseasePredictor, EarlyStopping
from train_eval_function import train_model_with_loss, evaluate_model_custom
from loss_functions import FocalLoss, WeightedAsymmetricLoss, calculate_class_weights
from resampling import prepare_balanced_data
from utils import set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
set_seed(42)

class ImprovedLossFunction:
    """개선된 손실 함수 클래스"""
    
    @staticmethod
    def balanced_cross_entropy(train_loader, device):
        """동적 클래스 가중치 계산"""
        class_counts = torch.zeros(3)
        
        for batch in train_loader:
            targets = batch['target'].flatten()
            for t in targets:
                class_counts[t] += 1
        
        total = class_counts.sum()
        # 역빈도 가중치 (더 강한 불균형 보정)
        class_weights = total / (3 * class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * 3  # 정규화
        
        print(f"클래스 분포: {class_counts.numpy()}")
        print(f"클래스 가중치: {class_weights.numpy()}")
        
        return nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    @staticmethod
    def improved_focal_loss(alpha=None, gamma=2.0):
        """강화된 Focal Loss (gamma 증가)"""
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    @staticmethod
    def combined_loss(train_loader, device, ce_weight=0.5, focal_weight=0.5):
        """CE + Focal Loss 조합"""
        ce_loss = ImprovedLossFunction.balanced_cross_entropy(train_loader, device)
        focal_loss = ImprovedLossFunction.improved_focal_loss(gamma=2.0)
        
        class CombinedLoss(nn.Module):
            def forward(self, inputs, targets):
                ce = ce_loss(inputs, targets)
                focal = focal_loss(inputs, targets)
                return ce_weight * ce + focal_weight * focal
        
        return CombinedLoss()


class AdvancedEnsemble:
    """고급 앙상블 기법"""
    
    def __init__(self, base_models=None):
        self.base_models = base_models or [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=2000, multi_class='multinomial'))
        ]
    
    def create_voting_ensemble(self, voting='soft'):
        """Soft/Hard Voting 앙상블"""
        return VotingClassifier(estimators=self.base_models, voting=voting, n_jobs=-1)
    
    def create_stacking_ensemble(self):
        """Stacking 앙상블"""
        return StackingClassifier(
            estimators=self.base_models,
            final_estimator=LogisticRegression(multi_class='multinomial', random_state=42),
            cv=5,
            n_jobs=-1
        )
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, ensemble_type='voting'):
        """앙상블 학습 및 평가"""
        if ensemble_type == 'voting':
            model = self.create_voting_ensemble()
        elif ensemble_type == 'stacking':
            model = self.create_stacking_ensemble()
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        print(f"Training {ensemble_type} ensemble...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # ROC AUC (OvR)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        
        print(f"\n=== {ensemble_type.upper()} Ensemble Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Macro): {f1:.4f}")
        print(f"ROC AUC (Macro): {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['개선', '유지', '악화']))
        
        return {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_prob
        }


class PerformanceExperiment:
    """성능 개선 실험 관리 클래스"""
    
    def __init__(self, data_path, disease_name='mets_transition'):
        self.data_path = data_path
        self.disease_name = disease_name
        self.results = {}
        self.device = device
        
        # 결과 저장 디렉토리
        self.result_dir = '../result/performance_improvement'
        os.makedirs(self.result_dir, exist_ok=True)
        
    def prepare_data(self, selection_method='max_disease_change', resample_method='smote'):
        """데이터 준비"""
        print(f"\n{'='*60}")
        print(f"데이터 준비 중...")
        print(f"선택 방법: {selection_method}, 리샘플링: {resample_method}")
        print(f"{'='*60}\n")
        
        # 데이터 로드
        preprocessor = DataPreprocessor(self.data_path, seed=42)
        df = preprocessor.load_and_preprocess_data()
        
        print(f"초기 데이터 형태: {df.shape}")
        
        # 피처 엔지니어링
        engineer = FeatureEngineer(preprocessor)
        df = engineer.create_interaction_features(df)
        df = engineer.create_pca_features(df)
        
        # 데이터 선택
        if selection_method == 'max_disease_change':
            df = preprocessor.select_max_disease_change_per_patient(df)
        
        print(f"선택 후 데이터 형태: {df.shape}")
        
        # 데이터 분할
        from sklearn.model_selection import train_test_split
        
        target_col = f'{self.disease_name}'
        feature_cols = [col for col in df.columns if col != target_col]
        
        train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df[target_col])
        train_df, val_df = train_test_split(train_val_df, test_size=0.15/(1-0.15), random_state=42, stratify=train_val_df[target_col])
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # 리샘플링 적용
        if resample_method == 'smote':
            train_df, val_df, test_df = prepare_balanced_data(
                train_df, val_df, test_df, 
                self.disease_name, 
                method='smote'
            )
            print(f"SMOTE 후 Train: {len(train_df)}")
        
        # DataLoader 생성
        loader_manager = DataLoaderManager(
            train_df, val_df, test_df,
            disease_names=[self.disease_name],
            batch_size=64,
            feature_engineer=engineer
        )
        
        train_loader = loader_manager.train_loaders[self.disease_name]
        val_loader = loader_manager.val_loaders[self.disease_name]
        test_loader = loader_manager.test_loaders[self.disease_name]
        
        # 차원 정보
        self.dims = {
            'diet_dim': len(preprocessor.ffq_cols),
            'demo_dim': len(preprocessor.demo_cols),
            'life_dim': len(preprocessor.lifestyle_cols),
            'bio_dim': len(preprocessor.biomarker_cols),
            'change_dim': len([col for col in df.columns if '_delta' in col and col != target_col]),
            'inter_dim': engineer.n_interaction_features,
            'pca_dim': engineer.n_pca_features
        }
        
        print(f"\n차원 정보: {self.dims}")
        
        return train_loader, val_loader, test_loader, train_df, val_df, test_df
    
    def experiment_loss_functions(self, train_loader, val_loader, test_loader):
        """실험 1: Loss 함수별 성능 비교"""
        print(f"\n{'='*60}")
        print(f"실험 1: Loss 함수별 성능 비교")
        print(f"{'='*60}\n")
        
        loss_configs = {
            'CrossEntropy': nn.CrossEntropyLoss(),
            'Balanced_CE': ImprovedLossFunction.balanced_cross_entropy(train_loader, self.device),
            'FocalLoss_gamma1.5': FocalLoss(gamma=1.5),
            'FocalLoss_gamma2.0': FocalLoss(gamma=2.0),
            'FocalLoss_gamma2.5': FocalLoss(gamma=2.5),
            'Combined_CE_Focal': ImprovedLossFunction.combined_loss(train_loader, self.device, 0.5, 0.5),
        }
        
        results = {}
        
        for loss_name, criterion in loss_configs.items():
            print(f"\n--- {loss_name} ---")
            
            # 모델 초기화
            model = MultiDiseasePredictor(
                **self.dims,
                disease_names=[self.disease_name],
                dropout_rate=0.357,
                l1_lambda=0.000366,
                l2_lambda=0.000596
            ).to(self.device)
            
            # 학습
            result = self._train_single_model(model, train_loader, val_loader, test_loader, criterion, loss_name)
            results[loss_name] = result
            
            print(f"F1 Score: {result['evaluation']['f1_score']:.4f}")
            print(f"Accuracy: {result['evaluation']['accuracy']:.4f}")
        
        self.results['loss_functions'] = results
        return results
    
    def _train_single_model(self, model, train_loader, val_loader, test_loader, criterion, loss_name):
        """단일 모델 학습 (커스텀 criterion 사용)"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=6e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=15, min_delta=0.001)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(100):
            # Train
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                diet_data = batch['diet'].to(self.device)
                demo_data = batch['demo'].to(self.device)
                life_data = batch['life'].to(self.device)
                bio_data = batch['bio'].to(self.device)
                change_data = batch['delta'].to(self.device)
                inter_data = batch['interaction'].to(self.device)
                pca_data = batch['pca'].to(self.device)
                target = batch['target'].long().to(self.device).squeeze()
                
                outputs = model(diet_data, demo_data, life_data, bio_data, change_data, inter_data, pca_data, self.disease_name)
                loss = criterion(outputs['disease_logits'], target) + model.regularization_loss()
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            train_loss = total_loss / batch_count
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_total_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    diet_data = batch['diet'].to(self.device)
                    demo_data = batch['demo'].to(self.device)
                    life_data = batch['life'].to(self.device)
                    bio_data = batch['bio'].to(self.device)
                    change_data = batch['delta'].to(self.device)
                    inter_data = batch['interaction'].to(self.device)
                    pca_data = batch['pca'].to(self.device)
                    target = batch['target'].long().to(self.device).squeeze()
                    
                    outputs = model(diet_data, demo_data, life_data, bio_data, change_data, inter_data, pca_data, self.disease_name)
                    val_loss = criterion(outputs['disease_logits'], target)
                    val_total_loss += val_loss.item()
                    val_batch_count += 1
            
            val_loss_avg = val_total_loss / val_batch_count
            val_losses.append(val_loss_avg)
            
            scheduler.step(val_loss_avg)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss_avg:.4f}")
            
            if early_stopping(val_loss_avg, model):
                print(f"Early stopping at epoch {epoch}")
                break
        
        early_stopping.load_best_model(model)
        
        # 평가
        accuracy, f1, roc_aucs, pr_aucs = evaluate_model_custom(model, test_loader, self.device, self.disease_name, 'deep')
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'evaluation': {
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_aucs': roc_aucs,
                'pr_aucs': pr_aucs
            }
        }
    
    def experiment_ensemble(self, train_df, val_df, test_df):
        """실험 2: 앙상블 기법 비교"""
        print(f"\n{'='*60}")
        print(f"실험 2: 앙상블 기법 비교")
        print(f"{'='*60}\n")
        
        target_col = self.disease_name
        feature_cols = [col for col in train_df.columns if col != target_col]
        
        # 데이터 준비
        full_train = pd.concat([train_df, val_df], ignore_index=True)
        X_train = full_train[feature_cols].fillna(0).values
        y_train = full_train[target_col].values
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df[target_col].values
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"클래스 분포 - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        ensemble = AdvancedEnsemble()
        
        results = {}
        
        # Soft Voting
        results['soft_voting'] = ensemble.train_and_evaluate(
            X_train, y_train, X_test, y_test, ensemble_type='voting'
        )
        
        # Stacking
        results['stacking'] = ensemble.train_and_evaluate(
            X_train, y_train, X_test, y_test, ensemble_type='stacking'
        )
        
        self.results['ensemble'] = results
        return results
    
    def save_results(self):
        """결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 요약 결과 생성
        summary = []
        
        # Loss 함수 결과
        if 'loss_functions' in self.results:
            for loss_name, result in self.results['loss_functions'].items():
                summary.append({
                    'experiment': 'Loss Function',
                    'method': loss_name,
                    'accuracy': result['evaluation']['accuracy'],
                    'f1_score': result['evaluation']['f1_score'],
                    'roc_auc_macro': result['evaluation']['roc_aucs'].get('Macro', 0),
                    'pr_auc_macro': result['evaluation']['pr_aucs'].get('Macro', 0)
                })
        
        # 앙상블 결과
        if 'ensemble' in self.results:
            for ensemble_name, result in self.results['ensemble'].items():
                summary.append({
                    'experiment': 'Ensemble',
                    'method': ensemble_name,
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score'],
                    'roc_auc': result['roc_auc'],
                    'pr_auc_macro': 0  # 앙상블은 PR AUC 미계산
                })
        
        # DataFrame 저장
        df_summary = pd.DataFrame(summary)
        df_summary = df_summary.sort_values('f1_score', ascending=False)
        
        excel_path = os.path.join(self.result_dir, f'performance_summary_{timestamp}.xlsx')
        df_summary.to_excel(excel_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"결과 저장 완료: {excel_path}")
        print(f"{'='*60}\n")
        print(df_summary.to_string(index=False))
        
        # 최고 성능 출력
        best_row = df_summary.iloc[0]
        print(f"\n🏆 최고 성능:")
        print(f"   방법: {best_row['method']}")
        print(f"   F1 Score: {best_row['f1_score']:.4f}")
        print(f"   Accuracy: {best_row['accuracy']:.4f}")
        
        # 전체 결과 pickle 저장
        pickle_path = os.path.join(self.result_dir, f'full_results_{timestamp}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        return df_summary


def main():
    """메인 실행 함수"""
    print(f"\n{'='*60}")
    print(f"성능 개선 실험 시작")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # 실험 초기화
    experiment = PerformanceExperiment(
        data_path='../data/total_again.xlsx',
        disease_name='mets_transition'
    )
    
    # 데이터 준비
    train_loader, val_loader, test_loader, train_df, val_df, test_df = experiment.prepare_data(
        selection_method='max_disease_change',
        resample_method='smote'
    )
    
    # 실험 1: Loss 함수 비교
    loss_results = experiment.experiment_loss_functions(train_loader, val_loader, test_loader)
    
    # 실험 2: 앙상블 비교
    ensemble_results = experiment.experiment_ensemble(train_df, val_df, test_df)
    
    # 결과 저장
    summary_df = experiment.save_results()
    
    print(f"\n{'='*60}")
    print(f"실험 완료!")
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    return summary_df


if __name__ == '__main__':
    summary = main()
