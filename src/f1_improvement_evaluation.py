import os
import pickle
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_fscore_support

from data_import import DataPreprocessor, DataLoaderManager, SingleDietImpactDataset
from feature_engineering import apply_to_all_data
from resampling import create_resampled_dataset
from improved_model import ImprovedMultiDiseasePredictor
from MetS_prediction_model import MultiDiseasePredictor, train_sklearn_ensemble, EarlyStopping
from loss_functions import improved_loss_methods_configs
from utils import mets_cols


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def evaluate_model(model, test_loader, device, disease_name):
    """Evaluate model and return metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            diet_data = batch['diet'].to(device)
            demo_data = batch['demo'].to(device)
            life_data = batch['life'].to(device)
            bio_data = batch['bio'].to(device)
            change_data = batch['delta'].to(device)
            inter_data = batch['interaction'].to(device)
            pca_data = batch['pca'].to(device)
            target = batch['target'].to(device).squeeze()

            outputs = model(diet_data, demo_data, life_data, bio_data,
                          change_data, inter_data, pca_data, disease_name)

            probs = torch.softmax(outputs['disease_logits'], dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    f1_weighted = f1_score(all_targets, all_preds, average='weighted')

    # Per-class metrics
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'precision': precision,
        'recall': recall
    }


def train_improved_model(train_loader, val_loader, test_loader, disease_name,
                        model_config, loss_method, num_epochs=100):
    """Train model with improved architecture"""

    model = model_config['model'].to(device)

    # Get criterion
    improved_configs = improved_loss_methods_configs(
        {disease_name: train_loader}, disease_name, device
    )
    criterion = improved_configs[loss_method]

    # Optimizer with better settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=6e-4,
        betas=(0.9, 0.999)
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)

    # Training loop
    best_val_f1 = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            diet_data = batch['diet'].to(device)
            demo_data = batch['demo'].to(device)
            life_data = batch['life'].to(device)
            bio_data = batch['bio'].to(device)
            change_data = batch['delta'].to(device)
            inter_data = batch['interaction'].to(device)
            pca_data = batch['pca'].to(device)
            target = batch['target'].long().to(device).squeeze()

            outputs = model(diet_data, demo_data, life_data, bio_data,
                          change_data, inter_data, pca_data, disease_name)

            loss = criterion(outputs['disease_logits'], target) + model.regularization_loss()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                diet_data = batch['diet'].to(device)
                demo_data = batch['demo'].to(device)
                life_data = batch['life'].to(device)
                bio_data = batch['bio'].to(device)
                change_data = batch['delta'].to(device)
                inter_data = batch['interaction'].to(device)
                pca_data = batch['pca'].to(device)
                target = batch['target'].long().to(device).squeeze()

                outputs = model(diet_data, demo_data, life_data, bio_data,
                              change_data, inter_data, pca_data, disease_name)

                loss = criterion(outputs['disease_logits'], target)
                val_loss += loss.item()
                val_batches += 1

                preds = torch.argmax(outputs['disease_logits'], dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        val_loss_avg = val_loss / val_batches
        val_f1 = f1_score(val_targets, val_preds, average='macro')

        scheduler.step(val_loss_avg)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/train_batches:.4f}, "
                  f"Val Loss: {val_loss_avg:.4f}, Val F1: {val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

        if early_stopping(val_loss_avg, model):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    early_stopping.load_best_model(model)

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device, disease_name)

    return model, test_metrics


def run_comprehensive_evaluation(data_processor, best_params):
    """
    Run comprehensive evaluation of all improvement methods
    """

    print("\n" + "="*80)
    print("F1 Score Improvement Comprehensive Evaluation")
    print("="*80)

    # Prepare data
    print("\nPreparing data...")
    train_df, val_df, test_df, _ = data_processor.process_all(
        normalize=True, selection_strategy='max_disease_change'
    )
    train_df, val_df, test_df, actual_change_dim, _pca = apply_to_all_data(
        train_df, val_df, test_df
    )

    temp_dataset = SingleDietImpactDataset(train_df)

    # Get dimensions
    sample_batch_loader = DataLoaderManager.create_disease_loaders(
        train_df, val_df, test_df, batch_size=32
    )[0]
    sample_batch = next(iter(sample_batch_loader[mets_cols[0]]))

    dims = {
        'diet': sample_batch['diet'].shape[1],
        'demo': sample_batch['demo'].shape[1],
        'life': sample_batch['life'].shape[1],
        'bio': sample_batch['bio'].shape[1],
        'delta': sample_batch['delta'].shape[1],
        'interaction': sample_batch['interaction'].shape[1],
        'pca': sample_batch['pca'].shape[1]
    }

    print(f"Data dimensions: {dims}")

    # Test configurations
    configurations = [
        {
            'name': 'Baseline (Original)',
            'model_type': 'original',
            'resampling': 'smote',
            'loss': 'FocalLoss_gamma2.5'
        },
        {
            'name': 'Improved Model + SMOTE',
            'model_type': 'improved',
            'resampling': 'smote',
            'loss': 'FocalLoss_gamma2.5'
        },
        {
            'name': 'Improved Model + ADASYN',
            'model_type': 'improved',
            'resampling': 'adasyn',
            'loss': 'FocalLoss_gamma2.5'
        },
        {
            'name': 'Improved Model + Borderline-SMOTE',
            'model_type': 'improved',
            'resampling': 'borderline_smote',
            'loss': 'FocalLoss_gamma2.5'
        }
    ]

    results = {}

    for config in configurations:
        print(f"\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"{'='*80}")

        config_results = {}

        # Prepare data loaders
        if config['resampling'] == 'none':
            train_loaders, val_loaders, test_loaders = DataLoaderManager.create_disease_loaders(
                train_df, val_df, test_df, batch_size=64
            )
        else:
            resampled_train = create_resampled_dataset(train_df, config['resampling'])

            train_loaders = {}
            for disease in mets_cols:
                disease_indices = [i for i, item in enumerate(resampled_train)
                                 if item['disease_name'] == disease]
                subset = torch.utils.data.Subset(resampled_train, disease_indices)
                train_loaders[disease] = torch.utils.data.DataLoader(
                    subset, batch_size=64, shuffle=True, drop_last=True
                )

            _, val_loaders, test_loaders = DataLoaderManager.create_disease_loaders(
                train_df, val_df, test_df, batch_size=64
            )

        # Test only first disease for speed
        disease = mets_cols[0]
        print(f"\nTesting on disease: {disease}")

        # Create model
        if config['model_type'] == 'improved':
            model = ImprovedMultiDiseasePredictor(
                diet_dim=dims['diet'],
                demo_dim=dims['demo'],
                life_dim=dims['life'],
                bio_dim=dims['bio'],
                change_dim=dims['delta'],
                inter_dim=dims['interaction'],
                pca_dim=dims['pca'],
                disease_names=mets_cols,
                dropout_rate=best_params.get('dropout_rate', 0.3),
                l1_lambda=best_params.get('l1_lambda', 0.00026),
                l2_lambda=best_params.get('l2_lambda', 0.00026)
            )
        else:
            model = MultiDiseasePredictor(
                diet_dim=dims['diet'],
                demo_dim=dims['demo'],
                life_dim=dims['life'],
                bio_dim=dims['bio'],
                change_dim=dims['delta'],
                inter_dim=dims['interaction'],
                pca_dim=dims['pca'],
                disease_names=mets_cols,
                dropout_rate=best_params.get('dropout_rate', 0.3),
                l1_lambda=best_params.get('l1_lambda', 0.00026),
                l2_lambda=best_params.get('l2_lambda', 0.00026)
            )

        # Train and evaluate
        trained_model, metrics = train_improved_model(
            train_loaders[disease],
            val_loaders[disease],
            test_loaders[disease],
            disease,
            {'model': model},
            config['loss'],
            num_epochs=50  # Reduced for speed
        )

        config_results[disease] = metrics
        results[config['name']] = config_results

        print(f"\nResults for {config['name']}:")
        print(f"  F1 Macro: {metrics['f1_macro']:.3f}")
        print(f"  F1 Weighted: {metrics['f1_weighted']:.3f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1 per class: {metrics['f1_per_class']}")

        # Clean up
        del model, trained_model, train_loaders, val_loaders, test_loaders
        if config['resampling'] != 'none':
            del resampled_train
        torch.cuda.empty_cache()

    return results


def create_comparison_report(results, save_dir='../result/f1_improvement'):
    """Create comparison report of all methods"""

    os.makedirs(save_dir, exist_ok=True)

    # Create DataFrame for comparison
    comparison_data = []

    for config_name, config_results in results.items():
        for disease, metrics in config_results.items():
            comparison_data.append({
                'Configuration': config_name,
                'Disease': disease,
                'F1_Macro': metrics['f1_macro'],
                'F1_Weighted': metrics['f1_weighted'],
                'Accuracy': metrics['accuracy'],
                'F1_Improve': metrics['f1_per_class'][0],
                'F1_Maintain': metrics['f1_per_class'][1],
                'F1_Worsen': metrics['f1_per_class'][2]
            })

    df = pd.DataFrame(comparison_data)

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(save_dir, f'f1_improvement_comparison_{timestamp}.csv')
    df.to_csv(csv_path, index=False)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: F1 Score Improvements")
    print("="*80)

    baseline_f1 = df[df['Configuration'] == 'Baseline (Original)']['F1_Macro'].values[0]

    print(f"\nBaseline F1 Score: {baseline_f1:.3f}")
    print("\nComparison with Baseline:")

    for config_name in df['Configuration'].unique():
        if config_name == 'Baseline (Original)':
            continue

        config_f1 = df[df['Configuration'] == config_name]['F1_Macro'].values[0]
        improvement = ((config_f1 - baseline_f1) / baseline_f1) * 100

        print(f"  {config_name}:")
        print(f"    F1 Score: {config_f1:.3f} ({improvement:+.1f}%)")

    # Find best configuration
    best_idx = df['F1_Macro'].idxmax()
    best_config = df.loc[best_idx]

    print(f"\n🏆 Best Configuration: {best_config['Configuration']}")
    print(f"   F1 Macro: {best_config['F1_Macro']:.3f}")
    print(f"   F1 Weighted: {best_config['F1_Weighted']:.3f}")
    print(f"   Accuracy: {best_config['Accuracy']:.3f}")

    print(f"\n✅ Results saved to: {csv_path}")

    return df, csv_path


if __name__ == '__main__':
    # Load best parameters
    study_dir = '../result/optuna_study'
    study_files = sorted([f for f in os.listdir(study_dir) if f.startswith('optuna_study_')])

    if study_files:
        latest_study = os.path.join(study_dir, study_files[-1])
        with open(latest_study, 'rb') as f:
            study = pickle.load(f)
        best_params = study.best_params
        print(f"Loaded best parameters: {best_params}")
    else:
        best_params = {
            'dropout_rate': 0.3017,
            'l1_lambda': 0.000261,
            'l2_lambda': 0.000258
        }
        print(f"Using default parameters: {best_params}")

    # Initialize data processor
    data_path = '../data/total_again.xlsx'
    data_processor = DataPreprocessor(data_path)

    # Run evaluation
    results = run_comprehensive_evaluation(data_processor, best_params)

    # Create report
    comparison_df, report_path = create_comparison_report(results)

    print("\n✅ Evaluation complete!")
