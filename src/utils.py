import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, auc, roc_curve, accuracy_score, f1_score
import matplotlib.pyplot as plt
import os

demo_cols = ['days_between', '나이_T0', '신장_T0', '성별_T0']
life_cols = ['흡연_T0', '활동량_T0', '음주_T0']
bio_cols = ['SBP_T0', 'DBP_T0', 'GLUCOSE_T0', 'HBA1C_T0', 'TG_T0', 'HDL CHOL_T0', 'LDL CHOL_T0', 'eGFR_T0', 'WAIST_T0', '체중_T0', 'BMI_T0']
diet_cols = ['간식빈도_T0', '고지방 육류_T0', '곡류_T0', '과일_T0', '단맛_T0', '단백질류_T0', '물_T0', '밥 양_T0', 
             '식사 빈도_T0', '식사량_T0', '외식빈도_T0', '유제품_T0', '음료류_T0', '인스턴트 가공식품_T0', '짠 간_T0', '짠 식습관_T0', '채소_T0', '커피_T0', '튀김_T0']
interaction_cols = ['bmi_waist_risk', 'bp_age_risk', 'tg_hdl_ratio', 'unhealthy_diet_score', 'healthy_diet_score', 'diet_change_rate']
mets_cols = ['Increased waist circumference', 'Elevated blood pressure', 'Impaired fasting glucose', 'Elevated triglycerides', 'Decreased HDL-C']
disease_delta_cols = [f'{disease}_delta' for disease in mets_cols]

def create_model_with_dynamic_dims(train_dataset, mets_cols, dropout_rate=0.4, l1_lambda=0, l2_lambda=0):
    from MetS_prediction_model import MultiDiseasePredictor
    if hasattr(train_dataset, 'dims'):
        dims = train_dataset.dims
    else:
        sample = train_dataset[0]
        dims = {
            'diet': len(sample['diet']),
            'demo': len(sample['demo']),
            'life': len(sample['life']),
            'bio': len(sample['bio']),
            'delta': len(sample['delta']),
            'interaction': len(sample['interaction']),
            'pca': len(sample['pca'])
        }
    
    model = MultiDiseasePredictor(
        diet_dim=dims['diet'],
        demo_dim=dims['demo'], 
        life_dim=dims['life'],
        bio_dim=dims['bio'],
        change_dim=dims['delta'],
        inter_dim=dims['interaction'],
        pca_dim=dims['pca'],
        disease_names=mets_cols,
        dropout_rate=dropout_rate,
        l1_lambda=l1_lambda,
        l2_lambda=l2_lambda
    )
    
    return model

def parse_combination(combo_name):
    selection_strategies = ['max_disease_change', 'first_last']
    model_methods = ['base', 'ensemble']
    resampling_methods = ['none', 'smote']
    loss_methods = ['CrossEntropy', 'FocalLoss', 'WeightedCE_log']

    data_strategy = None
    model_method = None
    resample_method = None
    loss_method = None
    
    for strategy in selection_strategies:
        if combo_name.startswith(strategy):
            data_strategy = strategy
            combo_name = combo_name[len(strategy)+1:]
            break
    
    for method in model_methods:
        if combo_name.startswith(method):
            model_method = method
            combo_name = combo_name[len(method)+1:]
            break
    
    for method in resampling_methods:
        if combo_name.startswith(method):
            resample_method = method
            combo_name = combo_name[len(method)+1:]
            break
    
    loss_method = combo_name
    
    return data_strategy, model_method, resample_method, loss_method

def evaluate_model(model, test_loader, device, disease_name):
    model.eval()
    all_pred = []
    all_true = []
    all_prob = []
    
    with torch.no_grad():
        for batch in test_loader:
            diet_data = batch['diet'].to(device)
            demo_data = batch['demo'].to(device)
            life_data = batch['life'].to(device)
            bio_data = batch['bio'].to(device)
            inter_data = batch['interaction'].to(device)
            pca_data = batch['pca'].to(device)
            change_data = batch['delta'].to(device)
            target = batch['target'].to(device).squeeze()
            
            outputs = model(diet_data, demo_data, life_data, bio_data, inter_data, pca_data, change_data, disease_name)
            pred = torch.argmax(outputs['disease_logits'], dim=1)
            prob = torch.softmax(outputs['disease_logits'], dim=1)
            
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(target.cpu().numpy())
            all_prob.extend(prob.cpu().numpy())
    
    accuracy = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average='macro')
    
    all_true = np.array(all_true)
    all_prob = np.array(all_prob)
    
    roc_aucs = {}
    pr_aucs = {}
    class_names = ['개선', '유지', '악화']
    
    for i in range(3):
        y_true_binary = (all_true == i).astype(int)
        y_prob_class = all_prob[:, i]

        fpr, tpr, _ = roc_curve(y_true_binary, y_prob_class)
        roc_auc = auc(fpr, tpr)
        roc_aucs[class_names[i]] = roc_auc

        precision, recall, _ = precision_recall_curve(y_true_binary, y_prob_class)
        pr_auc = auc(recall, precision)
        pr_aucs[class_names[i]] = pr_auc
    
    roc_aucs['Macro'] = np.mean(list(roc_aucs.values()))
    pr_aucs['Macro'] = np.mean(list(pr_aucs.values()))
    
    return accuracy, f1, roc_aucs, pr_aucs

def evaluate_all_models_auc(all_results, methods, test_loaders, device, mets_cols):
    roc_results = []
    pr_results = []
    
    for disease_name in mets_cols:
        for method in methods:
            model = all_results[method][disease_name]['model']
            test_loader = test_loaders[disease_name]
            
            _, _, roc_aucs, pr_aucs = evaluate_model(model, test_loader, device, disease_name)
            
            roc_results.append({
                'Disease': disease_name,
                'Method': method,
                '개선_ROC_AUC': f"{roc_aucs['개선']:.3f}",
                '유지_ROC_AUC': f"{roc_aucs['유지']:.3f}",
                '악화_ROC_AUC': f"{roc_aucs['악화']:.3f}",
                'Macro_ROC_AUC': f"{roc_aucs['Macro']:.3f}"
            })
            pr_results.append({
                'Disease': disease_name,
                'Method': method,
                '개선_PR_AUC': f"{pr_aucs['개선']:.3f}",
                '유지_PR_AUC': f"{pr_aucs['유지']:.3f}",
                '악화_PR_AUC': f"{pr_aucs['악화']:.3f}",
                'Macro_PR_AUC': f"{pr_aucs['Macro']:.3f}"
            })
    
    roc_pivot = pd.pivot_table(pd.DataFrame(roc_results),
        index='Method',
        columns='Disease',
        values=['개선_ROC_AUC', '유지_ROC_AUC', '악화_ROC_AUC', 'Macro_ROC_AUC'],
        aggfunc='first')
    
    pr_pivot = pd.pivot_table(pd.DataFrame(pr_results),
        index='Method',
        columns='Disease',
        values=['개선_PR_AUC', '유지_PR_AUC', '악화_PR_AUC', 'Macro_PR_AUC'],
        aggfunc='first')

    return roc_pivot, pr_pivot

def analyze_results(results, mets_cols):
    performance_data = []
    for resample_method, resample_results in results.items():
        for loss_method, loss_results in resample_results.items():
            for disease, disease_results in loss_results.items():
                eval_results = disease_results['evaluation']
                performance_data.append({
                    'resample_method': resample_method,
                    'loss_method': loss_method,
                    'Disease': disease,
                    'ACC': eval_results['accuracy'],
                    'F1': eval_results['f1_score'],
                    'ROC_AUC': eval_results['roc_aucs'],
                    'PR_AUC' : eval_results['pr_aucs']
                })
    
    performance_df = pd.DataFrame(performance_data)
    
    best_methods = {}
    for disease in mets_cols:
        disease_data = performance_df[performance_df['Disease'] == disease]
        best_idx = disease_data['F1'].idxmax()
        best_methods[disease] = {
            'resample_method': disease_data.loc[best_idx, 'resample_method'],
            'loss_method': disease_data.loc[best_idx, 'loss_method'],
            'F1': disease_data.loc[best_idx, 'F1'],
            'ROC_AUC': disease_data.loc[best_idx, 'ROC_AUC'],
            'PR_AUC' : disease_data.loc[best_idx, 'PR_AUC']
        }
    
    return best_methods, performance_df

def plot_training_curves(results, save_dir):
    total_plots = sum(len(loss_results) for resample_results in results.values() 
                     for loss_results in resample_results.values())
    
    cols = 5
    rows = (total_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if total_plots > 1 else [axes]
    
    plot_idx = 0
    for resample_method, resample_results in results.items():
        for loss_method, loss_results in resample_results.items():
            for disease_name, disease_results in loss_results.items():
                ax = axes[plot_idx]
                
                train_losses = disease_results['train_losses']
                val_losses = disease_results['val_losses']
                
                ax.plot(train_losses, label='Training Loss', linewidth=2)
                ax.plot(val_losses, label='Validation Loss', linewidth=2)
                
                ax.set_title(f'{disease_name}\n({resample_method}, {loss_method})')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
    
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])
    
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.close()