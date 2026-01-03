import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_curve, accuracy_score, f1_score
from MetS_prediction_model import EarlyStopping
from loss_functions import loss_methods_configs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model_with_loss(model, train_loader, val_loader, test_loader, disease_name, model_type, loss_method):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=6e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    loss_configs = loss_methods_configs({disease_name: train_loader}, disease_name, device)
    criterion = loss_configs[loss_method]
    
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    train_losses = []
    val_losses = []
    
    print("train model with loss start")
    for epoch in range(100):
        print(f"epoch: {epoch}")
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            target = batch['target'].long().to(device).squeeze()
            
            optimizer.zero_grad()
            
            diet_data = batch['diet'].to(device)
            demo_data = batch['demo'].to(device)
            life_data = batch['life'].to(device)
            bio_data = batch['bio'].to(device)
            change_data = batch['delta'].to(device)
            inter_data = batch['interaction'].to(device)
            pca_data = batch['pca'].to(device)
            target = batch['target'].long().to(device).squeeze()
            
            outputs = model(diet_data, demo_data, life_data, bio_data, change_data, inter_data, pca_data, disease_name)
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
                diet_data = batch['diet'].to(device)
                demo_data = batch['demo'].to(device)
                life_data = batch['life'].to(device)
                bio_data = batch['bio'].to(device)
                change_data = batch['delta'].to(device)
                inter_data = batch['interaction'].to(device)
                pca_data = batch['pca'].to(device)
                target = batch['target'].long().to(device).squeeze()
                
                outputs = model(diet_data, demo_data, life_data, bio_data, change_data, inter_data, pca_data, disease_name)
                
                val_loss = criterion(outputs['disease_logits'], target)
                val_total_loss += val_loss.item()
                val_batch_count += 1
        
        val_loss_avg = val_total_loss / val_batch_count
        val_losses.append(val_loss_avg)
        
        scheduler.step(val_loss_avg)
        
        if early_stopping(val_loss_avg, model):
            break
    
    early_stopping.load_best_model(model)
    
    accuracy, f1, roc_aucs, pr_aucs = evaluate_model_custom(model, test_loader, device, disease_name, model_type)
    
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

def evaluate_model_custom(model, test_loader, device, disease_name, model_type):
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
            change_data = batch['delta'].to(device)
            inter_data = batch['interaction'].to(device)
            pca_data = batch['pca'].to(device)
            target = batch['target'].long().to(device).squeeze()
            
            outputs = model(diet_data, demo_data, life_data, bio_data, change_data, inter_data, pca_data, disease_name)
                
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