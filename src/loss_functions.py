import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, class_weights=None, gamma_neg=2, gamma_pos=1, eps=1e-8):
        super(WeightedAsymmetricLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.eps = eps

    def forward(self, x, y):
        p = F.softmax(x, dim=1)
        y_one_hot = F.one_hot(y, num_classes=x.size(1)).float()
        ce = -torch.sum(y_one_hot * torch.log(p + self.eps), dim=1)
        
        p_t = torch.sum(y_one_hot * p, dim=1)
        focal_weight = torch.pow(1 - p_t, self.gamma_pos)
        
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, device=x.device)
            class_weight = torch.sum(y_one_hot * weights, dim=1)
            ce = ce * class_weight
        
        loss = focal_weight * ce
        
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_class_weights(train_loader):
    class_counts = [0, 0, 0]
    
    for batch in train_loader:
        targets = batch['target'].numpy().flatten()
        for t in targets:
            class_counts[t] += 1
    
    total = sum(class_counts)    
    class_weights = [np.log(total / count + 1) if count > 0 else 1.0 for count in class_counts]
    return class_weights

def loss_methods_configs(train_loaders, disease_name, device):
    train_loader = train_loaders[disease_name]
    loss_configs = {
        'CrossEntropy': nn.CrossEntropyLoss(),
        'FocalLoss': FocalLoss(gamma=1.5),
        'WeightedCE_log': nn.CrossEntropyLoss(weight=torch.tensor(calculate_class_weights(train_loader)).float().to(device)),
    }
    return loss_configs