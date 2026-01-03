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
def calculate_improved_class_weights(train_loader, method='inverse_freq'):
    """개선된 클래스 가중치 계산
    
    Args:
        train_loader: 학습 데이터 로더
        method: 'inverse_freq', 'effective_num', 'balanced'
    
    Returns:
        class_weights: 클래스 가중치 리스트
    """
    class_counts = np.zeros(3)
    
    for batch in train_loader:
        targets = batch['target'].numpy().flatten()
        for t in targets:
            class_counts[t] += 1
    
    total = class_counts.sum()
    
    if method == 'inverse_freq':
        # 역빈도 가중치 (더 강한 불균형 보정)
        class_weights = total / (3 * class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * 3  # 정규화
    
    elif method == 'effective_num':
        # Effective Number of Samples 기반
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        class_weights = (1.0 - beta) / (effective_num + 1e-6)
        class_weights = class_weights / class_weights.sum() * 3
    
    elif method == 'balanced':
        # Sklearn 스타일 balanced 가중치
        class_weights = total / (3 * class_counts)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"\n클래스 가중치 계산 ({method}):")
    print(f"  클래스 분포: {class_counts.astype(int)}")
    print(f"  클래스 비율: {(class_counts/total*100).round(2)}%")
    print(f"  가중치: {class_weights.round(4)}")
    
    return class_weights.tolist()

def improved_loss_methods_configs(train_loaders, disease_name, device):
    """개선된 Loss 설정 (성능 향상 버전)"""
    train_loader = train_loaders[disease_name]
    
    # 개선된 클래스 가중치 계산
    weights_inverse = calculate_improved_class_weights(train_loader, 'inverse_freq')
    weights_effective = calculate_improved_class_weights(train_loader, 'effective_num')
    weights_balanced = calculate_improved_class_weights(train_loader, 'balanced')
    
    loss_configs = {
        # 기존 (비교용)
        'CrossEntropy': nn.CrossEntropyLoss(),
        'FocalLoss_gamma1.5': FocalLoss(gamma=1.5),
        
        # 개선된 Focal Loss (gamma 증가)
        'FocalLoss_gamma2.0': FocalLoss(gamma=2.0),
        'FocalLoss_gamma2.5': FocalLoss(gamma=2.5),
        
        # 개선된 가중치 적용
        'WeightedCE_inverse': nn.CrossEntropyLoss(
            weight=torch.tensor(weights_inverse).float().to(device)
        ),
        'WeightedCE_effective': nn.CrossEntropyLoss(
            weight=torch.tensor(weights_effective).float().to(device)
        ),
        'WeightedCE_balanced': nn.CrossEntropyLoss(
            weight=torch.tensor(weights_balanced).float().to(device)
        ),
        
        # Focal Loss + 가중치 조합
        'FocalLoss_gamma2.0_weighted': FocalLoss(
            alpha=weights_inverse, 
            gamma=2.0
        ),
    }
    
    return loss_configs
