import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionLayer(nn.Module):
    """Self-attention mechanism for feature importance weighting"""
    def __init__(self, input_dim, attention_dim=16):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        attention_weights = F.softmax(self.attention(x), dim=1)
        attended = x * attention_weights
        return attended, attention_weights


class ResidualBlock(nn.Module):
    """Residual connection block for better gradient flow"""
    def __init__(self, dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)


class ImprovedMultiDiseasePredictor(nn.Module):
    """
    Improved model with:
    1. Attention mechanism for feature importance
    2. Residual connections for better gradient flow
    3. Larger hidden dimensions for better representation
    """
    def __init__(self, diet_dim, demo_dim, life_dim, bio_dim, change_dim, inter_dim, pca_dim,
                 disease_names, dropout_rate=0.3, l1_lambda=0, l2_lambda=0):
        super(ImprovedMultiDiseasePredictor, self).__init__()

        self.disease_names = disease_names
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # Increased hidden dimension for better representation
        hidden_dim = 16  # increased from 8

        # Individual encoders for each modality with attention
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

        # Attention layers for each disease
        combined_dim = hidden_dim * 7  # 7 components
        self.attention_layers = nn.ModuleDict({
            disease: AttentionLayer(combined_dim, attention_dim=32)
            for disease in disease_names
        })

        # Residual blocks for better gradient flow
        self.residual_blocks = nn.ModuleDict({
            disease: nn.Sequential(
                ResidualBlock(combined_dim, dropout_rate),
                ResidualBlock(combined_dim, dropout_rate)
            ) for disease in disease_names
        })

        # Classification heads with deeper architecture
        self.heads = nn.ModuleDict({
            disease: nn.Sequential(
                nn.Linear(combined_dim, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(16, 3)
            ) for disease in disease_names
        })

    def forward(self, diet_data, demo_data, life_data, bio_data, change_data, inter_data, pca_data, disease_name):
        # Encode each modality
        d = self.diet_encoders[disease_name](diet_data)
        demo = self.demo_encoders[disease_name](demo_data)
        life = self.life_encoders[disease_name](life_data)
        b = self.bio_encoders[disease_name](bio_data)
        c = self.change_encoders[disease_name](change_data)
        inter = self.inter_encoders[disease_name](inter_data)
        pca = self.pca_encoders[disease_name](pca_data)

        # Concatenate all encoded features
        concat = torch.cat([d, demo, life, b, c, inter, pca], dim=1)

        # Apply attention mechanism
        attended, attention_weights = self.attention_layers[disease_name](concat)

        # Apply residual blocks
        residual_out = self.residual_blocks[disease_name](attended)

        # Final classification
        disease_logits = self.heads[disease_name](residual_out)

        return {
            'disease_logits': disease_logits,
            'attention_weights': attention_weights
        }

    def regularization_loss(self):
        l1_loss = sum(p.abs().sum() for p in self.parameters())
        l2_loss = sum(p.pow(2).sum() for p in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss


class EnsemblePredictor:
    """
    Ensemble of deep learning and sklearn models
    """
    def __init__(self, deep_model, sklearn_models, weights=None, device='cpu'):
        self.deep_model = deep_model
        self.sklearn_models = sklearn_models  # dict of sklearn models per disease
        self.device = device

        # Default equal weights
        if weights is None:
            self.weights = {'deep': 0.5, 'sklearn': 0.5}
        else:
            self.weights = weights

    def predict_proba(self, data_loader, disease_name):
        """
        Get ensemble predictions combining deep learning and sklearn

        Returns:
            Combined probability predictions
        """
        # Get deep learning predictions
        self.deep_model.eval()
        deep_probs = []

        with torch.no_grad():
            for batch in data_loader:
                diet_data = batch['diet'].to(self.device)
                demo_data = batch['demo'].to(self.device)
                life_data = batch['life'].to(self.device)
                bio_data = batch['bio'].to(self.device)
                change_data = batch['delta'].to(self.device)
                inter_data = batch['interaction'].to(self.device)
                pca_data = batch['pca'].to(self.device)

                outputs = self.deep_model(diet_data, demo_data, life_data, bio_data,
                                         change_data, inter_data, pca_data, disease_name)
                probs = F.softmax(outputs['disease_logits'], dim=1)
                deep_probs.append(probs.cpu().numpy())

        deep_probs = np.vstack(deep_probs)

        # Get sklearn predictions
        # Note: This requires the full feature matrix, which needs to be extracted from data_loader
        # For now, we'll return just deep learning predictions
        # In practice, you'd need to extract features and get sklearn predictions

        return deep_probs

    def predict(self, data_loader, disease_name):
        probs = self.predict_proba(data_loader, disease_name)
        return np.argmax(probs, axis=1)
