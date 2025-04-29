import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class FeedForward(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']
        training_cfg = config.get('training', {})
        
        # Extract dimensions from config
        embed_dim = model_cfg['embed_dim']
        ffn_dim = model_cfg['ffn_dim']
        
        # Main linear transformations
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        
        # Activation function (configurable)
        self.activation = self._get_activation(training_cfg.get('activation', 'gelu'))
        
        # Dropout layers
        dropout_p = model_cfg.get('dropout', 0.1)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim) if training_cfg.get('ffn_pre_norm', False) else None
        
        # Residual connection scaling
        self.residual_scaling = training_cfg.get('residual_scaling', 1.0)
        
        # Initialize weights
        self._init_weights()

    def _get_activation(self, activation_name: str):
        """Return activation function based on config"""
        activations = {
            'relu': F.relu,
            'gelu': F.gelu,
            'silu': F.silu,
            'leaky_relu': F.leaky_relu,
        }
        return activations.get(activation_name.lower(), F.gelu)

    def _init_weights(self) -> None:
        """Initialize weights with appropriate schemes"""
        # Kaiming initialization for ReLU-like activations
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc2.weight)
        
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional normalization and residual connection.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of same shape with feed-forward transformation applied
        """
        residual = x
        
        # Apply pre-normalization if configured
        if self.norm is not None:
            x = self.norm(x)
        
        # First linear layer + activation + dropout
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        # Second linear layer + dropout
        x = self.fc2(x)
        x = self.dropout2(x)
        
        # Residual connection with optional scaling
        return residual * self.residual_scaling + x