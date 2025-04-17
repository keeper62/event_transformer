import torch
import torch.nn as nn
from typing import Dict, Any

class FeedForward(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']
        
        self.fc1: nn.Linear = nn.Linear(model_cfg['embed_dim'], model_cfg['ffn_dim'])
        self.fc2: nn.Linear = nn.Linear(model_cfg['ffn_dim'], model_cfg['embed_dim'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FeedForward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor of the same shape
        """
        return self.fc2(torch.relu(self.fc1(x)))
