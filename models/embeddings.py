import torch.nn as nn
import importlib
import torch

import torch
import torch.nn as nn
import importlib
from typing import Dict, Any

class Embeddings(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        model_cfg = config['model']
        embed_dim = model_cfg['embed_dim']
        
        # Token embedding layer
        self.embedding_layer: nn.Embedding = nn.Embedding(
            num_embeddings=model_cfg['vocab_size'],
            embedding_dim=embed_dim
        )

        # Positional embedding configuration
        self.conf_ape: str = config['ape']['ape_class']
        self.position_embedding: nn.Module = None

        if self.conf_ape:
            position_module = importlib.import_module("models.position")
            position_class = getattr(position_module, self.conf_ape)
            self.position_embedding = position_class(
                d_model=embed_dim,
                max_len=model_cfg['max_len']
            )

        # Dropout layer
        self.dropout: nn.Dropout = nn.Dropout(model_cfg.get('dropout', 0.1))
    
    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Embeddings.

        Args:
            x (torch.Tensor): Input token indices of shape (batch_size, seq_len)
            timestamps (torch.Tensor): Time-based input for positional encoding

        Returns:
            torch.Tensor: Embedded input of shape (batch_size, seq_len, embed_dim)
        """
        x = self.embedding_layer(x)  # (B, T) -> (B, T, D)

        if self.conf_ape and self.position_embedding is not None:
            if self.conf_ape == "UnixTimeDeltaPosition":
                positions = self.position_embedding(timestamps)
            else:
                positions = self.position_embedding(x)
            x = x + positions  # Element-wise add

        return self.dropout(x.float())  # Ensure float and apply dropout
