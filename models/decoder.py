import torch.nn as nn
from models.feed_forward import FeedForward
import importlib

import torch
import torch.nn as nn
import importlib
from typing import Dict, Any

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Dynamically import the attention class
        decoder_module = importlib.import_module("models.attention")
        decoder_class = getattr(decoder_module, config['decoder']['decoder_class'])
        
        # Initialize attention layers
        self.attn_masked: nn.Module = decoder_class(config)
        self.attn: nn.Module = decoder_class(config)
        
        # Feed-forward network
        self.ffn: nn.Module = FeedForward(config)
        
        # Layer Normalizations
        embed_dim = config['model']['embed_dim']
        self.norm1: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.norm2: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.norm3: nn.LayerNorm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TransformerDecoderLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor of the same shape
        """
        attn_out_masked = self.attn_masked(x, mask=True)
        x = self.norm1(x + attn_out_masked)
        
        attn_out = self.attn(x)
        x = self.norm2(x + attn_out)
        
        ffn_out = self.ffn(x)
        return self.norm3(x + ffn_out)
