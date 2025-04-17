import torch.nn as nn
from models.attention import MultiHeadAttention
from models.feed_forward import FeedForward
import torch
from typing import Dict, Any

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # Multi-head self-attention and feed-forward submodules
        self.attn: nn.Module = MultiHeadAttention(config)
        self.ffn: nn.Module = FeedForward(config)

        # Layer Normalization
        embed_dim = config['model']['embed_dim']
        self.norm1: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.norm2: nn.LayerNorm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TransformerEncoderLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor of same shape
        """
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)

