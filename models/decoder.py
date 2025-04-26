import torch.nn as nn
from models.feed_forward import FeedForward
import importlib

import torch
import torch.nn as nn
import importlib
from typing import Dict, Any

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
        
        # Optional: Project meaning to match embed_dim if it's not already
        self.meaning_projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor, timestamps: torch.Tensor = None) -> torch.Tensor:
        # Pass through the masked attention layer
        attn_out_masked = self.attn_masked(x, timestamps=timestamps, mask=True)
        x = self.norm1(x + attn_out_masked)

        # Pass through the normal attention layer
        attn_out = self.attn(x, timestamps=timestamps)
        x = self.norm2(x + attn_out)

        # Feed-forward network
        ffn_out = self.ffn(x)
        return self.norm3(x + ffn_out)

