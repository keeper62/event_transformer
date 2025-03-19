import torch.nn as nn
from models.feed_forward import FeedForward
import importlib

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        decoder_module = importlib.import_module("models.attention")
        decoder_class = getattr(decoder_module, config['decoder']['decoder_class'])
        
        self.attn_masked = decoder_class(config)
        self.attn = decoder_class(config)

        self.ffn = FeedForward(config)
        self.norm1 = nn.LayerNorm(config['model']['embed_dim'])
        self.norm2 = nn.LayerNorm(config['model']['embed_dim'])
        self.norm3 = nn.LayerNorm(config['model']['embed_dim'])
    
    def forward(self, x):
        attn_out_masked = self.attn_masked(x, mask=True)
        x = self.norm1(x + attn_out_masked)
        attn_out = self.attn(x)
        x = self.norm2(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm3(x + ffn_out)