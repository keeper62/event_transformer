from models.feed_forward import FeedForward

import torch
import torch.nn as nn
import importlib
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

class BiasInjection(nn.Module):
    def __init__(self, method: str, scale: float = 0.1):
        super().__init__()
        self.method = method  # "attention", "ffn", "residual"
        self.scale = scale

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        if self.method == "attention":
            return x + self.scale * bias  # Bias attention output
        elif self.method == "ffn":
            return x + self.scale * bias  # Bias FFN output
        elif self.method == "residual":
            return x + self.scale * bias  # Bias residual path
        else:
            raise ValueError(f"Unknown bias method: {self.method}")

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model_cfg = config['model']
        self.embed_dim = self.model_cfg['embed_dim']
        self.context_length = self.model_cfg['context_length']
        self.vocab_size = config['tokenizer'].get('vocab_size', 64)
        
        # Initialize components
        self._init_components(config)
        self._init_configurations(config)
        
        self.bias_injector = BiasInjection(self.model_cfg['bias_injection'], self.model_cfg['bias_scale'])
        
        self.template_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.bias_proj = nn.Linear(self.embed_dim, self.embed_dim)  
        
    def _init_components(self, config: Dict[str, Any]) -> None:
        """Initialize all layer components."""
        # Dynamic attention module loading with error handling
        try:
            attn_module = importlib.import_module(f"{Path(__file__).parent.name}.attention")
            decoder_class = getattr(attn_module, config['decoder']['decoder_class'])
            
            # Initialize attention layers with shared or separate weights
            if config['training'].get('share_attention_weights', False):
                self.attn_masked = decoder_class(config)
                self.attn = self.attn_masked
            else:
                self.attn_masked = decoder_class(config)
                self.attn = decoder_class(config)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to initialize attention module: {str(e)}")
        
        # Feed-forward network
        self.ffn = FeedForward(config)
        
        # Layer Normalizations with configurable epsilon
        norm_eps = config['training'].get('layer_norm_eps', 1e-5)
        self.norm1 = nn.LayerNorm(self.embed_dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=norm_eps)
        self.norm3 = nn.LayerNorm(self.embed_dim, eps=norm_eps)
        
        # Dropout layers
        dropout_p = self.model_cfg.get('dropout', 0.1)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.dropout3 = nn.Dropout(dropout_p)
        
        # Optional residual connections scaling
        self.residual_scaling = config['training'].get('residual_scaling', 1.0)
        
        # Mixed precision support
        self._autocast_kwargs = {
            'device_type': 'cuda' if torch.cuda.is_available() else 'cpu',
            'enabled': config['training'].get('mixed_precision', True)
        }

    def _init_configurations(self, config: Dict[str, Any]) -> None:
        """Initialize training configurations."""
        # Pre-normalization vs post-normalization
        self.pre_norm = config['training'].get('pre_norm', False)
        
        # Gradient checkpointing
        if config['training'].get('gradient_checkpointing', False):
            self.forward = torch.utils.checkpoint.checkpoint(self._forward_impl)
        else:
            self.forward = self._forward_impl

    def _forward_sublayer(self, x, sublayer_fn, norm, dropout, residual, bias=None):
        if self.pre_norm:
            x = norm(x)
        x_out = sublayer_fn(x)
        if bias is not None:
            x_out = self.bias_injector(x_out, bias)  # <-- Unified bias injection
        x_out = dropout(x_out)
        if not self.pre_norm:
            x_out = norm(x_out)
        return residual + (x_out * self.residual_scaling)

    def _forward_impl(self, x: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        embed_bias = self.template_embed(sequences)
        embed_bias = self.bias_proj(embed_bias).sum(dim=-2)  # [10, 10, embed_dim]
        
        # Sublayer 1: Masked Attention
        x = self._forward_sublayer(
            x, 
            lambda x: self.attn_masked(x, mask=True), 
            self.norm1, 
            self.dropout1, 
            x,  # residual
            embed_bias if self.bias_injector.method == "attention" else None
        )
        
        # Sublayer 2: Attention
        x = self._forward_sublayer(
            x,
            self.attn,
            self.norm2,
            self.dropout2,
            x,
            embed_bias if self.bias_injector.method == "attention" else None
        )
        
        # Sublayer 3: FFN
        x = self._forward_sublayer(
            x,
            self.ffn,
            self.norm3,
            self.dropout3,
            x,
            embed_bias if self.bias_injector.method == "ffn" else None
        )
        
        return x

    def get_attention_weights(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return attention weights from both attention layers if available."""
        masked_weights = getattr(self.attn_masked, 'last_attention_weights', None)
        weights = getattr(self.attn, 'last_attention_weights', None)
        return masked_weights, weights

