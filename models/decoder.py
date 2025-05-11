from models.feed_forward import FeedForward

import torch
import torch.nn as nn
import importlib
from typing import Callable, Optional, Dict, Any, Tuple
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
        
        # Initialize configurations first (needed for component init)
        self._init_configurations(config)  # <- Now called before components
        self._init_components(config)
        
        self.bias_injector = BiasInjection(self.model_cfg['bias_injection'], self.model_cfg['bias_scale'])
        self.template_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.bias_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _init_configurations(self, config: Dict[str, Any]) -> None:
        """Initialize training configurations before components."""
        # Pre-normalization flag (default to True for modern architectures)
        self.pre_norm = config['training'].get('pre_norm', True)
        
        # Gradient checkpointing
        if config['training'].get('gradient_checkpointing', False):
            self.forward = torch.utils.checkpoint.checkpoint(self._forward_impl)
        else:
            self.forward = self._forward_impl
        
    def _init_components(self, config: Dict[str, Any]) -> None:
        """Initialize layer components with proper typing."""
        try:
            attn_module = importlib.import_module(f"{Path(__file__).parent.name}.attention")
            decoder_class = getattr(attn_module, config['decoder']['decoder_class'])
            self.attn = decoder_class(config)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Attention module init failed: {str(e)}")
        
        self.ffn = FeedForward(config)
        
        norm_eps = config['training'].get('layer_norm_eps', 1e-5)
        self.norm1 = nn.LayerNorm(self.embed_dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=norm_eps)
        self.norm3 = nn.LayerNorm(self.embed_dim, eps=norm_eps)
        
        self.attn_dropout = nn.Dropout(self.model_cfg.get('attn_dropout', 0.1))
        self.ffn_dropout = nn.Dropout(self.model_cfg.get('ffn_dropout', 0.1))
        
        self.residual_scaling = config['training'].get('residual_scaling', 1.0)

    def _forward_sublayer(
        self,
        x: torch.Tensor,
        sublayer_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        norm: nn.Module,
        dropout: nn.Module,
        bias: Optional[torch.Tensor] = None,
        attention_mask: bool = False
    ) -> torch.Tensor:
        """Type-annotated sublayer forward pass."""
        residual_connection = x  # More explicit than modifying input
        
        if self.pre_norm:
            x = norm(x)
        
        x_out = sublayer_fn(x, attention_mask)
        
        if bias is not None:
            x_out = self.bias_injector(x_out, bias)
            
        x_out = dropout(x_out)
        
        if not self.pre_norm:
            x_out = norm(x_out)
            
        return residual_connection + (x_out * self.residual_scaling)

    def _forward_impl(self, x: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        embed_bias = self.bias_proj(self.template_embed(sequences)).sum(dim=-2)
        
        # Sublayer 1: Masked Self-Attention
        x = self._forward_sublayer(
            x,
            lambda x, mask: self.attn(x, mask=mask),
            self.norm1,
            self.attn_dropout,
            embed_bias if self.bias_injector.method == "attention" else None,
            attention_mask=True
        )
        
        # Sublayer 2: Cross-Attention (optional)
        x = self._forward_sublayer(
            x,
            lambda x, _: self.attn(x),  # Ignore mask for cross-attention
            self.norm2,
            self.attn_dropout,
            embed_bias if self.bias_injector.method == "attention" else None
        )
        
        # Sublayer 3: FFN
        x = self._forward_sublayer(
            x,
            lambda x, _: self.ffn(x),  # FFN doesn't use mask
            self.norm3,
            self.ffn_dropout,
            embed_bias if self.bias_injector.method == "ffn" else None
        )
        
        return x
    
    def forward(self, x: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        """Public forward with autocast support."""
        return self._forward_impl(x, sequences)
