from models.feed_forward import FeedForward
import torch
import torch.nn as nn
import importlib
from typing import Callable, Optional, Dict, Any, Tuple
from pathlib import Path

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model_cfg = config['model']
        self.embed_dim = self.model_cfg['embed_dim']
        self.context_length = self.model_cfg['context_length']
        self.vocab_size = config['tokenizer'].get('vocab_size', 64)
        
        # Initialize configurations first
        self._init_configurations(config)
        self._init_components(config)
        
        self.template_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.bias_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _init_configurations(self, config: Dict[str, Any]) -> None:
        """Initialize training configurations."""
        self.pre_norm = self.model_cfg.get('pre_norm', True)
        self.gradient_checkpointing = self.model_cfg.get('gradient_checkpointing', False)
        
    def _init_components(self, config: Dict[str, Any]) -> None:
        """Initialize layer components."""
        try:
            attn_module = importlib.import_module(f"{Path(__file__).parent.name}.attention")
            decoder_class = getattr(attn_module, config['decoder']['decoder_class'])
            self.attn = decoder_class(config)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Attention module init failed: {str(e)}")
        
        self.ffn = FeedForward(config)
        
        norm_eps = self.model_cfg.get('layer_norm_eps', 1e-5)
        self.norm1 = nn.LayerNorm(self.embed_dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=norm_eps)
        self.norm3 = nn.LayerNorm(self.embed_dim, eps=norm_eps)
        
        self.attn_dropout = nn.Dropout(self.model_cfg.get('attn_dropout', 0.1))
        self.ffn_dropout = nn.Dropout(self.model_cfg.get('ffn_dropout', 0.1))
        
        self.residual_scaling = nn.Parameter(
            torch.tensor(config['training'].get('residual_scaling', 1.0))
        )
        self.bias_scale = nn.Parameter(
            torch.tensor(self.model_cfg.get('bias_scale_init', 0.1))
        )

    def _forward_sublayer(
        self,
        x: torch.Tensor,
        sublayer_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        norm: nn.Module,
        dropout: nn.Module,
        sequences: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        attention_mask: bool = False
    ) -> torch.Tensor:
        """Sublayer forward pass with residual connection."""
        residual = x
        
        if self.pre_norm:
            x = norm(x)
        
        x_out = sublayer_fn(x, sequences, attention_mask)
        
        if bias is not None:
            x_out = x_out + self.bias_scale * bias
            
        x_out = dropout(x_out)
        
        if not self.pre_norm:
            x_out = norm(x_out)
            
        return residual + (x_out * torch.sigmoid(self.residual_scaling))

    def _forward_impl(self, x: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        """Implementation of forward pass."""
        embed_bias = None
        if self.model_cfg['bias_injection'] is not None:
            # Take the mean to prevent bias magnitude to grow with sequence length
            embed_bias = self.bias_proj(self.template_embed(sequences)).sum(dim=-2)
        
        # Self-attention
        x = self._forward_sublayer(
            x,
            lambda x, sequences, mask: self.attn(x, sequences, mask=mask),
            self.norm1,
            self.attn_dropout,
            sequences,
            embed_bias if self.model_cfg['bias_injection'] == "attention" else None,
            attention_mask=True
        )
        
        # Cross-attention
        x = self._forward_sublayer(
            x,
            lambda x, sequences, _: self.attn(x, sequences),
            self.norm2,
            self.attn_dropout,
            sequences,
            embed_bias if self.model_cfg['bias_injection'] == "attention" else None
        )
        
        # FFN
        x = self._forward_sublayer(
            x,
            lambda x, _, __: self.ffn(x),
            self.norm3,
            self.ffn_dropout,
            sequences,
            embed_bias if self.model_cfg['bias_injection'] == "ffn" else None
        )
        
        return x
    
    def forward(self, x: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        """Public forward pass with gradient checkpointing support."""
        if self.gradient_checkpointing and self.training:
            # Create a closure for proper checkpointing
            def create_custom_forward():
                def custom_forward(*inputs):
                    x, sequences = inputs
                    return self._forward_impl(x, sequences)
                return custom_forward
            
            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(),
                x,
                sequences,
                use_reentrant=False,  # Recommended for newer PyTorch versions
                preserve_rng_state=True
            )
        return self._forward_impl(x, sequences)