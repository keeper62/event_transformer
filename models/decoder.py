from models.feed_forward import FeedForward

import torch
import torch.nn as nn
import importlib
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model_cfg = config['model']
        self.embed_dim = self.model_cfg['embed_dim']
        self.context_length = self.model_cfg['context_length']
        self.tokenizer_length = config['tokenizer'].get('tokenizer_length', 64)
        
        # Initialize components
        self._init_components(config)
        self._init_configurations(config)
        
        # More efficient sequence processing
        self.sequence_processor = nn.Sequential(
            nn.Linear(self.tokenizer_length, self.embed_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dim)
        )
        
        # Skip the embedding layer and process raw token IDs directly
        self.token_embedding = None  # Not needed with this approach

    def _process_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Process sequences without flattening.
        
        Args:
            sequences: (batch_size, context_length, tokenizer_length)
            
        Returns:
            bias: (batch_size, context_length, embed_dim)
        """
        # Convert token IDs to one-hot (batch_size, ctx_len, token_len, vocab_size)
        # Instead we'll process the token patterns directly
        
        # Normalize token IDs to 0-1 range
        sequences = sequences.float() / (self.tokenizer_length - 1)
        
        # Process each position independently
        # (batch_size, ctx_len, token_len) -> (batch_size, ctx_len, embed_dim)
        return self.sequence_processor(sequences)
        
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
        self.pre_norm = config['training'].get('pre_norm', True)
        
        # Gradient checkpointing
        if config['training'].get('gradient_checkpointing', False):
            self.forward = torch.utils.checkpoint.checkpoint(self._forward_impl)
        else:
            self.forward = self._forward_impl

    def _forward_impl(self, x: torch.Tensor, sequences: List[List[int]]) -> torch.Tensor:
        """Main forward implementation with residual connections and RNN bias."""
        with torch.amp.autocast(**self._autocast_kwargs):
            # Get RNN bias terms for the sequences
            rnn_bias = self._process_sequences(sequences)
            
            # First sub-layer: masked self-attention
            residual = x
            if self.pre_norm:
                x = self.norm1(x)
            
            x = self.attn_masked(x, mask=True)
            x = self.dropout1(x)
            x = residual + (x * self.residual_scaling) + rnn_bias
            
            if not self.pre_norm:
                x = self.norm1(x)
            
            # Second sub-layer: self-attention
            residual = x
            if self.pre_norm:
                x = self.norm2(x)
            
            x = self.attn(x)
            x = self.dropout2(x)
            x = residual + (x * self.residual_scaling) + rnn_bias
            
            if not self.pre_norm:
                x = self.norm2(x)
            
            # Third sub-layer: feed forward
            residual = x
            if self.pre_norm:
                x = self.norm3(x)
            
            x = self.ffn(x)
            x = self.dropout3(x)
            x = residual + (x * self.residual_scaling)
            
            if not self.pre_norm:
                x = self.norm3(x)
            
            return x

    def get_attention_weights(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return attention weights from both attention layers if available."""
        masked_weights = getattr(self.attn_masked, 'last_attention_weights', None)
        weights = getattr(self.attn, 'last_attention_weights', None)
        return masked_weights, weights

