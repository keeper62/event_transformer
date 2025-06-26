import torch
import torch.nn as nn
import importlib
from typing import Dict, Any, Optional
from pathlib import Path

class Embeddings(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model_cfg = config['model']
        self.embed_dim: int = self.model_cfg['embed_dim']
        
        # Token embedding with padding_idx and scaling
        self.token_embedding = nn.Embedding(
            num_embeddings=self.model_cfg['vocab_size'],
            embedding_dim=self.embed_dim,
            padding_idx=0,  # Assuming 0 is padding index
            scale_grad_by_freq=config['training'].get('scale_grad_by_freq', False)
        )
        
        # Initialize embeddings properly
        self._init_embeddings(self.model_cfg.get('embed_init', 'normal'))
        
        # Positional embedding configuration
        self._init_positional_embeddings(config)
        
        # LayerNorm and Dropout
        self.layer_norm = nn.LayerNorm(self.embed_dim) if config['training'].get('pre_norm', True) else None
        self.dropout = nn.Dropout(p=self.model_cfg.get('dropout', 0.1))
        
        # Mixed precision support
        self._autocast_kwargs = {
            'device_type': 'cuda' if torch.cuda.is_available() else 'cpu',
            'enabled': config['training'].get('mixed_precision', True)
        }
        
        self.bias_scale = nn.Parameter(
            torch.tensor(self.model_cfg.get('bias_scale_init', 0.1))
        )
        
        self.template_embed = nn.Embedding(config['tokenizer']['vocab_size'], self.embed_dim)
        #self.bias_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.sequence_proj = nn.Linear(self.model_cfg['context_length'], 1)  

    def _init_embeddings(self, init_type: str) -> None:
        """Initialize embedding weights based on config."""
        if init_type == 'normal':
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.embed_dim**-0.5)
        elif init_type == 'xavier':
            nn.init.xavier_uniform_(self.token_embedding.weight)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(self.token_embedding.weight)
        
        # Zero out padding embeddings if specified
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].fill_(0)

    def _init_positional_embeddings(self, config: Dict[str, Any]) -> None:
        """Initialize positional embeddings dynamically."""
        self.position_embedding = None
        ape_config = config.get('position', {})
        
        if ape_config.get('ape_class', None) is not None:
            try:
                position_module = importlib.import_module(f"{Path(__file__).parent.name}.position")
                position_class = getattr(position_module, ape_config['ape_class'])
                self.position_embedding = position_class(
                    head_dim=self.embed_dim,
                    max_len=config['model']['max_len'],
                    **ape_config.get('params', {})
                )
                
                # Freeze positional embeddings if specified
                if ape_config.get('freeze', False):
                    for param in self.position_embedding.parameters():
                        param.requires_grad = False
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not initialize positional embeddings: {str(e)}")

    def forward(self, x: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional mixed precision and normalization."""
        with torch.amp.autocast(**self._autocast_kwargs):
            # Token embeddings
            x = self.token_embedding(x)  # (B, T) -> (B, T, D)
            
            # Scale embeddings if specified
            if hasattr(self, 'embed_scale'):
                x = x * self.embed_scale
            
            # Positional embeddings
            if self.position_embedding is not None:
                positions = self.position_embedding(x)
                x = x + positions.to(x.device)  # Ensure same device
                
            if self.model_cfg['bias_injection'] == "embedding":
                embed = self.template_embed(sequences)
                t_transposed = embed.permute(0, 3, 1, 2)     
                hm = self.sequence_proj(t_transposed).squeeze(-1)
                output = hm.permute(0, 2, 1)      
                x = x + self.bias_scale * output
                
            # Layer normalization before dropout if pre_norm
            if self.layer_norm is not None:
                x = self.layer_norm(x)
                
            return self.dropout(x)

    def get_output_dim(self) -> int:
        """Return the output dimension of the embeddings."""
        return self.embed_dim

    def load_pretrained_embeddings(self, weights: torch.Tensor, freeze: bool = False) -> None:
        """Load pretrained embedding weights."""
        if weights.shape != self.token_embedding.weight.shape:
            raise ValueError(f"Shape mismatch: expected {self.token_embedding.weight.shape}, got {weights.shape}")
            
        self.token_embedding.weight.data.copy_(weights)
        if freeze:
            self.token_embedding.weight.requires_grad = False