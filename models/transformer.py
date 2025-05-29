import torch
import torch.nn as nn
import importlib
from typing import Dict, Any, Optional, List
from .embeddings import Embeddings

class Transformer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']
        
        # Configuration
        self.out_features: int = model_cfg['vocab_size']
        self.embed_dim: int = model_cfg['embed_dim']
        self.num_layers: int = model_cfg['num_layers']
        
        # Embedding layer with gradient checkpointing option
        self.embedding_layer = Embeddings(config)
        
        # Dynamic decoder layer loading with cache
        decoder_module = importlib.import_module("models.decoder")
        decoder_class = getattr(decoder_module, "TransformerDecoderLayer")
        
        # Create decoder layers with optional gradient checkpointing
        self.decoder_layers = nn.ModuleList([
            self._create_decoder_layer(decoder_class, config, i)
            for i in range(self.num_layers)
        ])
        
        # Final output layer with weight tying option
        self.fc_out = nn.Linear(self.embed_dim, self.out_features)
        if config['training'].get('tie_weights', False):
            self._tie_weights()
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Mixed precision helpers
        self._autocast_kwargs = {
            'device_type': 'cuda' if torch.cuda.is_available() else 'cpu',
            'enabled': config['training'].get('mixed_precision', True)
        }

    def _create_decoder_layer(self, decoder_class, config: Dict[str, Any], layer_idx: int) -> nn.Module:
        """Create a decoder layer with optional gradient checkpointing."""
        layer = decoder_class(config)
        
        # Enable gradient checkpointing for middle layers to save memory
        if config['training'].get('gradient_checkpointing', False) and 0 < layer_idx < self.num_layers - 1:
            layer = torch.utils.checkpoint.checkpoint_wrapper(layer)
            
        return layer

    def _tie_weights(self) -> None:
        """Tie input and output embeddings weights if they have same dimension."""
        if hasattr(self.embedding_layer, 'token_embedding'):
            self.fc_out.weight = self.embedding_layer.token_embedding.weight
            print("Weight tying applied between embedding and output layers")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for different layer types."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embed_dim ** -0.5)

    def forward(self, x: torch.Tensor, sequences: List[List[int]]) -> torch.Tensor:
        """Forward pass with optional mixed precision."""
        with torch.amp.autocast(**self._autocast_kwargs):
            x = self.embedding_layer(x, sequences)
            
            # Process through decoder layers
            for layer in self.decoder_layers:
                x = layer(x, sequences)
                
            return self.fc_out(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, sequences: List[List[int]], temperature: float = 1.0, top_k: Optional[int] = None) -> List[int]:
        """
        Efficient prediction method with sampling options.
        
        Args:
            x: Input token indices of shape (seq_len,) or (batch_size, seq_len)
            temperature: Softmax temperature (1.0 = normal, <1.0 = more conservative)
            top_k: If set, only sample from top k most likely tokens
            
        Returns:
            List of predicted class indices
        """
        # Handle input shapes
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)
            
        # Get logits for last position only
        logits = self.forward(x, sequences)[:, -1, :] / temperature
        
        # Apply top-k filtering if specified
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        preds = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Convert to list and remove batch dim if needed
        result = preds.tolist()
        return result[0] if was_1d else result

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)