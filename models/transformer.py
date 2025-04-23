import torch
import torch.nn as nn
import importlib
from typing import Dict, Any
from .embeddings import Embeddings

class Transformer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        model_cfg = config['model']
        self.out_features: int = model_cfg['vocab_size']
        self.n_steps: int = model_cfg['prediction_steps']
        
        # Embedding layer
        self.embedding_layer: nn.Module = Embeddings(config)
        
        # Dynamically load decoder layer class
        decoder_module = importlib.import_module("models.decoder")
        decoder_class = getattr(decoder_module, "TransformerDecoderLayer")
        
        self.decoder_layers: nn.ModuleList = nn.ModuleList([
            decoder_class(config) for _ in range(model_cfg["num_layers"])
        ])
        
        self.fc_out: nn.Linear = nn.Linear(model_cfg["embed_dim"], self.out_features)

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer model.

        Args:
            x (torch.Tensor): Token indices of shape (batch_size, seq_len)
            timestamps (torch.Tensor): Timestamps for positional encoding

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size)
        """
        x = self.embedding_layer(x, timestamps)

        for layer in self.decoder_layers:
            x = layer(x)

        logits = self.fc_out(x)
        return logits

    def predict(self, x: torch.Tensor, timestamps: torch.Tensor|None = None) -> list[int]:
        """
        Predict method for autoregressive decoding or classification.

        Args:
            x (torch.Tensor): Input token indices of shape (seq_len,) or (batch_size, seq_len)

        Returns:
            list[int]: Predicted class indices
        """
        # Handle single-sequence input
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, seq_len)

        # Dummy timestamps for inference if not provided
        if timestamps == None:
            timestamps = torch.zeros_like(x, dtype=torch.float32)

        logits = self.forward(x, timestamps)[:, -1, :]

        # Remove batch dimension if single sequence
        if logits.shape[0] == 1:
            logits = logits.squeeze(0)  # (seq_len, vocab_size)

        return logits.argmax(dim=-1).tolist()
