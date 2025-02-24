import torch
import torch.nn as nn
import importlib
from .token import TokenEmbedding

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size=config['vocab_size'], embed_size=config['hidden_dim'])
        
        self.conf_ape = config['ape']
        
        if config['ape']:
            position_module = importlib.import_module("models.position")
            position_class = getattr(position_module, config["ape_class"])
            self.position_embedding = position_class(d_model=self.token_embedding.embedding_dim, max_len=config['max_len'])
        
        self.dropout = nn.Dropout(config.get('dropout', 0.1))  # Custom feature: dropout
    
    def forward(self, x):
        x = self.token_embedding(x)  # Shape: (batch_size, seq_len, d_model)
        
        if self.conf_ape:
            positions = self.position_embedding(x)
            positions = positions[0]  # Pick the first row to get (seq_len, d_model)
            positions = positions.unsqueeze(0).expand_as(x)  # (batch_size, seq_len, d_model)

            x = x + positions  # Now the shapes match
        return self.dropout(x)  # Apply dropout