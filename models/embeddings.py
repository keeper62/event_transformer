import torch
import torch.nn as nn
import importlib
from .token import LogTokenizer

import importlib
import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = LogTokenizer(vocab_size=config['vocab_size'], bpe_max_len=config['max_len'])
        
        self.embedding_layer = nn.Embedding(num_embeddings=config['vocab_size'], embedding_dim=config['embed_dim'])
        
        self.conf_ape = config['ape']
        
        if config['ape']:
            position_module = importlib.import_module("models.position")
            position_class = getattr(position_module, config["ape_class"])
            self.position_embedding = position_class(d_model=config['embed_dim'], max_len=config['max_len'])  # Use correct d_model

        self.dropout = nn.Dropout(config.get('dropout', 0.1))  # Custom feature: dropout
    
    def obtain_token_embeddings(self, x):
        self.token_embedding.train_tokenizer(x)
        return self.token_embedding.obtain_token_embedding(x)
    
    def forward(self, x):
        x = self.obtain_token_embeddings(x)  # Shape: (batch_size, seq_len)
        
        x = self.embedding_layer(x)  # Apply linear projection
        
        batch_size, _, _ = x.shape  # Extract sequence length

        if self.conf_ape:
            positions = self.position_embedding(x)  # Fix: Only pass `seq_len`
            positions = positions.expand(batch_size, -1, -1)  # Expand for batch dimension if necessary
            x = x + positions  # Add positional encoding element-wise

        return self.dropout(x)  # Apply dropout
