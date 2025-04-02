import torch.nn as nn
import importlib

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=config['model']['vocab_size'], embedding_dim=config['model']['embed_dim'])
        
        self.conf_ape = config['ape']['ape_class']
        
        if self.conf_ape:
            position_module = importlib.import_module("models.position")
            position_class = getattr(position_module, self.conf_ape)
            self.position_embedding = position_class(d_model=config['model']['embed_dim'], max_len=config['model']['max_len'])

        self.dropout = nn.Dropout(config['model'].get('dropout', 0.1)) 
    
    def forward(self, x):
        x = self.embedding_layer(x)
        
        if self.conf_ape:
            positions = self.position_embedding(x)
            x = x + positions  # Add positional encoding element-wise

        return self.dropout(x.float())  # Apply dropout
