import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_dim'])
        self.position_embedding = nn.Embedding(config['max_len'], config['hidden_dim'])
    
    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        return self.token_embedding(x) + self.position_embedding(positions)
    
class CustomEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_dim'])
        self.position_embedding = nn.Embedding(config['max_len'], config['hidden_dim'])
        self.dropout = nn.Dropout(config.get('dropout', 0.1))  # Custom feature: dropout
    
    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        embeddings = self.token_embedding(x) + self.position_embedding(positions)
        return self.dropout(embeddings)  # Apply dropout
