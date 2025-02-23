import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config['num_heads']
        self.head_dim = config['hidden_dim'] // self.num_heads
        self.qkv = nn.Linear(config['hidden_dim'], config['hidden_dim'] * 3)
        self.fc_out = nn.Linear(config['hidden_dim'], config['hidden_dim'])
    
    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        # Compute attention here (simplified)
        return self.fc_out(qkv[0])