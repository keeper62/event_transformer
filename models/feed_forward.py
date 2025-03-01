import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config['embed_dim'], config['ffn_dim'])
        self.fc2 = nn.Linear(config['ffn_dim'], config['embed_dim'])
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))