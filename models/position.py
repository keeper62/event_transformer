# Copied from LogBert

import torch.nn as nn
import torch
import math

AP_config = {}

class AbsolutePosition(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class RelativePosition(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Compute relative position encodings
        pe = torch.zeros(2 * max_len - 1, d_model).float()
        position = torch.arange(-(max_len - 1), max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x, q_len=10):
        """Returns relative positional encodings for a given sequence length."""
        pos_idx = torch.arange(q_len).unsqueeze(0) - torch.arange(q_len).unsqueeze(1)
        pos_idx += (self.max_len - 1)  # Shift index to be non-negative
        return self.pe[pos_idx]  # Shape: (q_len, q_len, d_model)

class LearnableAbsolutePosition(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """Returns learnable positional encodings matching the input sequence length."""
        seq_len = x.size(1)  # Get sequence length from input x
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # Shape: (1, seq_len)
        return self.pos_embedding(positions)  # Shape: (1, seq_len, d_model)
    
class KERPLE(nn.Module):
    def __init__(self, d_model, max_len=512, beta=0.5):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.beta = beta  # Kernel scaling factor
        
        # Compute kernel-based relative position encodings
        position = torch.arange(-max_len + 1, max_len, dtype=torch.float).unsqueeze(1)
        self.pe = torch.exp(-self.beta * torch.abs(position))
        self.pe = self.pe / self.pe.sum(dim=0, keepdim=True)  # Normalize
        
        self.register_buffer('pe', self.pe)
    
    def forward(self, q_len):
        """Returns kernel-based relative positional encodings for a given sequence length."""
        pos_idx = torch.arange(q_len).unsqueeze(0) - torch.arange(q_len).unsqueeze(1)
        pos_idx += (self.max_len - 1)  # Shift index to be non-negative
        return self.pe[pos_idx]  # Shape: (q_len, q_len)
    
class T5RPE(nn.Module):
    def __init__(self, d_model, num_buckets=32, max_len=512):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_len
        self.relative_buckets = nn.Embedding(num_buckets, 1)

    def compute_bucket(self, relative_position):
        """Assigns a bucket index based on the relative position following T5-RPE."""
        abs_pos = torch.abs(relative_position)
        
        bucket = torch.where(
            abs_pos < 8, abs_pos,
            torch.minimum(
                torch.tensor(15, device=abs_pos.device),
                8 + torch.ceil(torch.log(abs_pos.float() / 8) / math.log(self.max_distance / 8) * 8)
            ).long()
        )
        
        bucket = torch.where(relative_position < 0, bucket + 16, bucket)
        return bucket
    
    def forward(self, seq_len):
        position_ids = torch.arange(seq_len, device=self.relative_buckets.weight.device)
        relative_positions = position_ids.unsqueeze(0) - position_ids.unsqueeze(1)
        buckets = self.compute_bucket(relative_positions)
        return self.relative_buckets(buckets).squeeze(-1)  # Shape: (seq_len, seq_len)