import torch.nn as nn
import torch
import math

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
    
class LearnableAbsolutePosition(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """Returns learnable positional encodings matching the input sequence length."""
        batch_size, seq_len = x.shape[:2]  # Extract batch size and sequence length
        
        device = self.pos_embedding.weight.device # Get device of the embedding weights
        
        # Generate position indices for sequence length
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)  # (batch_size, seq_len)

        # Get learnable positional embeddings
        return self.pos_embedding(positions)  # Shape: (batch_size, seq_len, d_model)
    
class LearnableRelativePosition(nn.Module):
    def __init__(self, max_len: int, head_dim: int):
        """
        Initializes the relative position encoding module.

        Args:
            max_len (int): Maximum sequence length.
            dim (int): Embedding dimension.
        """
        super().__init__()
        self.max_len = max_len
        self.head_dim = head_dim

        # Learnable relative positional embeddings (max_len * 2 - 1)
        self.relative_position_embeddings = nn.Parameter(
            torch.randn(2 * max_len - 1, head_dim) * 0.02
        )

    def forward(self, seq_len: int):
        """
        Computes relative position encodings for a given sequence length.

        Args:
            seq_len (int): Current sequence length.

        Returns:
            torch.Tensor: Relative position embeddings of shape (seq_len, seq_len, dim).
        """

        # Compute relative position indices
        positions = torch.arange(seq_len, dtype=torch.long, device=self.relative_position_embeddings.device)
        relative_indices = positions[:, None] - positions[None, :]  # (seq_len, seq_len)
        relative_indices += self.max_len - 1  # Shift indices to positive range

        # Get relative position embeddings
        relative_embeddings = self.relative_position_embeddings[relative_indices]  # (seq_len, seq_len, dim)

        return relative_embeddings

class UnixTimeDeltaPosition(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)  # Projects deltas to d_model

    def forward(self, unix_timestamps):
        """
        Args:
            unix_timestamps: [batch_size, seq_len]
        Returns:
            time_embeddings: [batch_size, seq_len, d_model]
        """
        # Normalize each sequence to start at t=0
        timestamps = unix_timestamps - unix_timestamps[:, :1]  # [batch_size, seq_len]

        # Compute deltas (diff between consecutive timestamps)
        deltas = torch.zeros_like(timestamps)
        deltas[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]  # [batch_size, seq_len]

        # Log-scale deltas (add 1 to avoid log(0))
        deltas = torch.log1p(deltas).unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Normalize to [0, 1] range per sequence
        min_vals = deltas.min(dim=1, keepdim=True).values
        max_vals = deltas.max(dim=1, keepdim=True).values
        norm_deltas = (deltas - min_vals) / (max_vals - min_vals + 1e-8)  # Add epsilon to avoid divide-by-zero

        # Project to embedding space
        return self.embedding(norm_deltas)  # [batch_size, seq_len, d_model]