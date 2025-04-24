import torch.nn as nn
import torch
import math
import numpy as np

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

class TimeAwareRelativePosition(nn.Module):
    def __init__(self, max_len: int, head_dim: int):
        """
        Time-aware sinusoidal relative position encoding based on time deltas.

        Args:
            max_len (int): Maximum sequence length (used for safety, not strictly needed).
            head_dim (int): Dimension of the attention head.
        """
        super().__init__()
        self.max_len = max_len
        self.head_dim = head_dim

        assert head_dim % 2 == 0, "head_dim must be even for sinusoidal encoding."

        # Projection vector to reduce sinusoidal encoding to scalar bias
        self.w = nn.Parameter(torch.randn(head_dim))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute time-aware relative position bias matrix.

        Args:
            timestamps (torch.Tensor): Tensor of shape (seq_len,) with absolute time values (e.g., in seconds).

        Returns:
            torch.Tensor: Bias matrix of shape (seq_len, seq_len) to be added to attention logits.
        """
        seq_len = timestamps.shape[0]
        device = timestamps.device

        # Compute absolute time difference matrix
        delta_t = torch.abs(timestamps[:, None] - timestamps[None, :])  # shape (seq_len, seq_len)

        # Sinusoidal encoding
        d_t = self.head_dim
        enc = torch.zeros((seq_len, seq_len, d_t), device=device)
        position = delta_t.unsqueeze(-1)  # shape (seq_len, seq_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_t, 2, device=device).float() * (-math.log(10000.0) / d_t)
        )
        enc[..., 0::2] = torch.sin(position * div_term)
        enc[..., 1::2] = torch.cos(position * div_term)

        # Project to scalar biases
        bias = torch.einsum("ijh,h->ij", enc, self.w)  # shape (seq_len, seq_len)

        return bias
