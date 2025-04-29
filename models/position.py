import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dynamic length handling."""
    def __init__(self, head_dim: int, max_len: int = 512):
        super().__init__()
        self.head_dim = head_dim
        self.max_len = max_len
        
        # Initialize buffer with maximum length
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, head_dim, 2).float() * (-math.log(10000.0) / head_dim))
        
        pe = torch.zeros(max_len, head_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Positional encodings matching input sequence length
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
        return self.pe[:, :seq_len]  # (1, seq_len, d_model)

class LearnableAbsolutePosition(nn.Module):
    """Learnable absolute positional embeddings with dynamic length handling."""
    def __init__(self, head_dim: int, max_len: int = 512):
        super().__init__()
        self.head_dim = head_dim
        self.max_len = max_len
        self.pos_embedding = nn.Embedding(max_len, head_dim)
        
        # Initialize embeddings properly
        self._init_embeddings()
        
    def _init_embeddings(self) -> None:
        """Initialize embeddings similar to sinusoidal pattern."""
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.head_dim, 2).float() * (-math.log(10000.0) / self.head_dim))
        
        with torch.no_grad():
            self.pos_embedding.weight[:, 0::2] = torch.sin(position * div_term)
            self.pos_embedding.weight[:, 1::2] = torch.cos(position * div_term)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, ...)
        Returns:
            Positional embeddings of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
            
        positions = torch.arange(seq_len, device=x.device).expand(x.size(0), seq_len)
        return self.pos_embedding(positions)  # (batch_size, seq_len, d_model)

class DynamicPositionBias(nn.Module):
    """Dynamic position bias for relative positions with MLP."""
    def __init__(self, head_dim: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.mlp = nn.Sequential(
            nn.Linear(1, head_dim // 2),
            nn.ReLU(),
            nn.Linear(head_dim // 2, head_dim),
            nn.LayerNorm(head_dim)
        )
        
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Convert relative positions to buckets."""
        ret = torch.zeros_like(relative_position)
        n = -torch.ones_like(relative_position)
        
        # Linear buckets close to zero
        num_buckets = self.num_buckets // 2
        ret += (relative_position < num_buckets).long() * relative_position
        
        # Log-spaced buckets beyond linear range
        val_if_large = num_buckets + (
            torch.log(relative_position.float() / num_buckets) / 
            math.log(self.max_distance / num_buckets) * 
            (self.num_buckets - num_buckets)
        ).long()
        
        ret += torch.where(
            relative_position >= num_buckets,
            torch.min(val_if_large, torch.full_like(val_if_large, self.num_buckets - 1)),
            n
        )
        
        return ret
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """Generate position bias for attention scores."""
        device = next(self.parameters()).device
        positions = torch.arange(seq_len, device=device)
        relative_pos = positions[:, None] - positions[None, :]  # (seq_len, seq_len)
        
        # Bucketize relative positions
        rp_bucket = self._relative_position_bucket(relative_pos)
        
        # Get bias values from MLP
        bias = self.mlp(rp_bucket.float().unsqueeze(-1))  # (seq_len, seq_len, head_dim)
        return bias.permute(2, 0, 1)  # (head_dim, seq_len, seq_len) for easy addition

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
