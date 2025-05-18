import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from typing import Dict, Any
from pathlib import Path

class BaseAttention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        model_cfg = config['model']
        atten_cfg = config['attention']
        
        # Core attention parameters
        self.dim = model_cfg['embed_dim']
        self.heads = model_cfg['num_heads']
        self.head_dim = self.dim // self.heads
        self.seq_len = model_cfg['context_length']
        self.window_size = model_cfg['num_windows']
        self.scale = self.head_dim ** -0.5
        
        # Projection layers
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=atten_cfg.get('qkv_bias', False))
        self.out_proj = nn.Linear(self.dim, self.dim, bias=atten_cfg.get('out_bias', True))
        self.dropout = nn.Dropout(model_cfg['dropout'])
        
        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize attention parameters."""
        nn.init.xavier_uniform_(self.qkv.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, mask: bool = True) -> torch.Tensor:
        """Forward pass with standard causal masking (no windowing)."""
        B, T, _ = x.shape

        # Project queries, keys, values
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, T, D]
        
        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply standard causal mask (upper triangular)
        if mask:
            causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = attn_weights @ v
        out = out.transpose(1, 2).reshape(B, T, -1)
        
        return self.out_proj(out)

    def get_attention_weights(self, x: torch.Tensor, mask: bool = True) -> torch.Tensor:
        """Return attention weights with same masking as forward pass."""
        with torch.no_grad():
            B, T, _ = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            
            attn_scores = (q @ k.transpose(-2, -1)) * self.scale
            
            if mask:
                causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(diagonal=1)
                attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            
            return F.softmax(attn_scores, dim=-1)

class MultiHeadAttention(BaseAttention):
    """Standard multi-head attention with optional memory-efficient implementation."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if config['attention'].get('memory_efficient', False):
            self.forward = self._memory_efficient_forward

    def _memory_efficient_forward(self, x: torch.Tensor, mask: bool = True) -> torch.Tensor:
        """Memory-efficient implementation using chunking."""
        B, T, _ = x.shape
        chunk_size = self.config['attention'].get('chunk_size', 256)
        
        # Process in chunks if sequence is long
        if T <= chunk_size:
            return super().forward(x, mask)
            
        # Split into chunks
        chunks = x.split(chunk_size, dim=1)
        outputs = []
        
        for chunk in chunks:
            outputs.append(super().forward(chunk, mask))
            
        return torch.cat(outputs, dim=1)


class MultiHeadDenseCollaboration(BaseAttention):
    """Multi-head attention with dense collaboration between heads."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.inter_head_dim = config['attention'].get('inter_head_dim', 64)
        
        # Head collaboration layers
        self.head_collab1 = nn.Conv1d(
            in_channels=self.heads,
            out_channels=self.inter_head_dim,
            kernel_size=1,
            groups=1
        )
        self.head_collab2 = nn.Conv1d(
            in_channels=self.inter_head_dim,
            out_channels=self.heads,
            kernel_size=1,
            groups=1
        )
        
        # Initialize collaboration layers
        nn.init.kaiming_normal_(self.head_collab1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.head_collab2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor, mask: bool = True) -> torch.Tensor:
        B, T, _ = x.shape
        
        # Project queries, keys, values
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Compute base attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply head collaboration
        attn_scores = attn_scores.transpose(1, 2).reshape(B*T, self.heads, -1)
        attn_scores = self.head_collab2(F.relu(self.head_collab1(attn_scores)))
        attn_scores = attn_scores.reshape(B, T, self.heads, -1).transpose(1, 2)
        
        if mask:
            attn_scores = self._apply_attention_mask(attn_scores)
        
        # Compute final output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        
        return self.out_proj(out)


class CollaborativeAttention(BaseAttention):
    """Collaborative attention with learned mixing between heads."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mixing = nn.Parameter(torch.eye(self.heads, self.head_dim))
        self.content_bias = nn.Linear(self.dim, self.heads, bias=False)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.mixing)
        nn.init.xavier_uniform_(self.content_bias.weight)

    def forward(self, x: torch.Tensor, mask: bool = True) -> torch.Tensor:
        B, T, _ = x.shape
        
        # Project queries, keys, values
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Apply learned mixing to queries
        mixed_q = torch.einsum('bhnd,hd->bhnd', q, self.mixing)
        
        # Compute attention scores
        attn_scores = torch.matmul(mixed_q, k.transpose(-2, -1)) * self.scale
        
        # Add content-based bias
        k_reshaped = k.permute(0, 2, 1, 3).reshape(B, T, -1)
        content_bias = self.content_bias(k_reshaped).transpose(-1, -2).unsqueeze(-2)
        attn_scores += content_bias
        
        if mask:
            attn_scores = self._apply_attention_mask(attn_scores)
        
        # Compute final output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        
        return self.out_proj(out)