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
        self.scale = self.head_dim ** -0.5
        
        # Projection layers
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=atten_cfg.get('qkv_bias', False))
        self.out_proj = nn.Linear(self.dim, self.dim, bias=atten_cfg.get('out_bias', True))
        self.dropout = nn.Dropout(model_cfg['dropout'])
        
        if model_cfg['bias_injection'] == "attention":
            self.template_embed = nn.Embedding(config['tokenizer'].get('vocab_size', 64), self.dim)
            self.bias_proj = nn.Linear(self.dim, 1)
        
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

    def forward(self, x: torch.Tensor, sequences: torch.Tensor, mask: bool = True) -> torch.Tensor:
        """Forward pass with standard causal masking (no windowing)."""
        B, T, _ = x.shape

        # Project queries, keys, values
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, T, D]
        
        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        
        if hasattr(self, 'template_embed'):
            # Get (B, T_template, D)
            template_emb = self.template_embed(sequences)
            
            # Project and mean over template tokens to get (B, D)
            embed_bias = self.bias_proj(template_emb).mean(dim=1)  # dim=1 is token dim
            
            # Project embed_bias to a scalar per head or per sequence
            # Simple scalar bias added to all attention scores
            bias_scalar = embed_bias.mean(dim=1).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, 1)
            attn_scores = attn_scores + bias_scalar
        
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

class CollaborativeAttention(BaseAttention):
    """Collaborative attention with learned mixing across all heads for q, k, v."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Shared projections (single matrix for queries/keys)
        self.q_shared = nn.Linear(self.dim, self.head_dim, bias=False)  # W˜_Q: Din → D˜_k
        self.k_shared = nn.Linear(self.dim, self.head_dim, bias=False)  # W˜_K: Din → D˜_k
        self.v = nn.ModuleList([nn.Linear(self.dim, self.head_dim, bias=False) for _ in range(self.heads)])            # Separate V (per-head)
        
        # Per-head mixing vectors (m_i ∈ R^D˜_k)
        self.mixing_q = nn.Parameter(torch.randn(self.heads, self.head_dim))  # m_i for Q
        self.mixing_k = nn.Parameter(torch.randn(self.heads, self.head_dim))  # m_i for K
        
        # Output mixing (post-concat)
        self.post_mixing = nn.Linear(self.dim, self.dim, bias=False)
        
        # Initialize
        nn.init.xavier_uniform_(self.q_shared.weight)
        nn.init.xavier_uniform_(self.k_shared.weight)
        for layer in self.v:  # Initialize each head's value projection
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.mixing_q)
        nn.init.xavier_uniform_(self.mixing_k)
        nn.init.xavier_uniform_(self.post_mixing.weight)

    def forward(self, x: torch.Tensor, sequences: torch.Tensor, mask: bool = True) -> torch.Tensor:
        B, T, _ = x.shape
        
        # Shared projections
        q_shared = self.q_shared(x)  # [B, T, D˜_k]
        k_shared = self.k_shared(x)  # [B, T, D˜_k]

        # Apply per-head mixing
        q = torch.einsum('btd,hd->bhtd', q_shared, self.mixing_q)
        k = torch.einsum('btd,hd->bhtd', k_shared, self.mixing_k)

        # Compute per-head v projections
        v = torch.stack([self.v[i](x) for i in range(self.heads)], dim=2)  # [B, T, h, d]
        v = v.permute(0, 2, 1, 3)  # [B, h, T, d]

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if hasattr(self, 'template_embed'):
            # Get (B, T_template, D)
            template_emb = self.template_embed(sequences)
            
            # Project and mean over template tokens to get (B, D)
            embed_bias = self.bias_proj(template_emb).mean(dim=1)  # dim=1 is token dim
            
            # Project embed_bias to a scalar per head or per sequence
            # Option 1: Simple scalar bias added to all attention scores
            bias_scalar = embed_bias.mean(dim=-1).unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B, 1, 1, 1)
            attn_scores = attn_scores + bias_scalar

        if mask:
            causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # Softmax and weighted sum
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # [B, h, T, d]

        # Concatenate and post-mix
        out = out.transpose(1, 2).reshape(B, T, -1)
        out = self.post_mixing(out)
        
        return out
