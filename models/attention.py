import torch
import torch.nn as nn
import importlib

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config['num_heads']
        self.head_dim = config['hidden_dim'] // self.num_heads
        self.embed_dim = config['hidden_dim']
        self.rpe = config.get('rpe', False)

        assert self.embed_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)

        if self.rpe:
            rpe_module = importlib.import_module("models.position")
            rpe_class = getattr(rpe_module, config['rpe_class'])
            self.rpe_embedding = rpe_class(d_model=self.embed_dim, max_len=config['max_len'])

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (batch, num_heads, seq_len, head_dim)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # Shape: (batch, num_heads, seq_len, seq_len)

        if self.rpe:
            rpe_values = self.rpe_embedding(seq_len)  # Ensure output shape is (seq_len, seq_len, d_model)
            
            # Fix dimensional mismatch: reshape RPE to match attention weights
            rpe_values = rpe_values.permute(2, 0, 1)  # Change to (d_model, seq_len, seq_len)
            rpe_values = rpe_values.mean(dim=0, keepdim=True)  # Reduce d_model to make it (1, seq_len, seq_len)
            attn_weights = attn_weights + rpe_values  # Ensure shapes align

        attn_weights = attn_weights.softmax(dim=-1)

        attn_output = attn_weights @ v  # (batch, num_heads, seq_len, head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        return self.fc_out(attn_output)

