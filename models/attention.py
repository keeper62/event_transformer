import importlib
import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['embed_dim'] % config['num_heads'] == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = config['embed_dim']  # Total embedding size
        self.num_heads = config['num_heads']
        self.head_dim = config['embed_dim'] // config['num_heads']  # Each head gets this size
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention

        self.rpe = config['rpe']
        
        # Linear layer for Q, K, V projection (maps embed_dim -> 3 * embed_dim)
        self.qkv = nn.Linear(config['embed_dim'], 3 * config['embed_dim'])
        self.fc_out = nn.Linear(config['embed_dim'], config['embed_dim'])  # Final projection

        if self.rpe:
            position_module = importlib.import_module("models.position")
            position_class = getattr(position_module, config["rpe_class"])
            self.position_embedding = position_class(head_dim=self.head_dim, max_len=config.get('max_len', 512))

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape  
        assert embed_dim == self.embed_dim, f"Expected embedding dim {self.embed_dim}, but got {embed_dim}"

        # Compute Q, K, V
        qkv = self.qkv(x.float())  # Shape: (batch_size, seq_len, 3 * embed_dim)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)

        # Scaled Dot-Product Attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)

        if self.rpe:
            pe = self.position_embedding(seq_len).to(q.device)  # (seq_len, seq_len, head_dim)
            pe = pe.permute(2, 0, 1)  # (head_dim, seq_len, seq_len)

            # Ensure q and pe are aligned
            # Note: try to understand how this works
            pe_q = torch.einsum('cxy, bnxc->bnxy', pe, q) * self.scale  # (batch_size, num_heads, seq_len, seq_len)

            attn_weights = attn_weights + pe_q  # Apply positional encoding bias

        # Compute Attention Output
        attn_output = attn_weights @ v  # (batch_size, num_heads, seq_len, head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)  

        return self.fc_out(attn_output)  # Final linear projection
