import importlib
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class BaseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conf_rpe = config['rpe']['rpe_class']
        self.dim = config['model']['embed_dim']
        self.heads = config['model']['num_heads']
        self.seq_len = config['model']['context_length']
        self.window_size = config['model']['num_windows']
        self.scale = (self.dim // self.heads) ** -0.5
        

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(config['model']['dropout'])

        if self.conf_rpe:
            position_module = importlib.import_module("models.position")
            position_class = getattr(position_module, self.conf_rpe)
            self.position_embedding = position_class(head_dim=self.heads, max_len=self.seq_len)

    def forward(self, x, timestamps=None, mask=True):
        B, T, C = x.shape
        H, head_dim = self.heads, C // self.heads
        qkv = self.qkv(x).reshape(B, T, 3, H, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        return self.apply_attention(q, k, v, timestamps, mask)

    def apply_attention(self, q, k, v, timestamps=None, mask=True):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        return self.final_attention(attn_scores, v, timestamps, mask, q.shape[2])

    def final_attention(self, attn_scores, v, timestamps=None, mask=True, shape=None):
        if self.conf_rpe == "RelativePositionEmbedding":
            attn_scores += self.position_embedding(shape).permute(2, 0, 1)
        elif self.conf_rpe == "TimeAwareRelativePosition":
            assert timestamps is not None, "timestamps must be provided for TimeAwareRelativePosition"
            time_bias = self.position_embedding(timestamps[0])
            attn_scores += time_bias.unsqueeze(0).unsqueeze(0)

        if mask:
            device = attn_scores.device
            mask_matrix = torch.full((shape, shape), float('-inf'), device=device)
            for i in range(shape):
                start, end = max(0, i - self.window_size), min(shape, i + self.window_size + 1)
                mask_matrix[i, start:end] = 0
            attn_scores += mask_matrix

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v).transpose(1, 2).reshape(v.shape[0], shape, -1)

class MultiHeadAttention(BaseAttention):
    def apply_attention(self, q, k, v, timestamps=None, mask=True):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        return self.final_attention(attn_scores, v, timestamps, mask, q.shape[2])


class MultiHeadDenseCollaboration(BaseAttention):
    def __init__(self, config, timestamps):
        super().__init__(config, timestamps)
        self.conv1 = nn.Conv1d(in_channels=self.heads, out_channels=self.heads, kernel_size=1, groups=self.heads)
        self.conv2 = nn.Conv1d(in_channels=self.heads, out_channels=self.heads, kernel_size=1, groups=self.heads)

    def apply_attention(self, q, k, v, timestamps=None, mask=True):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = self.conv2(F.relu(self.conv1(attn_scores.view(attn_scores.shape[0], self.heads, -1)))).view_as(attn_scores)
        return self.final_attention(attn_scores, v, timestamps, mask, q.shape[2])

# https://arxiv.org/pdf/2006.16362
class CollaborativeAttention(BaseAttention):
    def __init__(self, config, timestamps):
        super().__init__(config, timestamps)
        self.mixing = self.init_mixing_matrix()
        self.content_bias = nn.Linear(self.dim, self.heads, bias=False)

    def init_mixing_matrix(self):
        mixing = torch.zeros(self.heads, self.dim // self.heads)

        dim_head = int(math.ceil((self.dim // self.heads) / self.heads))
        for i in range(self.heads):
            mixing[i, i * dim_head : (i + 1) * dim_head] = 1.0

        return nn.Parameter(mixing)

    def apply_attention(self, q, k, v, timestamps=None, mask=True):
        mixed_query = q * self.mixing.view(self.heads, 1, -1)
        attn_scores = torch.matmul(mixed_query, k.transpose(-2, -1)) * self.scale
    
        k_reshaped = k.permute(0, 2, 1, 3).reshape(k.shape[0], k.shape[2], -1)
        content_bias = self.content_bias(k_reshaped).transpose(-1, -2).unsqueeze(-2)
        attn_scores += content_bias
    
        return self.final_attention(attn_scores, v, timestamps, mask, q.shape[2])
