import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1, num_classes=None, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True,
                            bidirectional=bidirectional)
    
        # Don't use this as it wouldn't be a fair comparison with a masked transformer class    
        self.bidirectional = bidirectional
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, num_classes if num_classes is not None else vocab_size)

    def forward(self, x, _=None):
        """
        x: (batch_size, seq_len) — token indices
        _: ignored (kept for interface compatibility)
        """
        embedded = self.embedding(x)                 # (batch_size, seq_len, embed_dim)
        output, _ = self.lstm(embedded)              # output: (batch_size, seq_len, hidden_dim * num_directions)
        logits = self.fc(output)                     # (batch_size, seq_len, num_classes)
        return logits
