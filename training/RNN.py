import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1, num_classes=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes if num_classes is not None else vocab_size)

    def forward(self, x, _=None):
        """
        x: (batch_size, seq_len)  # token indices
        _: optional placeholder to match TransformerLightning forward() signature
        """
        embedded = self.embedding(x)                 # (batch_size, seq_len, embed_dim)
        output, hidden = self.rnn(embedded)          # output: (batch_size, seq_len, hidden_dim)
        logits = self.fc(output)                     # (batch_size, seq_len, num_classes)
        return logits
