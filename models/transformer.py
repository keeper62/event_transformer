import torch.nn as nn
import importlib
from .embeddings import Embeddings


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_layer = Embeddings(config)
        # Load Transformer layers dynamically
        encoder_module = importlib.import_module("models.encoder")
        encoder_class = getattr(encoder_module, "TransformerEncoderLayer")
        self.encoder_layers = nn.ModuleList([
            encoder_class(config) for _ in range(config["num_layers"])
        ])

        self.fc_out = nn.Linear(config["embed_dim"], config['vocab_size'])

    def forward(self, x):
        x = self.embedding_layer(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.fc_out(x)


