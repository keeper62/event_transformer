import torch.nn as nn
import importlib
from .embeddings import Embeddings

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.out_features = config['model']['vocab_size']
        self.n_steps = config['model']['prediction_steps']
        
        self.embedding_layer = Embeddings(config)
        
        decoder_module = importlib.import_module("models.decoder")
        decoder_class = getattr(decoder_module, "TransformerDecoderLayer")
        self.decoder_layers = nn.ModuleList([
            decoder_class(config) for _ in range(config['model']["num_layers"])
        ])
        
        self.fc_out = nn.Linear(config['model']["embed_dim"], self.out_features)

    def predict(self, x):
        # Handle single-sequence input by adding batch dimension
        if x.dim() == 1:  # If input is (T,)
            x = x.unsqueeze(0)  # Convert to (1, T)
            
        results = self.forward(x)
        
        # Remove batch dimension if single-sequence was input
        if results.shape[0] == 1:
            results = results.squeeze(0)  # Convert (1, n_steps, out_features) → (n_steps, out_features)
        
        return results.argmax(dim=-1).squeeze(0).tolist()

    def forward(self, x):
        x = self.embedding_layer(x)  # Apply embedding
            
        for layer in self.decoder_layers:
            x = layer(x)

        # Predict only the last token's output
        logits = self.fc_out(x)  # (batch_size, seq_len, vocab_size)
        return logits



