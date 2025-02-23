import torch
from models.transformer import Transformer

def test_transformer_forward():
    config = {
        'vocab_size': 1000,
        'hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'ffn_dim': 512,
        'max_len': 20
    }
    model = Transformer(config)
    x = torch.randint(0, config['vocab_size'], (1, config['max_len']))
    output = model(x)
    assert output.shape == (1, config['max_len'], config['vocab_size'])

test_transformer_forward()