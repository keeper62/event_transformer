import sys
sys.path.insert(1, '.')

import torch
import yaml
from models.transformer import Transformer
import torch.nn as nn
from training.trainer import train_model

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def test_train_model():
    config = load_config("configs/base_config.yaml")['base_config']
    model = Transformer(config).to('cpu')
    dataloader = [(torch.randint(0, config['vocab_size'], (32, 50)), torch.randint(0, config['vocab_size'], (32, 50)))]  # Dummy tokenized data
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    train_model(model, dataloader, optimizer, loss_fn, 'cpu')
    print("Training step executed successfully.")

test_train_model()