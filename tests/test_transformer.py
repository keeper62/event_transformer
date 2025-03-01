import sys
sys.path.insert(1, '.')

import torch
import yaml
from models.transformer import Transformer
from models.utils import Vocabulary
from data.bgl_dataset import BGLDataset
import torch.nn as nn
from training.trainer import train_model
from torch.utils.data import DataLoader

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
col_names = [
    "LineId", "Label", "Timestamp", "Date", "Node", "Time", 
    "NodeRepeat", "Type", "Component", "Level", "Content"
]

def test_train_model():
    config = load_config("configs/base_config.yaml")['base_config']
    model = Transformer(config).to('cpu')
    vocab = Vocabulary()
    dataset = BGLDataset(path='/data/BGL/BGL.log', labels=col_names)
    texts = [dataset[i][0] for i in range(len(dataset))]
    vocab.build_vocab(texts)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    #[(torch.randint(0, config['vocab_size'], (32, 50)), torch.randint(0, config['vocab_size'], (32, 50)))]  # Dummy tokenized data
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    train_model(model, dataloader, optimizer, loss_fn, 'cpu', vocab)
    print("Training step executed successfully.")

test_train_model()