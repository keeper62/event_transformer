import yaml
import torch
from models.transformer import Transformer

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config("configs/base_config.yaml")['base_config']
    model = Transformer(config)
    sample_input = torch.randint(0, config['vocab_size'], (1, config['max_len']))
    output = model(sample_input)
    print("Model output shape:", output.shape)

if __name__ == "__main__":
    main()