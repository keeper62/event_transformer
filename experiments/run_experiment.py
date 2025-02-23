import yaml
from models.transformer import Transformer

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config("configs/base_config.yaml")
    model = Transformer(config)
    print(model)

if __name__ == "__main__":
    main()