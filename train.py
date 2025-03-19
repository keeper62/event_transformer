import torch
from models import Transformer, LogTokenizer, load_all_configs
from dataset_class.bgl_dataset import BGLDataset
import torch.nn as nn
from training.trainer import train_model
from torch.utils.data import DataLoader
import pickle
import time

def train_with_config(config, config_name):
    tokenizer = LogTokenizer()
    
    dataset = BGLDataset(path=config['dataset']['path'], 
                         columns=config['dataset']['columns'], 
                         transform=tokenizer.transform(), 
                         max_lines=3000,
                         data_column="Content")
    dataset.construct_steps(config['model']['prediction_steps'], config['model']['context_length'])
    
    tokenizer.train_template_miner(dataset.data)
    config['model']['vocab_size'] = tokenizer.get_vocab_size()
    
    model = Transformer(config)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training'].get('lr', 1e-4))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)
    
    results = train_model(model, dataloader, optimizer, loss_fn, config['device'], config['model']['vocab_size'], config_name,
                          save_model=True, visualize=False, num_epochs=config['training'].get('num_epochs', 10))
    
    return {epoch: {"targets": res["targets"], "predictions": res["predictions"]} for epoch, res in results.items()}

def train_multiple_configs():
    config_data = load_all_configs()
    all_results = {}
    
    for config_name, config in config_data.items():
        print(f"Training with configuration: {config_name}")
        all_results[config_name] = train_with_config(config['base_config'], config_name)
    
    return all_results

if __name__ == "__main__":
    results = train_multiple_configs()
    with open(f'results/{int(time.time())}_data.p', 'wb') as fp:
        pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("Training completed successfully.")
