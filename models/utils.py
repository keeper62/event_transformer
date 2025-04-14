import yaml
import os
import glob

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def load_all_configs(config_dir="configs"):
    config_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    configs = {}
    for config_file in config_files:
        config_name = os.path.basename(config_file).replace(".yaml", "")
        if config_name == "base_config": continue
        with open(config_file, 'r') as file:
            configs[config_name] = yaml.safe_load(file)
    return configs

from collections import Counter
import torch

def compute_class_weights(dataset, vocab_size, test_mode=False):
    if test_mode: dataset = dataset.dataset
    log_ids = [log_id for log_id, _ in dataset.data]
    counter = Counter(log_ids)

    weights = torch.zeros(vocab_size)
    for class_idx in range(vocab_size):
        weights[class_idx] = 1.0 / (counter[class_idx] + 1e-6)
    weights = weights / weights.sum()
    return weights