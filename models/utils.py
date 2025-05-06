import os
import glob
import yaml
from typing import Dict, Any
from collections import Counter
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict[str, Any]: Parsed configuration as a dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_all_configs(config_dir: str = "configs") -> Dict[str, Dict[str, Any]]:
    """
    Load all YAML configuration files in a directory, excluding 'base_config.yaml'.

    Args:
        config_dir (str): Directory containing config files.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping from config filename (without extension) to config dict.
    """
    config_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    configs = {}

    for config_file in config_files:
        config_name = os.path.basename(config_file).replace(".yaml", "")
        if config_name == "base_config":
            continue
        with open(config_file, 'r') as file:
            configs[config_name] = yaml.safe_load(file)

    return configs


def compute_class_weights(dataset: list[int], vocab_size: int) -> torch.Tensor:
    """
    Compute inverse frequency weights normalized so the maximum weight is 1.

    Args:
        dataset (list[int]): List of class indices.
        vocab_size (int): Total number of classes.

    Returns:
        torch.Tensor: Weights tensor of shape (vocab_size,), scaled to [0, 1].
    """
    counter = Counter(dataset)
    total_count = sum(counter.values())  # Keep as Python integer
    weights = torch.zeros(vocab_size, dtype=torch.float32)

    for class_idx in range(vocab_size):
        class_count = counter.get(class_idx, 0)
        # Compute using Python scalars, then convert to tensor
        weights[class_idx] = torch.log(torch.tensor(total_count / (class_count + 1e-6), dtype=torch.float32))

    # Handle case where all weights are zero (unlikely but possible)
    if weights.max() == 0:
        return torch.ones_like(weights)
    
    return weights / weights.max()
