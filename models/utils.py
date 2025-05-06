import os
import glob
import yaml
from typing import Dict, Any
from collections import Counter
import torch
import math


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


def compute_class_weights(dataset: list[int], vocab_size: int, device: torch.device = None) -> torch.Tensor:
    """
    Compute inverse frequency weights normalized so the maximum weight is 1.
    Ensures proper device placement and numerical stability.

    Args:
        dataset (list[int]): List of class indices.
        vocab_size (int): Total number of classes.
        device (torch.device): Device to place weights tensor on. Defaults to CPU.

    Returns:
        torch.Tensor: Weights tensor of shape (vocab_size,) on specified device, scaled to [0, 1].
    """
    if not dataset:
        return torch.ones(vocab_size, dtype=torch.float32, device=device)
    
    # Safely compute counts
    counter = Counter(dataset)
    total_count = max(sum(counter.values()), 1)  # Ensure never zero
    
    # Initialize on CPU first (more efficient for this computation)
    weights = torch.zeros(vocab_size, dtype=torch.float32)
    
    # Compute weights with numerical stability
    for class_idx in range(vocab_size):
        class_count = counter.get(class_idx, 0)
        weights[class_idx] = math.log(total_count / (class_count + 1e-6))
    
    # Normalize and handle edge cases
    max_weight = weights.max()
    if max_weight <= 0:  # All weights zero or negative
        weights = torch.ones_like(weights)
    else:
        weights = weights / max_weight
    
    # Move to specified device if provided
    if device is not None:
        weights = weights.to(device)
    
    return weights
