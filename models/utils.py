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
    Compute normalized inverse frequency weights for a dataset.

    Args:
        dataset (list[int]): List of class indices.
        vocab_size (int): Total number of classes in vocabulary.

    Returns:
        torch.Tensor: Tensor of shape (vocab_size,) with weights normalized to [0, 1].
    """
    counter = Counter(dataset)
    weights = torch.zeros(vocab_size, dtype=torch.float32)
    total_count = sum(counter.values())

    for class_idx in range(vocab_size):
        weights[class_idx] = total_count / (counter[class_idx] + 1e-6)

    return weights / weights.max()
