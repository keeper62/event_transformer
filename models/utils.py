import os
import glob
import yaml
from typing import Dict, Any
from collections import Counter
import torch
import math
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict[str, Any]: Parsed configuration as a dictionary.
    """
    config_path = Path(config_path)
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


def compute_class_weights(dataset, vocab_size):
    counts = Counter(dataset)
    weights = torch.zeros(vocab_size)
    for i in range(vocab_size):
        weights[i] = 1 / (counts.get(i, 0) + 1e-6)  # More aggressive weighting
    return weights / weights.max()
