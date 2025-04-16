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

def compute_class_weights(dataset, vocab_size):
    log_ids = [log_id for log_id, _ in dataset.dataset.data]
    counter = Counter(log_ids)

    weights = torch.zeros(vocab_size)
    for class_idx in range(vocab_size):
        weights[class_idx] = 1.0 / (counter[class_idx] + 1e-6)
    weights = weights / weights.sum()
    return weights

import random
import numpy as np
from scipy.stats import entropy
from collections import defaultdict

def window_sequence(seq, window_size, stride=1):
    return [seq[i:i+window_size] for i in range(0, len(seq) - window_size + 1, stride)]

def window_entropy(window):
    event_ids = [x[0] for x in window]  # Extract just the event_id
    counts = Counter(event_ids)
    probs = np.array(list(counts.values())) / len(window)
    return entropy(probs)

def reduce_sequence_by_entropy(seq, window_size=10, stride=1, num_bins=5):
    windows = window_sequence(seq, window_size, stride)
    entropies = [window_entropy(w) for w in windows]

    # Bin windows by entropy (from low to high diversity)
    bin_edges = np.linspace(min(entropies), max(entropies), num_bins + 1)
    binned_indices = defaultdict(list)
    
    for idx, ent in enumerate(entropies):
        for b in range(num_bins):
            if bin_edges[b] <= ent < bin_edges[b + 1]:
                binned_indices[b].append(idx)
                break

    # Balance: sample equally from all entropy bins
    min_size = min(len(v) for v in binned_indices.values() if len(v) > 0)
    sampled_window_idxs = []
    for idxs in binned_indices.values():
        if len(idxs) >= min_size:
            sampled_window_idxs.extend(random.sample(idxs, min_size))

    # Determine which indices from the original sequence to keep
    indices_to_keep = set()
    for i in sampled_window_idxs:
        indices_to_keep.update(range(i, i + window_size))

    # Return the full tuples for kept indices
    return [seq[i] for i in sorted(indices_to_keep)]