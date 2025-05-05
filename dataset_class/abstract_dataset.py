from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

class AbstractBGLDataset(Dataset, ABC):
    def __init__(self, path, prediction_steps, context_length, template_miner=None, tokenizer=None, test_mode=False): 
        self.path = path
        self.template_miner = template_miner
        self.tokenizer = tokenizer
        self.prediction_steps = prediction_steps
        self.context_length = context_length
        self.test_mode = test_mode
        
        self.data = self._read_data(path)  # -> list of grouped (event_id, message) lists

        # Process each group into tokens + sequences
        self.grouped_data = []
        for group in self.data:
            processed = [
                (int(i), self.tokenizer(d)) for i, d in group
            ]
            tokens, sequences = zip(*processed)
            sequences = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in sequences])
            tokens = torch.tensor(tokens, dtype=torch.long)
            self.grouped_data.append((tokens, sequences))

        # Precompute total number of available samples per group
        self.group_lengths = [
            max(0, len(group[0]) - self.context_length - self.prediction_steps)
            for group in self.grouped_data
        ]

        # Map flat index to (group_idx, local_idx)
        self.sample_index = []
        for group_idx, length in enumerate(self.group_lengths):
            for local_idx in range(length):
                self.sample_index.append((group_idx, local_idx))
                
        del self.data
        del self.group_lengths

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        group_idx, local_idx = self.sample_index[idx]
        tokens, sequences = self.grouped_data[group_idx]

        input_window = tokens[local_idx : local_idx + self.context_length]
        output_window = tokens[local_idx + self.context_length : local_idx + self.context_length + self.prediction_steps]
        input_sequences = sequences[local_idx : local_idx + self.context_length]

        return input_window, output_window, input_sequences

    @abstractmethod
    def _read_data(self, path):
        pass
