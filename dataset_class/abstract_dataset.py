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
        
        self.data = self._read_data(path)
        
        self.data = [(self.template_miner(d), self.tokenizer(d)) for d in self.data]

        # Unpack and store tensors separately for speed
        self.tokens, self.sequences = zip(*self.data)
        self.sequences = torch.stack([torch.tensor(self.sequence, dtype=torch.long) for self.sequence in self.sequences])
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        
        self.num_lines = len(self.tokens)

    @abstractmethod
    def _read_data(self, path):
        pass

    def __len__(self):
        return max(0, self.num_lines - self.context_length - self.prediction_steps)

    def __getitem__(self, idx):
        input_window = self.tokens[idx: idx + self.context_length]
        output_window = self.tokens[idx + self.context_length: idx + self.context_length + self.prediction_steps]
        
        input_sequences = self.sequences[idx: idx + self.context_length]

        return input_window, output_window, input_sequences

