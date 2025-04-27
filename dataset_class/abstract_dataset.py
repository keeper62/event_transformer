from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

class AbstractBGLDataset(Dataset, ABC):
    def __init__(self, path, prediction_steps, context_length, transform=None, tokenizer=None, test_mode=False): 
        self.path = path
        self.transform = transform
        self.tokenizer = tokenizer
        self.prediction_steps = prediction_steps
        self.context_length = context_length
        self.test_mode = test_mode
        
        if self.test_mode:
            self.data = self._read_data_training(path)
        else:
            self.data = self._read_data(path)
        
        self.data = [self.transform(d) for d in self.data]

        # Unpack and store tensors separately for speed
        self.tokens = torch.tensor(self.data, dtype=torch.long)
        
        self.num_lines = len(self.tokens)

    @abstractmethod
    def _read_data(self, path):
        pass

    @abstractmethod
    def _read_data_training(self, path):
        pass

    def __len__(self):
        return max(0, self.num_lines - self.context_length - self.prediction_steps)

    def __getitem__(self, idx):
        input_window = self.tokens[idx: idx + self.context_length]
        output_window = self.tokens[idx + self.context_length: idx + self.context_length + self.prediction_steps]

        return input_window, output_window

