from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

class AbstractBGLDataset(Dataset, ABC):
    def __init__(self, path, prediction_steps, context_length, transform=None):
        self.path = path
        self.transform = transform
        self.prediction_steps = prediction_steps
        self.context_length = context_length
        
        self.data = self._read_data(path)
        
        self.num_lines = len(self.data)  # Number of loaded lines

    @abstractmethod
    def _read_data(self, path):
        """Read data and return a list. Must be implemented in subclasses."""
        pass

    def __len__(self):
        return self.num_lines - self.context_length - self.prediction_steps + 1

    def __getitem__(self, idx):
        """Dynamically construct input/output sequences per batch."""
        input_window = self.data[idx: idx + self.context_length]
        output_window = self.data[idx + self.context_length: idx + self.context_length + self.prediction_steps]

        # Apply transformation if available
        if self.transform:
            input_window = [self.transform(log) for log in input_window]
            output_window = [self.transform(log) for log in output_window]

        # Convert to tensor
        input_window = torch.tensor(input_window, dtype=torch.long)
        output_window = torch.tensor(output_window, dtype=torch.long)

        return input_window, output_window