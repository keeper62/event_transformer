from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

class AbstractBGLDataset(Dataset, ABC):
    def __init__(self, path, prediction_steps, context_length, transform=None, test_mode=False):
        self.path = path
        self.transform = transform
        self.prediction_steps = prediction_steps
        self.context_length = context_length
        self.test_mode = test_mode
        
        if self.test_mode: self.data = self._read_data_training(path) 
        else: self.data = self._read_data(path)
        
        self.num_lines = len(self.data)  # Number of loaded lines

    @abstractmethod
    def _read_data(self, path):
        """Read data and return a list. Must be implemented in subclasses."""
        pass
    
    @abstractmethod
    def _read_data_training(self, path):
        """Read data and return a list. Must be implemented in subclasses for test_mode to work."""
        pass

    def __len__(self):
        return max(0, self.num_lines - self.context_length - self.prediction_steps + 1)

    def __getitem__(self, idx):
        """Dynamically construct input/output sequences per batch."""
        data, timestamps = [d[0] for d in self.data], [d[1] for d in self.data]
        
        input_window = data[idx: idx + self.context_length]
        output = data[idx + self.context_length + 1]
        
        input_timestamps = timestamps[idx: idx + self.context_length]
        
        output_window = input_window[1:] + [output]  # Shifted input window for output

        # Apply transformation if available
        if self.transform:
            input_window = [self.transform(log) for log in input_window]
            output_window = [self.transform(log) for log in output_window]  # No list wrapping needed here

        # Convert to tensor
        input_window = torch.tensor(input_window, dtype=torch.long)
        input_timestamps = torch.tensor(input_timestamps, dtype=torch.float32)
        output_window = torch.tensor(output_window, dtype=torch.long)

        return input_window, output_window, input_timestamps
