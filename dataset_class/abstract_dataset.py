from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset

class AbstractBGLDataset(Dataset, ABC):
    def __init__(self, path, columns, data_column, transform=None, max_lines=np.inf):
        self.path = path
        self.columns = columns
        self.max_lines = max_lines
        self.transform = transform
        self.data_column = data_column
        
        self.data = self._load_data(self.path)  # Must return a list
        self.training_data = None

    @abstractmethod
    def _load_data(self, path):
        """Load data and return a list. Must be implemented in subclasses."""
        pass

    def construct_steps(self, prediction_steps, context_length):
        windows = []
        
        for i in range(len(self.data) - context_length - prediction_steps + 1):
            input_window = self.data[i:i+context_length]
            output_window = self.data[i+context_length:i+context_length+prediction_steps]
            windows.append((input_window, output_window))
        
        self.training_data = windows

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        assert self.training_data != None, "construct_steps() needs to be called beforehand"
        input, output = self.training_data[idx] 

        # Apply transform function to tokenize sequences
        if self.transform:
            input = [self.transform(log) for log in input]
            output = [self.transform(label) for label in output]
            
        input, output = torch.tensor(input, dtype=torch.long), torch.tensor(output, dtype=torch.long)
    
        return input, output
