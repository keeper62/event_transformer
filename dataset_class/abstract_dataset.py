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
        
        if self.test_mode:
            self.data = self._read_data_training(path)
        else:
            self.data = self._read_data(path)

        # Unpack and store tensors separately for speed
        data, timestamps = zip(*self.data)
        self.tokens = list(data)  # still raw strings or transformed later
        self.timestamps = torch.tensor(timestamps, dtype=torch.float32)
        
        self.num_lines = len(self.tokens)

    @abstractmethod
    def _read_data(self, path):
        pass

    @abstractmethod
    def _read_data_training(self, path):
        pass

    def __len__(self):
        return max(0, self.num_lines - self.context_length - self.prediction_steps + 1)

    def __getitem__(self, idx):
        input_window = self.tokens[idx: idx + self.context_length]
        output = self.tokens[idx + self.context_length + 1]
        output_window = input_window[1:] + [output]

        if self.transform:
            input_window = [self.transform(log) for log in input_window]
            output_window = [self.transform(log) for log in output_window]

        input_window = torch.tensor(input_window, dtype=torch.long)
        output_window = torch.tensor(output_window, dtype=torch.long)

        input_timestamps = self.timestamps[idx: idx + self.context_length]

        return input_window, output_window, input_timestamps

