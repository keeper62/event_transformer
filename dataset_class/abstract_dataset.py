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

        # Preprocess once here
        if self.transform:
            self.data = [(self.transform(log), ts) for log, ts in self.data]

        self.num_lines = len(self.data)

    def __len__(self):
        return max(0, self.num_lines - self.context_length - self.prediction_steps + 1)

    def __getitem__(self, idx):
        data, timestamps = zip(*self.data)

        input_window = data[idx: idx + self.context_length]
        output = data[idx + self.context_length + 1]

        output_window = list(input_window[1:]) + [output]
        input_timestamps = timestamps[idx: idx + self.context_length]

        input_window = torch.tensor(input_window, dtype=torch.long)
        output_window = torch.tensor(output_window, dtype=torch.long)
        input_timestamps = torch.tensor(input_timestamps, dtype=torch.float32)

        return input_window, output_window, input_timestamps


