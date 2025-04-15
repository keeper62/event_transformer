from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

class AbstractBGLDataset(Dataset, ABC):
    def __init__(self, path, prediction_steps, context_length, transform=None, test_mode=False, time_window=5.0): #time window in seconds
        self.path = path
        self.transform = transform
        self.prediction_steps = prediction_steps
        self.context_length = context_length
        self.test_mode = test_mode
        
        if self.test_mode:
            self.data = self._read_data_training(path)
        else:
            self.data = self._read_data(path)
            
        self.data = self._preprocess(self.data, time_window)  # Preprocess the data if needed

        # Unpack and store tensors separately for speed
        data, timestamps = zip(*self.data)
        self.tokens = torch.tensor(data, dtype=torch.long)
        self.timestamps = torch.tensor(timestamps, dtype=torch.float32)
        
        self.num_lines = len(self.tokens)

    @abstractmethod
    def _read_data(self, path):
        pass

    @abstractmethod
    def _read_data_training(self, path):
        pass

    def _preprocess(self, data, time_window):
        """
        Remove temporally correlated duplicates while preserving the original order after sorting by timestamp.
        The function checks if the difference between the timestamps of two consecutive events is greater than the specified time window.
        It can take as input either a list of event IDs or logs which will be transformed using the provided transform function.
    
        Args:
            data (List[Tuple[str, float]]): List of (event_id, timestamp) tuples.
            time_window (float): Time window in seconds to consider as duplicate.
    
        Returns:
            List[Tuple[str, float]]: Filtered list with duplicates removed.
        """
        from collections import defaultdict
    
        # Sort by timestamp (Unix time)
        data = sorted(data, key=lambda x: x[1])
    
        last_seen = defaultdict(lambda: -float('inf'))
        filtered_logs = []
    
        for event_id, timestamp in data:
            if self.transform:
                event_id = self.transform(event_id)
    
            if timestamp - last_seen[event_id] > time_window:
                filtered_logs.append((event_id, timestamp))
                last_seen[event_id] = timestamp
    
        return filtered_logs

    def __len__(self):
        return max(0, self.num_lines - self.context_length - self.prediction_steps + 1)

    def __getitem__(self, idx):
        input_window = self.tokens[idx: idx + self.context_length]
        output = self.tokens[idx + self.context_length + 1]
        output_window = torch.cat((input_window[1:], output.unsqueeze(0)))

        input_timestamps = self.timestamps[idx: idx + self.context_length]

        return input_window, output_window, input_timestamps

