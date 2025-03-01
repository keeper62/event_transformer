import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class BGLDataset(Dataset):
    def __init__(self, path, labels, max_lines=np.inf):
        self.path = path
        self.labels = labels
        self.max_lines = max_lines
        self.data = self.load_data(self.path)

    def load_data(self, path, max_lines=np.inf):
        """Load .log file and extract structured data."""
        data = []
        count_line = 0
        
        with open(path, "r") as file:
            for line in file:
                if count_line > self.max_lines:
                    continue
                parts = line.split(maxsplit=9)  # Split into max 10 parts (first 9 + rest)
                if len(parts) < 10:  
                    parts.extend([""] * (10 - len(parts)))  # Pad if fewer than 10 columns
                data.append(parts)
                count_line += 1
    
        df = pd.DataFrame(data, columns=self.labels)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row  # Model will handle tokenization
