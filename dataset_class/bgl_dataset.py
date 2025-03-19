from .abstract_dataset import AbstractBGLDataset
import pandas as pd

class BGLDataset(AbstractBGLDataset):
    def _load_data(self, path):
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
    
        return list(pd.DataFrame(data, columns=self.columns)[self.data_column])
