from .abstract_dataset import AbstractBGLDataset

class BGLDataset(AbstractBGLDataset):
    def _read_data(self, path):
        """Read a specific line and extract only the last column."""
        # Load entire file into memory
        with open(path, "r", encoding="utf8") as f:
            data = [line.split(maxsplit=9)[-1] for line in f]  # Extract last column from each line
        return data
