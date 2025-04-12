from .abstract_dataset import AbstractBGLDataset

class Dataset(AbstractBGLDataset):
    def _read_data(self, path):
        """Read a specific line and extract only the last column."""
        # Load entire file into memory
        with open(path, "r", encoding="utf8") as f:
            data = [(line.split(maxsplit=9)[-1], int(line.split(maxsplit=9)[1])) for line in f]  # Extract last column from each line
        return data

    def _read_data_training(self, path):
        """Read only the necessary lines and extract the last column."""
        count = 0
        data = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                if count >= 1000:  # Limit to 200 lines for testing
                    break
                data.append((line.split(maxsplit=9)[-1], int(line.split(maxsplit=9)[1])))  # Extract last column
                count += 1
        return data