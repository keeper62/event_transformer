from .abstract_dataset import AbstractBGLDataset

class Dataset(AbstractBGLDataset):
    def _read_data(self, path):
        with open(path, "r", encoding="utf8") as f:
            raw = [line.split(maxsplit=9) for line in f]
            data = [r[-1] for r in raw]
        return list(data)