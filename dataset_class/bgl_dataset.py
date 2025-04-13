from .abstract_dataset import AbstractBGLDataset

class Dataset(AbstractBGLDataset):
    def _read_data(self, path):
        with open(path, "r", encoding="utf8") as f:
            raw = [line.split(maxsplit=9) for line in f]
            data = [r[-1] for r in raw]
            timestamps = [int(r[1]) for r in raw]
        return list(zip(data, timestamps))

    def _read_data_training(self, path):
        data = []
        with open(path, "r", encoding="utf8") as f:
            for i, line in enumerate(f):
                if i >= 1000:
                    break
                parts = line.split(maxsplit=9)
                data.append((parts[-1], int(parts[1])))
        return data