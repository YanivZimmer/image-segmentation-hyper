import os
import random
from typing import Iterable


class DataSplit:
    def __init__(self):
        pass

    def get_files(self, dir):
        return (
            self.read_file(os.path.join(dir, "test.txt")),
            self.read_file(os.path.join(dir, "val.txt")),
            self.read_file(os.path.join(dir, "train.txt")),
        )

    def create_files(self, samples_dir, target_dir, test_fraction, val_fraction):
        test_data, val_data, train_data = self.split(
            samples_dir, test_fraction, val_fraction
        )
        self.write_files(test_data, os.path.join(target_dir, "test.txt"))
        self.write_files(val_data, os.path.join(target_dir, "val.txt"))
        self.write_files(train_data, os.path.join(target_dir, "train.txt"))

    def split(self, samples_dir, test_fraction, val_fraction):
        data = sorted(os.listdir(samples_dir))
        # Inplace, return None
        random.shuffle(data)
        test_data = data[: int(len(data) * test_fraction)]
        val_data = data[
            int(len(data) * test_fraction) : int(len(data) * test_fraction)
            + int(len(data) * val_fraction)
        ]
        train_data = data[
            int(len(data) * test_fraction) + int(len(data) * val_fraction) :
        ]
        return test_data, val_data, train_data

    def write_files(self, samples: Iterable[str], path: str):
        with open(path, mode="w") as f:
            for s in samples:
                f.write(f"{s}\n")

    def read_file(self, path):
        with open(path, mode="r") as f:
            data = f.readlines()
        data = [d.strip() for d in data]
        return data
