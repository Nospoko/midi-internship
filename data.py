import pandas as pd
from datasets import load_dataset


class MyDataset:
    def __init__(self, path: str = "roszcz/internship-midi-data-science"):
        self.dataset = load_dataset(path, split="train")
        self.record = self.dataset[0]

    def __getitem__(self, index):
        return pd.DataFrame(self.dataset[index]["notes"])

    def __len__(self):
        return len(self.dataset)
