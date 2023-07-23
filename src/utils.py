import pandas as pd
from datasets import load_dataset


def load_records(huggingface_path: str):
    records = []

    dataset = load_dataset(huggingface_path, split="train")

    for d in dataset:
        # cast record as pandas dataframe
        record = pd.DataFrame(d["notes"])
        # change index of each note as start timedelta with seconds as unit
        record.index = pd.to_timedelta(record.start, "S")

        records.append(record)

    return records
