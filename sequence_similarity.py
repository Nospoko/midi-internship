import numpy as np
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("roszcz/internship-midi-data-science", split="train")

record = dataset[0]
df = pd.DataFrame(record["notes"])
print(df.head())


def cos_sim_score(sequence: pd.DataFrame, window: pd.DataFrame) -> float:
    """
    Calculating cosine similarity between sequence and window
    Args:
        sequence (pd.DataFrame): input sequence
        window (pd.DataFrame): subset of rolling window
    Returns:
        float: cosine similarity score
    """

    # extracting numpy array and transposing, shape: [num_features, window_size]
    sequence_arr = sequence.values.T
    # shape: [window_size, num_features]
    window_arr = window.values

    # det product shape: [features, features]
    sequence_x_window = sequence_arr @ window_arr

    # shape: [num_features, 1]
    sequence_norm = np.linalg.norm(sequence_arr, axis=1, keepdims=True)
    # shape: [1, num_features]
    window_norm = np.linalg.norm(window_arr, axis=0, keepdims=True)

    # shape: [num_features, num_features]
    normalization = sequence_norm * window_norm

    # calculating cosine similarity for each entry
    cos_sim = sequence_x_window / (normalization + 1e-8)

    num_features = cos_sim.shape[0]

    # returning normalized trace of cosine similarity because the values of interest are along main diagonal
    return np.sum(cos_sim) / (num_features * num_features)


x = df.iloc[0:16]
x = x[["pitch", "velocity"]]

scores = {"score": [], "idx": []}

seq_len = len(x)

for i in range(0, len(df) - seq_len):
    seq = df.iloc[i : i + seq_len]
    seq = seq[["pitch", "velocity"]]
    score = cos_sim_score(x, seq)
    scores["score"].append(score)
    scores["idx"].append(i)

similarity = pd.DataFrame(scores)
similarity.sort_values(by="score", ascending=False, inplace=True)

print(similarity)
