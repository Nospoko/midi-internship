import numpy as np
import pandas as pd
import fortepyan as ff
import matplotlib.pyplot as plt
from datasets import load_dataset


def lcs_length(x, y) -> int:
    """
    Returns the length of the longest common subsequence
    between two lists.
    """
    m, n = len(x), len(y)
    L = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                L[i, j] = L[i - 1, j - 1] + 1
            else:
                L[i, j] = max(L[i - 1, j], L[i, j - 1])

    return L[m, n]


def show_piano_roll_at_index(record, index, length=20, title="Piano Roll"):
    piece = ff.MidiPiece.from_huggingface(record)
    ff.view.draw_pianoroll_with_velocities(piece[index : index + length], title=title)
    plt.show()


def preprocess_record(record: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a record by removing unnecessary columns
    and sorting by start time.
    """
    record["abs_pitch"] = record["pitch"] % 12

    return record


def find_lcs_similarity(preprocessed_record: pd.DataFrame, sequence: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with the start index and similarity of the longest common subsequence
    between the subsequences of the record and the sequence.
    """
    results = {
        "start_index": [],
        "similarity": [],
    }

    for i in range(len(preprocessed_record) - len(sequence)):
        results["start_index"].append(i)
        results["similarity"].append(
            lcs_length(preprocessed_record["abs_pitch"][i : i + len(sequence)].tolist(), sequence["abs_pitch"].tolist())
        )

    # convert similarity to percentage
    results["similarity"] = np.array(results["similarity"]) / len(sequence)

    # sort by similarity
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["similarity"], ascending=False)

    return results_df


def cosine_similarity(df1, df2):
    """
    Returns the cosine similarity between two dataframes.
    """
    vector1 = df1.to_numpy().flatten()
    vector2 = df2.to_numpy().flatten()

    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def find_cosine_similarity(record, sequence):
    """
    Returns a dataframe with the start index and similarity of the cosine similarity
    between the subsequences of the record and the sequence.
    """
    results = {
        "start_index": [],
        "similarity": [],
    }

    for i in range(len(record) - len(sequence)):
        results["start_index"].append(i)
        results["similarity"].append(cosine_similarity(record.iloc[i : i + len(sequence)], sequence))

    # sort by similarity
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["similarity"], ascending=False)

    return results_df


if __name__ == "__main__":
    # load dataset
    dataset = load_dataset("roszcz/internship-midi-data-science", split="train")
    record = dataset[0]

    preprocessed_record = preprocess_record(pd.DataFrame(record["notes"]))

    # set random number seed for reproducibility
    np.random.seed(12345)
    # choose length for the sequence
    sequence_length = 20

    # pick a random sequence for testing
    start_index = np.random.randint(0, len(preprocessed_record) - sequence_length)
    sequence = preprocessed_record.iloc[start_index : start_index + sequence_length]

    # LCS similarity:
    lcs_similarity = find_lcs_similarity(preprocessed_record, sequence)

    # pick a random index from 5 most similar ones
    rand_index = np.random.choice(lcs_similarity.iloc[1:6].index)

    # show the piano roll of the sequence and the record
    show_piano_roll_at_index(
        record, start_index, length=sequence_length, title="Test Sequence, starting at index {}".format(start_index)
    )
    show_piano_roll_at_index(
        record, rand_index, length=sequence_length, title="Record subsequence, starting at index {}".format(rand_index)
    )

    # Cosine similarity:
    cosine_sim = find_cosine_similarity(preprocessed_record, sequence)

    # pick a random index from 5 most similar ones
    rand_index_cos = np.random.choice(cosine_sim.iloc[1:6].index)

    show_piano_roll_at_index(
        record, rand_index_cos, length=sequence_length, title="Record subsequence, starting at index {}".format(rand_index_cos)
    )

    # most dissimilar:
    rand_index_diss = cosine_sim.iloc[-1].name
    rand_index_diss_lcs = lcs_similarity.iloc[-1].name

    show_piano_roll_at_index(
        record, rand_index_diss, length=sequence_length, title="Most dissimilar cosine at index {}".format(rand_index_diss)
    )
    show_piano_roll_at_index(
        record, rand_index_diss_lcs, length=sequence_length, title="Most dissimilar LCS at index {}".format(rand_index_diss_lcs)
    )
