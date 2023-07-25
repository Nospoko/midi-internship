from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils import load_records


def preprocess_record(record: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering some attributes than will help in identifying sequence

    Args:
        record (pd.DataFrame): single record with columns [start, end, pitch and velocity]

    Returns:
        pd.DataFrame: record with additional feature engineered columns [duration, step, absolute pitch]
    """

    # getting note duration
    record["duration"] = record["end"] - record["start"]

    # getting note step: distance between start of the note and start of previous note
    record["step"] = record["start"].diff(periods=1)
    # filling na for first value
    record["step"].fillna(0)

    # getting absolute note pitch
    record["absolute_pitch"] = record["pitch"] % 12

    return record


def extract_relevant_attributes(df: pd.DataFrame) -> pd.DataFrame:
    # extracting relevant attributes for sequence identification
    # start and end are absolute values in context of the record so instead of them duration and step are used
    return df[["pitch", "velocity", "duration", "step", "absolute_pitch"]]


def upper_triangle(df: pd.DataFrame) -> np.ndarray:
    """
    Extracting upper triangular values from correlation matrix

    Args:
        df (pd.DataFrame): correlation scores within sequence

    Returns:
        np.ndarray: upper triangular values
    """
    matrix = df.values
    mask = np.triu_indices(matrix.shape[0], k=1)
    return matrix[mask]


def spearman_corr_score(sequence: pd.DataFrame, window: pd.DataFrame) -> float:
    """
    Calculating spearman correlation between correlations within sequence and window

    Args:
        sequence (pd.DataFrame): input sequence
        window (pd.DataFrame): subset of rolling window

    Returns:
        float: spearman correlation score
    """

    # calculating spearman correlation between attributes in input sequence
    seq_corr = sequence.corr(method="spearman")
    # calculating spearman correlation between attributes in window sequence
    window_corr = window.corr(method="spearman")

    # getting upper triangle values
    seq_upper = upper_triangle(seq_corr)
    window_upper = upper_triangle(window_corr)

    # calculating correlation between sequences
    similarity_score = spearmanr(seq_upper, window_upper).statistic

    return similarity_score


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
    return np.trace(cos_sim) / num_features


def similarity_scores(
    sequence: pd.DataFrame, record: pd.DataFrame, similarity: Callable[[pd.DataFrame, pd.DataFrame], float] = cos_sim_score
) -> pd.DataFrame:
    """
    Calculating similarity scores between

    Args:
        sequence (pd.DataFrame): input sequence [pitch, velocity, duration, step, absolute pitch]
        record (pd.DataFrame): single record with columns [pitch, velocity, duration, step, absolute pitch]
        similarity (Callable[[pd.DataFrame, pd.DataFrame], float], optional): similarity score function. Defaults to cos_sim_score.

    Returns:
        pd.DataFrame: dataframe with start_time of sequence and its score
    """

    # dict for storing start of sequences and similarity scores
    similarity_scores = {"start_time": [], "similarity_score": []}

    # rolling window of the same size as input sequence
    for subset in record.rolling(window=len(sequence)):
        # not sure why but first windows of rolling function have shorter lengths so filter them out here
        if len(subset) != len(sequence):
            continue

        # calculate similarity
        score = similarity(sequence, subset)

        # add time when sequence started and score to dict
        similarity_scores["start_time"].append(subset.index[0])
        similarity_scores["similarity_score"].append(score)

    # cast scores to dataframe
    return pd.DataFrame(similarity_scores)


def sort_by_scores(similarity: pd.DataFrame):
    # sotting by similarity scores
    return similarity.sort_values(by="similarity_score", ascending=False)


if __name__ == "__main__":
    record = load_records("roszcz/internship-midi-data-science")[0]

    processed_record = preprocess_record(record)
    processed_record = extract_relevant_attributes(processed_record)

    # get some random test sequence
    test_sequence = processed_record.iloc[1024 : 1024 + 32, :]

    similarity = similarity_scores(test_sequence, processed_record, similarity=cos_sim_score)

    sorted_similarity = sort_by_scores(similarity)

    sorted_similarity.to_csv("../tables/cos_sim_0.csv")
