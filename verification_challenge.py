import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

import speed
import chords

dataset = load_dataset("roszcz/maestro-v1", split="train")
record = dataset[0]
df = pd.DataFrame(record["notes"])

"""
A piece with the fastest 15 seconds of music
"""


def get_n_notes_played_in_consecutive_n_seconds(df, n_seconds=15):
    start = df.start.values

    sta, end = start.min(), start.min() + n_seconds
    n_notes_played = []
    time_range = []
    while end < start.max():
        time_condition = (start <= end) & (start >= sta)
        n_notes_played.append(np.sum(time_condition))
        time_range.append((sta, end))

        sta += 1
        end += 1

    results = {"time_range": time_range, "n_notes_played": n_notes_played}
    df_for_n_sec = pd.DataFrame(results)
    return df_for_n_sec


def get_fastest_n_seconds(df_for_n_sec):
    idx_fastest = np.argmax(df_for_n_sec.n_notes_played)
    fastest_n_sec = df_for_n_sec.iloc[idx_fastest, :]
    return fastest_n_sec


"""
A piece where a single chord is repeated the most (each piece will have a different chord)
"""


def get_row_indices_correspodning_to_chords(chords_df, repeated_chords_df):
    records = []
    for jt in range(len(repeated_chords_df)):
        considered_chord = repeated_chords_df["repeated_chords"].reset_index(drop=True)[
            jt
        ]
        row_idxs = [
            it
            for it, chord in enumerate(chords_df["sorted_pitches"].values)
            if chord == considered_chord
        ]
        record = {"chord": considered_chord, "row_idxs": row_idxs}
        records.append(record)
        df_chords_rows = pd.DataFrame(records)
    return df_chords_rows


def cluster_times_when_chords_are_repeated(chords_times, max_time_difference):
    clusters = []
    cluster = []
    for it in range(0, len(chords_times) - 1):
        condition = chords_times[it + 1] - chords_times[it] < max_time_difference
        if condition == False:
            if len(cluster) > 0:
                clusters.append(np.unique(cluster))
            cluster = []
        if condition == True:
            cluster.append(chords_times[it])
            cluster.append(chords_times[it + 1])
    if condition == True:
        clusters.append(np.unique(cluster))
    return clusters


def get_most_repeated_chords_in_time(chords_df, max_time_difference=5):
    repeated_chords_df = chords.get_table_with_occurences_of_chords(chords_df=chords_df)
    df_chords_rows = get_row_indices_correspodning_to_chords(
        chords_df=chords_df, repeated_chords_df=repeated_chords_df
    )

    n_repetitions = df_chords_rows.row_idxs.apply(lambda x: len(x)).values
    records = []
    for it in range(np.sum(n_repetitions > 1)):
        idxs = df_chords_rows["row_idxs"][it]
        chords_times = chords_df.iloc[idxs].time.values
        clusters = cluster_times_when_chords_are_repeated(
            chords_times=chords_times, max_time_difference=max_time_difference
        )
        clusters = [cluster for cluster in clusters if cluster != []]
        if len(clusters) > 0:
            record = {
                "chord": df_chords_rows["chord"][it],
                "clustered_times": clusters,
                "n_repetitions_of _chords": n_repetitions[it],
            }
            records.append(record)
    df_most_repeated_chords = pd.DataFrame(records)
    return df_most_repeated_chords


def get_separate_clusters_of_chords(df_most_repeated_chords):
    records = []
    for it, row in df_most_repeated_chords.iterrows():
        clusters = row.clustered_times
        for cluster in clusters:
            mean_time = np.mean(cluster)
            n_repetitions = len(cluster)

            record = {
                "n_repetitions": n_repetitions,
                "chords_time": mean_time,
                "chords": row.chord,
            }
            records.append(record)
    chord_clusters_in_time = pd.DataFrame(records)
    return chord_clusters_in_time


def plot_chord_clusters_in_time(df_most_repeated_chords):
    chord_clusters_in_time = get_separate_clusters_of_chords(
        df_most_repeated_chords=df_most_repeated_chords
    )
    chord_clusters_in_time = chord_clusters_in_time.sort_values(by="chords_time")
    x = chord_clusters_in_time.chords_time.values
    y = chord_clusters_in_time.n_repetitions.values

    f, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.scatter(x, y)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("n_repetitions of a chord")

    return f
