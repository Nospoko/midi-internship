import math

import pandas as pd
import music21 as m21
import matplotlib.pyplot as plt

from utils import load_records


def note_pitches_played_simultaneously(record: pd.DataFrame, window: float) -> pd.Series:
    """
    Function that resamples record and counts occurences of notes in each window

    Args:
        record (pd.DataFrame): record to resample
        window (float): window [seconds] used to resample record

    Returns:
        pd.Series: series with number of notes counted in each window
    """

    # resampling window, counting occurrences in each
    notes_in_each_window = record.resample(f"{window}S", origin="start_day")["pitch"].agg(list)
    # filtering series so that there are only windows with more than one note (chords)
    chords = notes_in_each_window[notes_in_each_window.map(lambda d: len(d) > 1)]

    return chords


def plot_time_vs_chords_per_minute(records: list[pd.DataFrame], threshold: float):
    """
    Plot time vs chords per minute

    Args:
        records (List[pd.DataFrame]): list of records
        threshold (float): threshold for determining whether the note is played simultaneously with another
    """
    # creating subplots for each record
    fig, axes = plt.subplots(nrows=3, ncols=math.ceil(len(records) / 3), figsize=(12, 12))
    # flattening axes to get easier access to them
    axes_flat = axes.flatten()

    for i, record in enumerate(records):
        # determining unit (minutes or seconds) based on length of record
        unit = 60.0 if record.end.max() > 120 else 1.0
        minutes = abs(unit - 1) > 1e-6

        # get simultaneously played notes (chords)
        chords = note_pitches_played_simultaneously(record, threshold)

        # resampling
        chords_per_minute = chords.resample("1 min").count()

        # casting timedelta to minutes or seconds depending on the unit used
        time = chords_per_minute.index.astype("timedelta64[m]") if minutes else chords_per_minute.index.astype("timedelta64[s]")

        # plotting time vs speed
        axes_flat[i].plot(time, chords_per_minute)

        # adding labels
        unit_label = "min" if minutes else "s"
        axes_flat[i].set_xlabel(f"Time [{unit_label}]")
        axes_flat[i].set_ylabel("Chords per minute")
        axes_flat[i].set_title(f"Record {i}")

    fig.tight_layout()
    plt.show()


def most_common_chords(records: list[pd.DataFrame], save_path: str, filter_octaves: bool = False):
    """
    Creates table counting occurences of each chord in the record

    Args:
        records (List[pd.DataFrame]): list of records
        filter_octaves (bool, optional): if True octaves are filtered out. Defaults to False.
    """
    for i, record in enumerate(records):
        # get simultaneously played notes (chords)
        chords = note_pitches_played_simultaneously(record, 0.1)

        # using music21 chord object to get chord names (it can run pretty long if there are a lot of chords)
        chord_names = chords.map(lambda x: m21.chord.Chord(x).pitchedCommonName)

        if filter_octaves:
            chord_names = chord_names[~chord_names.str.contains("octave", case=False)]

        chord_occurences = chord_names.value_counts()

        chord_occurences.to_csv(f"{save_path}/record_{i}.csv")


if __name__ == "__main__":
    records = load_records("roszcz/internship-midi-data-science")

    # chords 1
    plot_time_vs_chords_per_minute(records, threshold=0.1)
    # chords 2
    most_common_chords(records, save_path="../tables", filter_octaves=True)
