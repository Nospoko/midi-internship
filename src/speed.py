import math
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utils import load_records


def notes_in_window(record: pd.DataFrame, window: float) -> pd.Series:
    """
    Function that resamples record and counts occurences of notes in each window

    Args:
        record (pd.DataFrame): record to resample
        window (float): window [seconds] used to resample record

    Returns:
        pd.Series: series with number of notes counted in each window
    """

    # resampling window, counting occurrences in each
    notes_in_each_window = record.resample(f"{window}S", origin="start_day")["start"].count()

    return notes_in_each_window


def plot_time_vs_speed(records: list[pd.DataFrame], window: float = None):
    """
    Function which plots time [seconds or minutes] vs speed [notes per second]

    Args:
        records (List[pd.DataFrame]): list of records
        window (float): window in which notes will be counted and then normalized by that factor to get notes per second
    """

    # creating subplots for each record
    fig, axes = plt.subplots(nrows=3, ncols=math.ceil(len(records) / 3), figsize=(10, 10))
    # flattening axes to get easier access to them
    axes_flat = axes.flatten()

    for i, record in enumerate(records):
        # determining unit (minutes or seconds) based on length of record
        unit = 60.0 if record.end.max() > 120 else 1.0
        minutes = abs(unit - 1) > 1e-6
        # if window for counting notes is not specified it defaults to unit (minutes or seconds)
        window = unit if window is None else window

        # get notes counted in each window
        notes_in_each_window = notes_in_window(record, window)
        # normalization to get notes per second in each window
        notes_per_second = (notes_in_each_window / window).round()

        # casting timedelta to minutes or seconds depending on the unit used
        time = notes_per_second.index.astype("timedelta64[m]") if minutes else notes_per_second.index.astype("timedelta64[s]")

        # plotting time vs speed
        axes_flat[i].plot(time, notes_per_second)

        # adding labels
        unit_label = "min" if minutes else "s"
        axes_flat[i].set_xlabel(f"Time [{unit_label}]")
        axes_flat[i].set_ylabel("Notes per second")
        axes_flat[i].set_title(f"Record {i}")

    fig.tight_layout()
    plt.show()


def plot_notes_played_simultaneously(records: list[pd.DataFrame], threshold: float = 0.1):
    """
    Plot time [minutes or seconds] vs notes played simultaneously

    Args:
        records (List[pd.DataFrame]): list of records
        threshold (float): window to count notes that are played simultaneous
    """

    # creating subplots for each record
    fig, axes = plt.subplots(nrows=math.ceil(len(records) / 2), ncols=2, figsize=(20, 10))
    # flattening axes to get easier access to them
    axes_flat = axes.flatten()

    for i, record in enumerate(records):
        # determining unit (minutes or seconds) based on length of record
        unit = 60.0 if record.end.max() > 120 else 1.0
        minutes = abs(unit - 1) > 1e-6

        # get notes simultaneously played (tolerance is given by threshold)
        notes_in_each_window = notes_in_window(record, threshold)

        # casting timedelta to minutes or seconds depending on the unit used
        time = (
            notes_in_each_window.index.astype("timedelta64[m]")
            if minutes
            else notes_in_each_window.index.astype("timedelta64[s]")
        )

        # computing cross tabulation between timedelta and notes played simultaneously
        num_sim_notes_in_window = pd.crosstab(time, notes_in_each_window)

        # heatmap
        sns.heatmap(
            num_sim_notes_in_window.iloc[:, 2:].T, norm=LogNorm(), ax=axes_flat[i], cbar_kws={"label": "Num notes in window"}
        )

        # adding labels
        unit_label = "min" if minutes else "s"
        axes_flat[i].set_xlabel(f"Time [{unit_label}]")
        axes_flat[i].set_ylabel("Notes simultaneously played")
        axes_flat[i].set_title(f"Record {i}")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # records = load_records("roszcz/internship-midi-data-science")

    # plot_time_vs_speed(records)
    # plot_notes_played_simultaneously(records, threshold=0.1)

    parser = argparse.ArgumentParser(prog="Speed")

    parser.add_argument("--fp", type=str, help="Provide HuggingFace filepath")
    parser.add_argument(
        "--plot_number", type=int, choices=[1, 2], help="1. Plot time vs notes per second, 2. Notes played simulateneously"
    )
    parser.add_argument(
        "--window",
        type=float,
        default=60,
        help="Window for plot 1. Specifies in what window notes will be counted before being normalized to notes per second. Default 60 seconds.",
    )
    parser.add_argument(
        "--sim_note_threshold",
        type=float,
        default=0.1,
        help="Threshold window for plot 2. Specifies window where notes are counted as being played simultaneously. Default 0.1 second.",
    )

    args = parser.parse_args()

    records = load_records(args.fp)

    if args.plot not in [1, 2]:
        raise ValueError("No such plot. Type either 1 or 2!")
    elif args.plot == 1:
        plot_time_vs_speed(records, args.window)
    else:
        plot_notes_played_simultaneously(records, args.window)
