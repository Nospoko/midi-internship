import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from datasets import load_dataset

# dataset = load_dataset("roszcz/internship-midi-data-science", split="train")


"""
SPEED
"""


def get_time_vs_speed(record):
    df = pd.DataFrame(record["notes"])

    start = df.start.values
    end = df.end.values

    if (max(end) - min(start)) > 120:
        step = 60
    else:
        step = 1

    bins = range(df["start"].min().astype(int), df["start"].max().astype(int), step)
    notes_in_time = df.groupby(pd.cut(df["start"], bins))["pitch"].count().reset_index()
    notes_in_time.columns = ["time_range_in_sec", "number_of_notes_played"]

    return notes_in_time


def plot_chart_time_vs_speed(notes_in_time):
    interval = notes_in_time.time_range_in_sec[0]
    if interval.right - interval.left == 60:
        unit = 'min'
        step = 60
    else:
        unit = 'sec'
        step = 1
   
    f, ax = plt.subplots(1, 1, figsize=[14, 6])
    ax.plot(
        notes_in_time.index,
        notes_in_time["number_of_notes_played"].values / step,
    )
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("Notes played per second")
    ax.set_title("Time vs Speed")
    ax.grid(alpha=0.5, color="r", linestyle="--", linewidth=0.25)
    f.tight_layout()

    return f


"""
NOTES PRESSED AT THE SAME TIME
"""

# df = pd.DataFrame(record["notes"])


def get_notes_idxs_pressed_simultaneously(df, threshold=0.05):
    start = df.start.values

    records = []
    it = 0

    while it < len(df):
        t_0 = start[it]
        where_close = (start[it + 1: it + 11] - t_0) < threshold
        n_close = np.sum(where_close)

        if n_close > 0:
            idxs = np.arange(it, it + n_close + 1)
            it += n_close + 1

            record = {
                "pressed_simultaneously": list(idxs),
                "time": df.iloc[idxs].start.values.mean(),
                "pitches": list(df.iloc[idxs].pitch.values),
            }
            records.append(record)
        else:
            it += 1
    pressed_simultaneously_df = pd.DataFrame(records)
    return pressed_simultaneously_df


def plot_chart_notes_pressed_at_the_same_time(pressed_simultaneously_df):
    x_chart = pressed_simultaneously_df.time.values
    y_chart = [
        len(el) for el in pressed_simultaneously_df.pressed_simultaneously.values
    ]

    f1, ax = plt.subplots(1, 1, figsize=[16, 6])
    ax.scatter(x_chart, y_chart)
    ax.set_title("Number of notes pressed at the same time vs Time")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Number of notes")
    f1.tight_layout()

    f2, ax = plt.subplots(1, 1, figsize=[14, 6])
    ax.hist(y_chart, bins=np.arange(2, max(y_chart) + 1, 0.5) - 0.25)
    ax.set_title("Histogram: number of notes pressed at the same time")
    f2.tight_layout()

    return f1, f2
