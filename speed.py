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
        unit = "min"
    else:
        step = 1
        unit = "sec"

    notes_in_time = pd.DataFrame(
        df.groupby(
            pd.cut(
                df["start"], range(int(min(df["start"])), int(max(df["start"])), step)
            )
        ).pitch.count()
    )
    notes_in_time.index = np.arange(
        int(min(start)) + 1, len(notes_in_time) + int(min(start)) + 1
    )
    notes_in_time.index.name = f"{unit}_number"
    notes_in_time.rename(columns={"pitch": "number_of_notes_played"}, inplace=True)

    return notes_in_time


def plot_chart_time_vs_speed(notes_in_time):
    unit = notes_in_time.index.name.split("_")[0]
    step = 1 if unit == "sec" else 60

    f, ax = plt.subplots(1, 1, figsize=[14, 6])
    ax.scatter(
        x=notes_in_time.index,
        y=notes_in_time["number_of_notes_played"].values / step,
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


def get_notes_idxs_pressed_simultaneously(record):
    df = pd.DataFrame(record["notes"])
    start = df.start.values

    threshold = 0.05  # can be adjust/changed, eg. to 0.1
    pressed_simultaneously = []
    it = 0

    while it < len(df):
        t_0 = start[it]
        where_close = (start[it + 1 :] - t_0) < threshold
        n_close = np.sum(where_close)

        if n_close > 0:
            chord = np.arange(it, it + n_close + 1)
            it += n_close + 1
            pressed_simultaneously.append(list(chord))
        else:
            it += 1

    return pressed_simultaneously


def plot_chart_notes_pressed_at_the_same_time(df, pressed_simultaneously):
    x_time, y_notes = [], []

    for idxs in pressed_simultaneously:
        t = df.iloc[idxs].start.values.mean()
        n = len(idxs)
        x_time.append(t)
        y_notes.append(n)

    f1, ax = plt.subplots(1, 1, figsize=[16, 6])
    ax.scatter(x_time, y_notes)
    ax.set_title("Number of notes pressed at the same time vs Time")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Number of notes")
    f1.tight_layout()

    f2, ax = plt.subplots(1, 1, figsize=[14, 6])
    ax.hist(y_notes, bins=np.arange(2, max(y_notes) + 1, 0.5) - 0.25)
    ax.set_title("Histogram: number of notes pressed at the same time")
    f2.tight_layout()

    return f1, f2
