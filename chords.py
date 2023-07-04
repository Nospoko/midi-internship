import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import speed


def get_chords(df, threshold=0.05):
    pressed_simultaneously_df = speed.get_notes_idxs_pressed_simultaneously(
        df=df, threshold=threshold
    )

    records = []
    for it, row in pressed_simultaneously_df.iterrows():
        idxs = row.pressed_simultaneously
        ends = df.iloc[idxs].end.values
        where_close_to_min = np.where(ends - min(ends) < threshold)[0]
        close_end_idxs = np.array(idxs)[where_close_to_min]
        if len(close_end_idxs) > 1:
            notes_idxs = list(close_end_idxs)
            record = {
                "notes_idxs": notes_idxs,
                "time": df.iloc[notes_idxs].start.values.mean(),
                "pitches": list(df.iloc[notes_idxs].pitch.values),
            }
            records.append(record)

    chords_df = pd.DataFrame(records)
    return chords_df


def plot_chart_of_time_vs_number_of_chords_played(chords_df):
    times = chords_df["time"]
    step = 60
    bins = np.arange(times.min().astype(int), times.max().astype(int), step)
    chart_y = chords_df.groupby(pd.cut(times, bins)).time.count().values

    f, ax = plt.subplots(1, 1, figsize=[16, 6])
    ax.plot(bins[:len(chart_y)]/60, chart_y)
    ax.set_title("Number of chords played vs Time")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Number of chords played per minute")
    ax.grid(alpha=0.5, color="r", linestyle="--", linewidth=0.25)
    f.tight_layout()

    return f
