import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def plot_record_speed(record: pd.DataFrame):
    """
    Plots the number of notes played in a record
    """
    record_duration = int(np.ceil(record["end"].max()))
    unit = "minutes" if record_duration > 120 else "seconds"

    if record_duration > 120:
        # notes per seconds on a scale of minutes
        notes_per_sec = record.groupby(["minute"])["minute"].count() / 60
    else:
        # notes per sonds on a seconds scale of seconds
        notes_per_sec = record.groupby(["second"])["second"].count().to_dict()
        # it may happen that are no notes at the beggining
        notes_per_sec = {t: 0 if t not in notes_per_sec else notes_per_sec[t] for t in range(record_duration)}
        notes_per_sec = pd.Series(notes_per_sec)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(notes_per_sec)
    ax.set_title("Number of notes per second"), ax.set_xlabel(f"Time [{unit}]"), ax.set_ylabel("Notes count")
    ax.set_ylim([0, notes_per_sec.max()])
    fig.tight_layout()


def plot_aggregated_notes(ax, record_grouped, title):
    """
    Plotting for aggregated values of counts
    """
    ax.plot(record_grouped["minute"], record_grouped["notes_mean"], "k")
    ax.fill_between(
        record_grouped["minute"], y1=record_grouped["lower_bound"], y2=record_grouped["upper_bound"], color="blue", alpha=0.3
    )
    ax.set_title(title)
    return ax
