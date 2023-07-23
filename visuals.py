import pandas as pd

from speed import record_length, notes_per_interval


def plot_speed_over_time(record: pd.DataFrame, name: str):
    length = record_length(record)

    if length > 120:
        unit = 60
        unit_label = "[minutes]"
    else:
        unit = 1
        unit_label = "[seconds]"

    nps = notes_per_interval(record, duration=unit)
    nps["time"] = nps.index.categories.right / unit

    plot = nps.plot(x="time", y="count", legend=False)
    plot.set_xlabel(f"Time {unit_label}")
    plot.set_ylabel("Speed [notes per second]")
    plot.set_title("Speed over time for " + name)

    return plot
