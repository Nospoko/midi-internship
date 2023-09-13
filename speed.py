import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset


def plot_speed_time(df: pd.DataFrame) -> plt.Figure:
    duration = df["end"].max() - df["start"][0]

    if duration > 120:
        time_unit = "minutes"
    else:
        time_unit = "seconds"

    bins = []
    for index, row in df.iterrows():
        if time_unit == "minutes":
            bins.append(round(row["end"] / 60))
        else:
            bins.append(round(row["end"]))

    df["bin"] = bins

    notes_per_minutes = {}

    for index, row in df.iterrows():
        if row["bin"] in notes_per_minutes:
            notes_per_minutes[row["bin"]] += 1
        else:
            notes_per_minutes[row["bin"]] = 1

    x = list(notes_per_minutes.keys())
    y = list(notes_per_minutes.values())

    fig, ax = plt.subplots()
    ax.plot(x, y)
    # ax.scatter(x, y)
    ax.set(xlabel="Time (" + time_unit + ")", ylabel="Number of notes", title="Number of notes over time")  # show the plot

    return fig


if __name__ == "__main__":
    dataset = load_dataset("roszcz/internship-midi-data-science", split="train")
    for record in dataset:
        df = pd.DataFrame(record["notes"])
        fig = plot_speed_time(df)
        plt.show()
