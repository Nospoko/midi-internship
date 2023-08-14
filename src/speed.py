import numpy as np
import pandas as pd
import fortepyan as ff
from tqdm import tqdm
from datasets import load_dataset

from helpers import plot


def calculate_speed(midi_data: pd.DataFrame, timeframe=None) -> (pd.Series, str):
    """
    Calculate the number of notes played per second or per minute from MIDI data.

    Parameters:
        midi_data (pd.DataFrame): A DataFrame containing MIDI data with 'start' and 'end' columns.
        timeframe (int, optional): The timeframe in seconds. If None, the function will choose between
                                   seconds and minutes automatically based on the recording duration.

    Returns:
        pd.Series, str: A Series containing the number of notes played in each timeframe,
                        and a string representing the xlabel for the plot.

    """
    recording_duration_seconds = midi_data["end"].max()
    if timeframe is not None:
        midi_data["timeframe"] = midi_data["start"] // timeframe
        note_counts = midi_data.groupby("timeframe").size()
        x_label = "{:d} second(s)".format(timeframe)
        return note_counts, x_label

    if recording_duration_seconds > 120:
        midi_data["timeframe"] = midi_data["start"] // 60
        note_counts = midi_data.groupby("timeframe").size()
        x_label = "Minutes"
    else:
        midi_data["timeframe"] = np.floor(midi_data["start"])
        note_counts = midi_data.groupby("timeframe").size()
        x_label = "Seconds"

    return note_counts, x_label


def plot_speed(midi_data, title="speed", timeframe=None):
    """
    Plot the number of notes played per second or per minute from MIDI data.

    """
    note_counts, x_label = calculate_speed(midi_data, timeframe)
    plot(x=note_counts.index, y=note_counts.values, xlabel=x_label, ylabel="Number of notes", title=title)


def plot_simultaneous_notes(midi_data: pd.DataFrame, time_threshold, title="simultaneous-notes"):
    """
    Plot the number of notes played per second or per minute from MIDI data.

    """
    # Round the 'start' column to the nearest threshold to group the notes
    midi_data["Start Rounded"] = np.round(midi_data["start"] / time_threshold) * time_threshold
    note_counts = midi_data.groupby("Start Rounded").size()
    x_label = "Second"
    plot(
        x=note_counts.index,
        y=note_counts.values,
        xlabel=x_label,
        ylabel="Number of Notes Pressed Simultaneously",
        title=str(time_threshold) + " " + title,
    )


def find_fastest(dataset):
    """
    Find the fastest 15-second continuous timeframe with the most notes pressed per second.

    Parameters:
        dataset (list): A list of MIDI data records.

    Returns:
        tuple: A tuple containing the composer, title, and number of notes pressed for the fastest timeframe.

    Example:
        # Assuming 'dataset' is a list of MIDI data records
        fastest_record = find_fastest(dataset)
        print(fastest_record)  # Output: ('Composer', 'Title', 100)
    """
    fastest = []
    window_size = 15
    for record in tqdm(dataset):
        piece = ff.MidiPiece.from_huggingface(record)
        # print(record["composer"] + ', ' + record["title"])
        midi_data = piece.df
        note_counts, _ = calculate_speed(midi_data, timeframe=1)
        full_time_range = list(range(np.max(note_counts.index.astype(int))))
        note_counts = pd.DataFrame(note_counts, columns=["count"])
        note_counts = note_counts.reindex(full_time_range, fill_value=0)
        start_index = note_counts.rolling(window=window_size).sum().idxmax()
        number_of_notes = note_counts.iloc[int(start_index) : int(start_index + window_size - 1)].sum()
        fastest.append([record["composer"], record["title"], number_of_notes.values[0]])
    pd.DataFrame(fastest, columns=["composer", "title", "count"]).to_csv("speed.csv")

    return max(fastest, key=lambda row: row[2])


if __name__ == "__main__":
    dataset = load_dataset("roszcz/maestro-v1", split="train+test+validation")

    record = dataset[3]
    midi_data = pd.DataFrame(record["notes"])
    plot_speed(midi_data, title=record["composer"] + " " + record["title"])
    time_thresholds = [0.01, 0.03, 0.05, 0.08, 0.1]
    for threshold in time_thresholds:
        plot_simultaneous_notes(midi_data, time_threshold=threshold, title=record["composer"] + " " + record["title"])
    print(find_fastest(dataset))
    # test: ['Franz Liszt', 'Transcendental Etude No. 11 "Harmonies du Soir"', 416]
    # train: ['Frédéric Chopin', 'Etude Op. 25 No. 10 in B Minor', 483]
