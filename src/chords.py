import queue

import numpy as np
import pandas as pd
import fortepyan as ff
from tqdm import tqdm
import chorder.dechorder
import chorder.timepoints
from datasets import Dataset, load_dataset
from miditoolkit.midi.containers import Note
from pretty_midi.pretty_midi import PrettyMIDI

from helpers import plot, save_table


def start_to_ticks(row, piece: PrettyMIDI):
    return piece.time_to_tick(row["start"])


def end_to_ticks(row, piece: PrettyMIDI):
    return piece.time_to_tick(row["end"])


def end_to_time(row, piece: PrettyMIDI):
    return piece.tick_to_time(row["end"])


def get_chords(piece: ff.MidiPiece):
    """
    Extracts chords from a MidiPiece object.
    Treats every group of simultaneously playing notes,
    which can be interpreted as a chord by a dechorder, as a chord.

    Parameters:
        piece (MidiPiece): A MidiPiece object containing musical events.

    Returns:
        list: A list of detected chords in the format [chord, start, end].
    """
    # define comparator
    setattr(Note, "__lt__", lambda self, other: self.end <= other.end)
    pq = queue.PriorityQueue()
    start = 0
    count = 0
    chords = []

    prev_chord = None
    midi_data = piece.df.copy()
    piece = piece.to_midi()
    threshold = 24
    midi_data["start"] = midi_data.apply(start_to_ticks, axis=1, args=(piece,))
    midi_data["end"] = midi_data.apply(end_to_ticks, axis=1, args=(piece,))
    notes_midi = np.array(midi_data[["velocity", "pitch", "start", "end"]])

    for note in notes_midi:
        note = Note(*note)
        pq.put(note)
        end = note.end
        # Pop notes which ended before current note started
        while pq.queue[0].end < note.start or (pq.queue[0].end - note.start > threshold and pq.queue[0] != note):
            pq.get()
            start = pq.queue[0].start
        notes = [note for note in pq.queue]
        # Try to recognize chord from list of notes
        chord, _ = chorder.Dechorder.get_chord_quality(notes, start, end)
        if chord.is_complete():
            # Check if the chord has changed
            if prev_chord != chord:
                prev_chord = chord
                chords.append([chord.__str__(), start, end])
                count += 1
    return chords


def plot_chords_vs_time(chords: list, prettymidi_piece: PrettyMIDI, title: str, timeframe=None):
    """
    Plots the number of chords played over time.

    Parameters:
        chords (list): A list of detected chords in the format [chord_name, start_time, end_time].
        prettymidi_piece (PrettyMIDI): A PrettyMIDI object representing the MIDI data.
        title (str): The title of the plot.
        timeframe (float, optional): The time interval (in seconds) for grouping chords. If not provided,
                                   it will be automatically calculated based on the recording duration.
    """
    chords = pd.DataFrame(chords, columns=["chord", "start", "end"])
    chords["end"] = chords.apply(end_to_time, axis=1, args=(prettymidi_piece,))

    recording_duration_seconds = chords["end"].max()
    # Determine the timeframe for grouping chords based on the recording duration
    if timeframe is None:
        if recording_duration_seconds > 120:
            # Calculate the number of notes played per minute
            timeframe = 60
            x_label = "Minutes"
        else:
            timeframe = 1
            x_label = "Seconds"
    else:
        x_label = "{:d} seconds".format(timeframe)
    # Group chords based on the specified timeframe and count the number of chords in each group
    chords["timeframe"] = chords["end"] // timeframe
    chords_count = chords.groupby(["timeframe"]).size()
    plot(
        x=chords_count.index, y=chords_count.values, xlabel=x_label, ylabel="Chords per {:s}".format(x_label.lower()), title=title
    )


def get_chord_count(record: dict) -> pd.Series:
    """

    Counts how many times each chord was played in a recording

    Parameters:
        - record (dict): A Hugging Face MidiPiece object representing the MIDI recording.

    Returns:
        - pandas Series: A Series containing the chord counts, with chord names as the index and their respective counts as values.

    """
    piece = ff.MidiPiece.from_huggingface(record)
    chords = get_chords(piece)
    chords_count = pd.DataFrame(chords, columns=["chord", "start", "end"])
    chords_count = chords_count.groupby(["chord"]).size()
    return chords_count


def get_most_played_chords(dataset: Dataset) -> pd.DataFrame:
    """
    Processes a dataset of MIDI records, detects the most played chord for each record,
    and saves the results to a CSV file.

    Parameters:
        dataset (list): A list of MIDI records to process.

    Returns:
        pd.Series: A row containing information about the most played chord for a specific composer and title.
                   The row contains the following columns: 'composer', 'title', 'chord', and 'repetitions'.
    """
    most_played_chords = []
    for record in tqdm(dataset):
        piece = ff.MidiPiece.from_huggingface(record)
        # print(record["composer"] + ", " + record["title"])
        chords = get_chords(piece)
        chords_count = pd.DataFrame(chords, columns=["chord", "start", "end"])
        chords_count = chords_count.groupby(["chord"]).size()
        max_index = np.argmax(chords_count.values[:])
        most_played_chords.append(
            [record["composer"], record["title"], chords_count.index[max_index], chords_count.values[max_index]]
        )

    most_played_chords = pd.DataFrame(most_played_chords, columns=["composer", "title", "chord", "repetitions"])
    return most_played_chords


def make_plots(dataset, index_to_plot):
    for index in index_to_plot:
        record = dataset[index]
        piece = ff.MidiPiece.from_huggingface(record)
        chords = get_chords(piece)
        plot_chords_vs_time(chords, piece.to_midi(), record["composer"] + ", " + record["title"])


def main():
    dataset = load_dataset("roszcz/maestro-v1", split="train+test+validation")
    index_to_plot = [137, 289, 347, 0, 87]
    # make_plots(dataset, index_to_plot)
    for index in index_to_plot:
        record = dataset[index]
        path = record["composer"] + "-" + record["title"]
        path = path.replace(" ", "-").lower()
        chord_count = get_chord_count(record=record)

        chord_count = np.array([chord_count.index.tolist(), chord_count.values.tolist()]).T
        chord_count = pd.DataFrame(chord_count, columns=["chord", "repetitions"])
        chord_count.to_csv(path + ".csv")
        save_table(chord_count, path + ".pdf")

    chord_reps = get_most_played_chords(dataset)
    max_index = chord_reps["repetitions"].idxmax()
    print(chord_reps.iloc[max_index])
    chord_reps.to_csv("most-played-chords-v2.csv")
    # composer                Franz Schubert
    # title          Sonata in D Major, D850
    # chord                               DM
    # repetitions                       1582
    # index: 347


if __name__ == "__main__":
    main()
