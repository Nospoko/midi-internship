import queue

import numpy as np
import pandas as pd
import fortepyan as ff
from tqdm import tqdm
import chorder.dechorder
import chorder.timepoints
from datasets import load_dataset
from miditoolkit.midi.containers import Note
from pretty_midi.pretty_midi import PrettyMIDI

from helpers import plot


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
        list: A list of detected chords in the format [chord_name, start_time, end_time].
    """
    # mido_obj = midipiece_to_midifile(piece)
    # define comparator
    setattr(Note, "__lt__", lambda self, other: self.end <= other.end)
    pq = queue.PriorityQueue()
    start = 0
    count = 0
    chords = []
    prev_chord = None
    midi_data = piece.df
    piece = piece.to_midi()
    midi_data["start"] = midi_data.apply(start_to_ticks, axis=1, args=(piece,))
    midi_data["end"] = midi_data.apply(end_to_ticks, axis=1, args=(piece,))
    notes_midi = np.array(midi_data[["velocity", "pitch", "start", "end"]])

    for note in notes_midi:
        note = Note(*note)
        pq.put(note)
        end = note.end
        notes = [note for note in pq.queue]
        # Pop notes which ended before current note started
        if pq.empty():
            continue
        while note.start > pq.queue[0].end:
            pq.get()
            start = pq.queue[0].start
        # Try to recognize chord from list of notes
        chord = chorder.Dechorder.get_chord_quality(notes, start, end)
        if chord[0].is_complete():
            # Check if the chord has changed
            if prev_chord != chord[0]:
                prev_chord = chord[0]
                chords.append([chord[0].__str__(), start, end])
                count += 1
    return chords


def plot_chords_vs_time(chords, prettyMIDIpiece, title, timeframe=None):
    """
    Plots the number of chords played over time.

    Parameters:
        chords (list): A list of detected chords in the format [chord_name, start_time, end_time].
        prettyMIDIpiece (PrettyMIDI): A PrettyMIDI object representing the MIDI data.
        title (str): The title of the plot.
        timeframe (int, optional): The time interval (in seconds) for grouping chords. If not provided,
                                   it will be automatically calculated based on the recording duration.

    Returns:
        None
    """
    chords = pd.DataFrame(chords, columns=["chord", "start", "end"])
    chords["end"] = chords.apply(end_to_time, axis=1, args=(prettyMIDIpiece,))

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


def process_dataset(dataset, csv_file_path):
    """
    Processes a dataset of MIDI records, detects the most played chord for each record,
    and saves the results to a CSV file.

    Parameters:
        dataset (list): A list of MIDI records to process.
        csv_file_path (str): The file path to save the results as a CSV file.

    Returns:
        pd.Series: A row containing information about the most played chord for a specific composer and title.
                   The row contains the following columns: 'composer', 'title', 'chord', and 'repetitions'.
    """
    most_played_chords = []
    for record in tqdm(dataset):
        piece = ff.MidiPiece.from_huggingface(record)
        print(record["composer"] + ", " + record["title"])
        chords = get_chords(piece)
        chords_count = pd.DataFrame(chords, columns=["chord", "start", "end"])
        chords_count = chords_count.groupby(["chord"]).size()
        max_index = np.argmax(chords_count.values[:])
        most_played_chords.append(
            [record["composer"], record["title"], chords_count.index[max_index], chords_count.values[max_index]]
        )

    most_played_chords = pd.DataFrame(most_played_chords, columns=["composer", "title", "chord", "repetitions"])
    most_played_chords.to_csv(csv_file_path)
    max_index = most_played_chords["repetitions"].idxmax()
    return most_played_chords.iloc[max_index]


def make_plots(dataset, index_to_plot):
    for index in index_to_plot:
        record = dataset[index]
        piece = ff.MidiPiece.from_huggingface(record)
        chords = get_chords(piece)
        plot_chords_vs_time(chords, piece.to_midi(), record["composer"] + ", " + record["title"])


if __name__ == "__main__":
    dataset = load_dataset("roszcz/maestro-v1", split="train+test+validation")
    # index_to_plot = [0, 16, 25, 50, 30, 55]
    # make_plots(dataset, index_to_plot)

    most_chord_reps = process_dataset(dataset, "most-played-chords-v3.csv")
    print(most_chord_reps)

    # all:
    # composer                Franz Schubert
    # title          Sonata in D Major, D850
    # chord                               AM
    # repetitions                        805
