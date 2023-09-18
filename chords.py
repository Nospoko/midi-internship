from collections import Counter

import pandas as pd
import fortepyan as ff
from chorder import Dechorder
from datasets import load_dataset
from miditoolkit.midi.containers import Note


def main():
    dataset = load_dataset("roszcz/internship-midi-data-science", split="train")
    record = dataset[0]
    piece = ff.MidiPiece.from_huggingface(record)

    df = create_chords_table(piece)
    df.to_csv(f"{piece.source['record_id']}.csv")


def create_chords_table(piece):
    """
    Generate a table of chord counts for a given piece.
    """
    chords = get_chords(piece)
    counter = Counter(chords)

    chords_count = {item: count for item, count in counter.items()}
    chords_count = pd.Series(chords_count)

    df = chords_count.reset_index()
    df.columns = ["Note", "Count"]
    return df


def get_simultaneous_notes(piece: ff.MidiPiece):
    """
    Find simultaneous notes within a MIDI piece.

    Parameters:
    - piece: ff.MidiPiece
        The MIDI piece from which simultaneous notes are to be extracted.
    """
    # Set a tolerance threshold (e.g., 0.1 seconds) for considering notes as simultaneous
    tolerance = 0.1

    # Initialize a list to store groups of simultaneous notes
    simultaneous_notes = []

    # Iterate through each row in the DataFrame
    df = piece.df
    it = 0
    while it < len(df):
        row = df.iloc[it]
        current_start_time = row["start"]

        # Filter notes within the tolerance window
        simultaneous_group = df[(df["start"] >= current_start_time - tolerance) & (df["start"] <= current_start_time + tolerance)]
        notes = []

        for _, row in simultaneous_group.iterrows():
            note = Note(velocity=row["velocity"], pitch=int(row["pitch"]), start=int(row["start"]) - 1, end=int(row["end"]) + 1)
            notes.append(note)

        # Add the group to the result if it contains more than one note
        if len(simultaneous_group) > 1:
            simultaneous_notes.append(notes)

        it += len(simultaneous_group)

    return simultaneous_notes


def get_chords(piece: ff.MidiPiece):
    """
    Get the chords for simultaneous notes in a MIDI piece.

    Parameters:
    - piece: ff.MidiPiece
        The MIDI piece from which chord qualities are to be extracted.
    """
    simultaneous_notes = get_simultaneous_notes(piece)
    chords = []
    for group in simultaneous_notes:
        chord = Dechorder.get_chord_quality(notes=group, start=group[0].start, end=group[-1].end)
        if chord is not None:
            chords.append(str(chord[0]))

    return chords


if __name__ == "__main__":
    main()
