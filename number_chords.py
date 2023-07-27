from speed_time import *

def count_chords(notes_df, threshold):
    """A chart of time vs. the number of chords played.


    Args:
        notes_df: A Pandas DataFrame containing the notes from a MIDI file.
        threshold: The minimum number of notes required to form a chord.

    Returns:
        A tuple of two Pandas DataFrames: the first DataFrame contains the time,
        number of chords, and chord notes for each chord, and the second DataFrame
        contains the number of occurrences of each unique chord.
    """

    chords = []
    chords_occurrences = {}
    notes_df = notes_df.sort_values(by="start")

    for _, row in notes_df.iterrows():
        simultaneous_notes = notes_df[
            (notes_df["start"] >= row["start"]) & (notes_df["end"] <= row["end"])
        ]
        if len(simultaneous_notes) >= threshold:
            chord_notes = tuple(simultaneous_notes["pitch"].tolist())
            chords.append((row["start"], len(simultaneous_notes), chord_notes))

            if chord_notes in chords_occurrences:
                chords_occurrences[chord_notes] += 1
            else:
                chords_occurrences[chord_notes] = 1

    chords_df = pd.DataFrame(chords, columns=["time", "num_chords", "chord_notes"])
    return chords_df, chords_occurrences


def main():
    """Loads the dataset, counts the number of chords, and plots the results."""
    from datasets import load_dataset

    dataset = load_dataset("roszcz/internship-midi-data-science", split="train")

    # Get the notes column from the first record
    record = dataset[0]
    notes_df = pd.DataFrame(record["notes"])

    # Set the threshold for chord detection (experiment with different values)
    threshold = 3

    # Count the number of chords and detect chords
    chords_df, chords_occurrences = count_chords(notes_df, threshold)

    # Plot the time vs. number of chords chart
    plt.plot(chords_df["time"], chords_df["num_chords"], marker="o", linestyle="-")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of Chords played")
    plt.title(f"Time vs. Number of Chords (Threshold: {threshold})")
    plt.grid(True)
    plt.show()

    #A table with the number of occurrences of each unique chord
    chords_table = pd.DataFrame(list(chords_occurrences.items()), columns=["Chord", "Occurrences"])
    chords_table = chords_table.sort_values(by="Occurrences", ascending=False)
    print(chords_table)


if __name__ == "__main__":
    main()
