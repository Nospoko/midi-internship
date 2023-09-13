from speed_time import *

def count_chords(notes_df, threshold):
  """Create a chart showing the number of notes pressed at the same time.
  Experiment with different thresholds.2, 4, 8, 12
  Args:
    notes_df: A Pandas DataFrame containing the notes from a MIDI file.
    threshold: The minimum number of simultaneous notes to be considered a chord.

  Returns:
    A Pandas DataFrame with the following columns:
      time: The time of the chord.
      num_simultaneous_notes: The number of simultaneous notes in the chord.
      chord_notes: The notes in the chord.
  """

  chords = []
  chords_occurrences = {}
  notes_df = notes_df.sort_values(by="start")

  for _, row in notes_df.iterrows():
    simultaneous_notes = notes_df[
        (notes_df["start"] >= row["start"]) & (notes_df["end"] <= row["end"])
    ]
    num_simultaneous_notes = len(simultaneous_notes)
    if num_simultaneous_notes >= threshold:
      chord_notes = tuple(simultaneous_notes["pitch"].tolist())
      chords.append((row["start"], num_simultaneous_notes, chord_notes))

      if chord_notes in chords_occurrences:
        chords_occurrences[chord_notes] += 1
      else:
        chords_occurrences[chord_notes] = 1

  chords_df = pd.DataFrame(chords, columns=["time", "num_simultaneous_notes", "chord_notes"])
  return chords_df


def main():
  # Load the dataset
  from datasets import load_dataset
  dataset = load_dataset("roszcz/internship-midi-data-science", split="train")

  # Get the notes column from the first record
  record = dataset[0]
  notes_df = pd.DataFrame(record["notes"])

  # Set different threshold values (experiment with different values)
  thresholds = [2, 4, 8, 12]

  # Plot the charts for different thresholds
  plt.figure(figsize=(13, 9))
  for idx, threshold in enumerate(thresholds):
    chords_df = count_chords(notes_df, threshold)

    plt.subplot(2, 2, idx + 1)
    plt.plot(chords_df["time"], chords_df["num_simultaneous_notes"], marker="o", linestyle="-")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of notes pressed at the same time")
    plt.title(f"Threshold: {threshold}")
    plt.grid(True)

  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  main()
