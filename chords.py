from speed_time import *
def count_chords(notes_df, threshold):
  """Create a chart showing the number of notes pressed at the same time.
   Experiment with different thresholds.

  Args:
    notes_df: A Pandas DataFrame containing the notes from a MIDI file.
    threshold: The minimum number of notes required to form a chord.

  Returns:
    A Pandas DataFrame with the time and number of chords for each chord.
  """

  chords = []
  notes_df = notes_df.sort_values(by="start")
  for _, row in notes_df.iterrows():
    simultaneous_notes = notes_df[
        (notes_df["start"] >= row["start"]) & (notes_df["end"] <= row["end"])]
    if len(simultaneous_notes) >= threshold:
      chords.append((row["start"], len(simultaneous_notes)))
  return pd.DataFrame(chords, columns=["time", "num_chords"])


def main():
  # Load the dataset
  from datasets import load_dataset
  dataset = load_dataset("roszcz/internship-midi-data-science", split="train")

  # Get the notes column from the first record
  record = dataset[0]
  notes_df = pd.DataFrame(record["notes"])

  # Set the threshold for chord detection
  threshold = 6

  # Count the number of chords
  chords_df = count_chords(notes_df, threshold)

  # Plot the time vs. number of chords chart
  plt.plot(chords_df["time"], chords_df["num_chords"], marker="o", linestyle="-")
  plt.xlabel("Time (seconds)")
  plt.ylabel("Number of Chords")
  plt.title(f"Time vs. Number of Chords (Threshold: {threshold})")
  plt.grid(True)
  plt.show()


if __name__ == "__main__":
  main()
