import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd

def plot_speed(df):
  """Plots of speed vs time.
   Time_interval will be computed as df['end']-df['start']

   speed, the note per second can be computed as 1/time_interval

   incase the time exceed 120 second we will convert it into minutes

  Args:
    df: A Pandas DataFrame containing the notes from a MIDI file.

  Returns:
    A  plot of matplotlib .
  """

# Calculate the time intervals and speed
  time_intervals = df["end"] - df["start"]
  speed = 1 / time_intervals

# Set the time unit
  time_unit = "seconds" if time_intervals.max() <= 120 else "minutes"

# Plot the data
  plt.plot(df["start"], speed)
  plt.xlabel(f"Time ({time_unit})")
  plt.ylabel("Speed (notes per second)")
  plt.title("Time vs. Speed")
  plt.show()

# Save the image
  plt.savefig("speed.png")


if __name__ == "__main__":
  # Load the dataset
  dataset = load_dataset("roszcz/internship-midi-data-science", split="train")

  # Extract the "notes" column for a given record
  record = dataset[0]
  df = pd.DataFrame(record["notes"])

  # Plot the speed
  #plot_speed(df)
