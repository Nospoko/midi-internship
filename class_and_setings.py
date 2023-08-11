import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset


record_long_limit = 120
thresholds = 2
pitch_threshold = [2, 3, 4, 5]
min_chord_size = 3
max_chord_size = 5
dataset = load_dataset("roszcz/internship-midi-data-science", split="train")
time_delay = 0.01
top_results = 10
record = dataset[0]
df = pd.DataFrame(record["notes"])


def get_sec_to_min(df):
  """Converts the duration of musical notes in seconds to minutes,
    considering the specified record long limit."""
  last_note_end = df.end.max()
  if last_note_end > record_long_limit:
    record_time_min = last_note_end/60
    return record_time_min
  return last_note_end


def calculate_octave(pitch):
  """Calculating octave for given pitch"""
  return (pitch // 12) - 1


def count_decimal_places(df, col):
  """
    Count the maximum number of decimal places
     in a specified column of a DataFrame."""
  max_precision = 0
  for val in df[col]:
    str_num = str(val)
    if '.' in str_num:
      num_precision = len(str_num) - str_num.index('.') - 1
      if num_precision > max_precision:
        max_precision = num_precision
  return max_precision


def find_chords(df, pitch_threshold, min_chord_size=3, max_chord_size=5):
  """
  Identify and label chords in a DataFrame containing musical pitch data.

  Args:
      * df (DataFrame): DF with musical pitch data.
      * pitch_threshold (list): A list of pitch differences
      that define the threshold for identifying chords.
      * min_chord_size (int, optional):
      The minimum size of a chord. Defaults to 3.
      * max_chord_size (int, optional):
      The maximum size of a chord. Defaults to 5.

  Returns:
      3 differece variables : A variables containing:
          - List of detected chords (each chord is a list of pitch values).
          - List of indices where chords were detected.
          - A modified copy of the input DataFrame with added
           'chord' column indicating the chord number."""
  chords = []
  chord_index = []
  chord_num = 1
  df_copy = df.copy()
  chord_sizes_5_4 = list(range(max_chord_size, min_chord_size - 1, -1))
  chord_sizes_3 = list(range(min_chord_size - 1, 2, -1))

  for i in range(len(df)):
    for chord_size in chord_sizes_5_4 + chord_sizes_3:
      index_list = df['pitch'][i:i + chord_size].index.to_list()

      if not any(idx in chord_index for idx in index_list):
        chord_frame = sorted(df['pitch'][i:i + chord_size].to_list())
        chord_set = set(chord_frame)

        if len(chord_set) == chord_size:
          results = [abs(chord_frame[j + 1] - chord_frame[j]) for j in range(len(chord_frame) - 1)]
          if max(results) <= max(pitch_threshold):
            chords.append(df['pitch'][i:i + chord_size].to_list())
            chord_index.extend(index_list)


            df_copy.loc[index_list, 'chord'] = chord_num
            chord_num += 1

  return chords, chord_index, df_copy


def get_speed_calculation(df):
  """
    Calculate the speed of musical notes in beats per second (bps)
     or beats per minute (bpm) based on the given DataFrame."""
  bps = df['start'].apply(lambda x: int(x))
  if df['end'].max() < record_long_limit:
    return bps
  else:
    return bps // 60


def draw_speed_plot(df,
                    bps,
                    record_name='default_record_name',
                    y_name='note'):
  """
    Draw a plot showing the quantities of notes or chords
     in relation to time or speed."""
  plt.title(f'Chart time/speed in record "{record_name}"')
  plt.ylabel(f'Quantities of {y_name} in tick')
  if y_name == 'note':
    plt.plot(bps.value_counts(sort=False))
  else:
    _, _, chord_df = find_chords(df, pitch_threshold, min_chord_size, max_chord_size)
    chord_df['tick'] = bps
    chord_counts = chord_df.groupby('tick')['chord'].nunique()
    plt.plot(chord_counts.index, chord_counts.values)
  if df['end'].max() < record_long_limit:
    plt.xlabel('Time [s]')
  else:
    plt.xlabel('Time [min]')
  plt.show()


def draw_notes_at_same_time_plot(df,
                                 start_col,
                                 time_threshold=0.2,
                                 top_number=None):
  """
    Draw a bar plot showing the frequency of the number of simultaneous
    notes played at the same time.
  """
  df_sorted = df.sort_values(by=start_col)
  time_diffs = np.diff(df_sorted[start_col])
  chords_count = np.ones(len(time_diffs), dtype=int)
  for i, time_diff in enumerate(time_diffs):
    j = i + 1
    while j < len(time_diffs) and time_diffs[j] <= time_threshold:
      chords_count[i] += 1
      j += 1
  unique_counts, count_occurrences = np.unique(chords_count, return_counts=True)
  if top_number is not None:
    top_number = min(top_number, len(unique_counts))
    top_indices = np.argsort(count_occurrences)[-top_number:]
    unique_counts = unique_counts[top_indices]
    count_occurrences = count_occurrences[top_indices]
  plt.bar(unique_counts, count_occurrences)
  plt.xlabel('Number of simultaneous notes')
  plt.ylabel('Occurrences')
  plt.title('Frequency of Simultaneous Notes')
  plt.show()



def to_integer(x):
  return int(x)


def get_table_with_counts(df):
  """
    Generate a modified DataFrame with added columns
    representing the counts of start and end times.
  """
  modified_df = df.copy()
  modified_df['time_stamp'] = modified_df['start'].apply(to_integer)
  start_counts = modified_df['start'].value_counts()
  modified_df['start_counts'] = modified_df['start'].map(start_counts)
  end_counts = modified_df['end'].value_counts()
  modified_df['end_counts'] = modified_df['end'].map(end_counts)
  return modified_df


def get_table_with_chords(df, min_threshold=1, max_threshold=100):
  """
    Generate a modified DataFrame with
    filtered rows based on start and end counts.
  """
  min_start_value = (df['start_counts'] > min_threshold)
  max_start_value = (df['start_counts'] < max_threshold)
  df_start_filter = df[(min_start_value) & (max_start_value)]
  min_end_value = (df_start_filter['start_counts'] > min_threshold)
  max_end_value = (df_start_filter['start_counts'] < max_threshold)
  df_end_filter = df_start_filter[(min_end_value) & (max_end_value)]
  return df_end_filter


def get_dict_for_list_counter(list_to_counter):
  """
    Count the occurrences of lists in a list and create
    a dictionary of their counts.
  """
  dict_chord_counts = {}
  for chord in list_to_counter:
    chord_tuple = tuple(chord)
    if chord_tuple in dict_chord_counts:
      dict_chord_counts[chord_tuple] += 1
    else:
      dict_chord_counts[chord_tuple] = 1
  return dict_chord_counts


def draw_chords_counter_plot(list_to_counter, top=None):
  """
    Draw a bar plot showing
    the frequency of occurrences of different chord lists.
  """
  chord_dict_counter = get_dict_for_list_counter(list_to_counter)
  chord_items = list(chord_dict_counter.items())

  if top is not None:
    sorted_chords = sorted(chord_items, key=lambda item: item[1], reverse=True)
    sorted_chords = sorted_chords[:top]
    chords_list, counts = zip(*sorted_chords)
  else:
    chords_list, counts = zip(*chord_items)

  plt.figure(figsize=(12, 6))
  plt.bar([str(chord) for chord in chords_list], counts)
  plt.xlabel('Chords')
  plt.ylabel('Occurrences')
  plt.title('Frequency of Chords in Record')
  plt.xticks(rotation=90, ha='center', fontsize=7)
  plt.xlim(-0.4, len(chords_list) - 0.3)
  plt.tight_layout()
  plt.show()
