import chorder
from datasets import load_dataset
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mingus.core import notes, chords
from music21 import chord, duration
from pychord import find_chords_from_notes

with load_dataset("roszcz/internship-midi-data-science", split="train") as dataset:
    record = dataset[0]
    df = (pd.DataFrame(record["notes"])).sort_values(by='start')
    df['pitch'] = df['pitch'] % 12

#function to get the speed of the song
def get_events_frequency(events_df):
    """
    Calculate the frequency of events in 60-second intervals.

    Parameters:
        events_df (DataFrame): A pandas DataFrame containing events with 'start' and 'end' columns.

    Returns:
        DataFrame: A new DataFrame with the frequency of events in each 60-second interval.

    Example:
        events_df = pd.DataFrame({
            'start': [0, 30, 45, 80],
            'end': [15, 60, 50, 120]
        })
        result_df = get_events_frequency(events_df)
        print(result_df)

        Output:
           minute     speed
        0       0  0.250000
        1       1  0.750000
        2       2  0.750000
        3       3  0.750000
        4       4  0.750000
        5       5  0.750000
        6       6  0.750000
        7       7  0.750000
        8       8  0.750000
        9       9  0.750000
        10     10  0.750000
        11     11  0.750000
        12     12  0.750000
        13     13  0.750000
        14     14  0.750000
        15     15  0.750000
        16     16  0.750000
        17     17  0.750000
        18     18  0.750000
        19     19  0.750000
        20     20  0.750000
        21     21  0.750000
        22     22  0.750000
        23     23  0.750000
    """

    # Sort events based on their start times
    sorted_events = events_df.sort_values(by='start')

    # Create a dictionary to store the count of events in each 60-second interval
    interval_counts = defaultdict(int)

    # Iterate through the sorted events and count the events in each interval
    for index, event in sorted_events.iterrows():
        start_time = event['start']
        end_time = event['end']

        # Determine the start and end intervals for the event
        start_interval = start_time // 60
        end_interval = (end_time - 1) // 60

        # Count the event in each interval it spans
        for interval in range(int(start_interval), int(end_interval) + 1):
            interval_counts[interval] += 1 / 60

    # Convert the dictionary to a Pandas DataFrame
    result_df = pd.DataFrame(interval_counts.items(), columns=['minute', 'speed'])
    return result_df

#Function to find overlap using intersection over union
def iou(start1, end1, start2, end2):
    intersection = max(0, min(end1, end2) - max(start1, start2))
    end = max(end1, end2)
    start = min(start1, start2)
    support = intersection / (end - start + (end == start))
    return support ** 2

# Function to find overlap using Sorensen-Dice coefficient
def sorensen_dice(start1, end1, start2, end2):
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - intersection
    support = 2 * intersection / union
    return support
# Function to check if there is enough overlap for the pitch sequence
def enough_support(start1, end1, start2, end2, treshold):
    support = iou(start1, end1, start2, end2)
    return support >= treshold

# Function to map the numbers to notes
def num_to_note_arr(number_array):
    notes_to_map = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return (np.array(notes_to_map)[number_array]).tolist()

def chord_recognition_evaluation_metric(df):
    # Count the number of rows with an empty list in the 'chord' column
    empty_chord_count = df[df['chord'].apply(lambda x: isinstance(x, list) and len(x) == 0)].shape[0]

    # Calculate the metric
    metric = (1 - (empty_chord_count / df.shape[0])) if empty_chord_count != 0 else 1

    return metric


def check_freq_threshold(df, freq, range=3, coef=1):
    # Calculate the threshold value
    threshold = coef / (12 ** range)

    # Calculate the frequency divided by the number of rows in the DataFrame
    freq_divided_by_shape = freq / df.shape[0]

    # Check if the calculated value is smaller than the threshold
    if not (freq_divided_by_shape > threshold or freq > 2):
        return False
    else:
        return True


def detect_chords_conv(df, threshold, max_sequence_length, min_length):
    pitch_sequence_count = {}  # Dictionary to store the count of each pitch sequence

    # Sort the DataFrame by the 'start' column
    df = df.sort_values(by='start')
    # Function to extract pitch sequences using sliding window approach
    def extract_chord(row):
        pitch_sequence = [row['pitch'].astype(int)]
        end_time = row['end']

        for _, next_row in df.iloc[row.name + 1:].iterrows():
            if not enough_support(row['start'], end_time, next_row['start'], next_row['end'],
                                  treshold=threshold) or len(pitch_sequence) >= max_sequence_length:
                break
            else:
                pitch_sequence.append(next_row['pitch'].astype(int))
                end_time = next_row['end']

        # Convert the pitch sequence to a tuple to use it as a dictionary key
        pitch_sequence_tuple = tuple(pitch_sequence)
        pitch_sequence_count[pitch_sequence_tuple] = pitch_sequence_count.get(pitch_sequence_tuple, 0) + 1

        # Create a DataFrame to store the detected chord
        start = df.loc[row.name, 'start']
        end = df.loc[row.name, 'end']
        chord_name = chords.determine(num_to_note_arr(pitch_sequence),
                                      True)  # Implement the name_chord function to get the chord name

        chord_info = {
            'start': start,
            'end': end,
            'chord': chord_name,
            'length': len(pitch_sequence)
        }

        return pd.Series(chord_info)

    # Apply the function to each row in the DataFrame and drop rows with None (break statements)
    chord_df = df.apply(extract_chord, axis=1).dropna().reset_index(drop=True)
    chord_df = chord_df[chord_df['length'] >= min_length].reset_index(drop=True)
    return chord_df


chords_df = detect_chords_conv(df, 0.7, 7, 2)

chords_freq_df = get_events_frequency(chords_df)

if __name__ == '__main__':
    plt.figure(figsize=(12, 6))
    plt.plot(chords_freq_df['minute'], chords_freq_df['speed'])
    plt.xlabel('Minute')
    plt.ylabel('Speed')
    plt.title('Speed of the song over time')
    plt.show()