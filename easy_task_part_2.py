import pandas as pd
from datasets import load_dataset

dataset = load_dataset("roszcz/maestro-v1", split="test")

def find_window_with_most_events(df, window_size=15):
    max_event_count = 0
    max_event_speed = None
    max_event_start_time = None
    # Sort the DataFrame by the 'start' column to ensure events are in chronological order
    df = df.sort_values(by='start')

    # Convert the 'start' column to pandas Timestamp for easier manipulation
    df['start'] = pd.to_datetime(df['start'])

    # Sliding window
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size]

        # Check if the number of events in the window is greater than the current maximum
        event_count = window['chord'].sum()
        if event_count > max_event_count:
            max_event_count = event_count
            max_event_speed = event_count / 15
            # Record the start time of the window with the most events
            max_event_start_time = window.iloc[0]['start']

    return max_event_speed, max_event_start_time

if __name__ == "__main__":
    max_event_speed = 0
    max_event_start_time = None
    max_event_record_id = None
    for i in range(len(dataset)):
        record = dataset[i]
        df = pd.DataFrame(record['notes'])
        df['pitch'] = df['pitch'] % 12
        event_speed, event_start_time = find_window_with_most_events(df)
        if event_speed > max_event_speed:
            max_event_speed = event_speed
            max_event_start_time = event_start_time
            max_event_record_id = i
    max_freq = 0
    max_freq_record_id = None
    for i in range(len(dataset)):
        record = dataset[i]
        df = pd.DataFrame(record['notes'])
        df['pitch'] = df['pitch'] % 12
        chords_df = detect_chords_conv(df, 0.7, 4, 2)
        # Calculate the frequency of each chord and get the maximum frequency
        chord_freq = chords_df['chord'].value_counts().max()
        # divide the frequency by the number of rows in the DataFrame
        freq_divided_by_shape = chord_freq / chords_df.shape[0]
        if freq_divided_by_shape > max_freq:
            max_freq = freq_divided_by_shape
            max_freq_record_id = i

fastest_15_seconds_dataset = dataset[max_event_record_id]
dataset_with_most_frequent_chord = dataset[max_freq_record_id]