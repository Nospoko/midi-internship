from datasets import load_dataset
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

with load_dataset("roszcz/internship-midi-data-science", split="train") as dataset:
    record = dataset[0]
    df = (pd.DataFrame(record["notes"])).sort_values(by='start')

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


speed_df = get_events_frequency(df)

if __name__ == '__main__':
    plt.figure(figsize=(12, 6))
    plt.plot(speed_df['minute'], speed_df['speed'])
    plt.xlabel('Minute')
    plt.ylabel('Speed')
    plt.title('Speed of the song over time')
    plt.show()

#plot the distribution of starts in the song
if __name__ == '__main__':
    plt.figure(figsize=(12, 6))
    y, binEdges = np.histogram(df['start']/60, bins=175)
    plt.hist(df['start']/60, bins=175, edgecolor='blue')
    plt.subplot(212)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y/60, '-', c='blue')
    plt.show()