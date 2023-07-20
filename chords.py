import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import speed

def chordDetection(df, chord_interval):
    """
    takes a dataframe of notes and chord_interval
    and returns a dataframe of chords
    """
    chords = []
    current_chord = {'start': df.iloc[0]['start'], 'end': df.iloc[0]['end'], 'pitches': [df.iloc[0]['pitch']]}

    for i in range(1, len(df)):
        current_note = df.iloc[i]
        time_difference = current_note['start'] - current_chord['end']
        
        if time_difference <= chord_interval:
            current_chord['end'] = current_note['end']
            current_chord['pitches'].append(current_note['pitch'])
        else:
            # Append the current_chord only if it has more than one note
            if len(current_chord['pitches']) > 1:
                chords.append(current_chord)
            current_chord = {'start': current_note['start'], 'end': current_note['end'], 'pitches': [current_note['pitch']]}

    # Append the last chord only if it has more than one note
    if len(current_chord['pitches']) > 1:
        chords.append(current_chord)

    return pd.DataFrame(chords)


def plot_chords_vs_time(df_chords):
    binSize, unit = speed.binSize_unit(record)

    bins = range(df_chords["start"].min().astype(int), df_chords["end"].max().astype(int), binSize)
    chords = df_chords.groupby(pd.cut(df_chords["start"], bins=bins))["end"].count().reset_index(name="count")

    chords.start = np.arange(1, len(chords) + 1)
    chords.columns = [unit, "chords"]

    plt.plot(chords[unit], chords["chords"])
    plt.xlabel(unit)
    plt.ylabel("Chords per Minute")
    plt.title("Chords vs Time")
    plt.show()
    

def chord_count(df_chords):
    """
    takes a dataframe of chords and returns a dataframe of chords and their counts
    """
    #transfer list to tuple
    df_chords['pitches'] = df_chords['pitches'].apply(tuple)
    # remove chords with less than 3 notes
    df_chords = df_chords[df_chords['pitches'].map(len) > 2]
    # return pitches, count table sorted by count
    return df_chords.groupby('pitches').size().reset_index(name='count').sort_values(by='count', ascending=False)