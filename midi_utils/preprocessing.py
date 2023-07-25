import numpy as np
import pandas as pd

def create_time_features(record: pd.DataFrame):
    '''
    Creates time related features to a given MIDI dataframe
    '''
    record['duration'] = record['end'] - record['start']
    record['second'] = np.floor(record['start']).astype(int)
    record['minute'] = np.floor(record['start'] / 60).astype(int)
    return record


def count_notes_lookahead(record: pd.DataFrame):
    '''
    Counts how many notes were played simultaneously by looking at records ahead
    '''
    overlapping_indices = []

    for idx_current, row_current in record.iterrows():
        # list of note incdices that are played with the current note
        notes = [idx_current]

        for idx_next, row_next in record.iloc[idx_current+1:].iterrows():
            # gather all notes that start before the current note ends
            if row_current['end'] > row_next['start']:
                notes.append(idx_next)   
            else:
                break

        overlapping_indices.append(notes.copy())

    return overlapping_indices