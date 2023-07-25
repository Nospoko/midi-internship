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

