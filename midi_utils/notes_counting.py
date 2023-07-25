
import pandas as pd
import numpy as np

def aggregate_counts_by_time(record, count_col, time_col):
    """
    Utility to aggregate created note counting methods
    """
    grouped_record = (
        record.groupby([time_col])
        .agg(**{
            'notes_mean': pd.NamedAgg(count_col, 'mean'),
            'notes_std': pd.NamedAgg(count_col, 'std'),
        })
        .reset_index()
        .assign(
            lower_bound = lambda x: x['notes_mean'] - x['notes_std'],
            upper_bound = lambda x: x['notes_mean'] + x['notes_std'],
        )
    )
    # capping count at 0 as we can have negative count
    grouped_record.loc[:, 'lower_bound'] = np.where(grouped_record['lower_bound'] > 0, grouped_record['lower_bound'], 0)    
    return grouped_record



def count_notes_lookahead(record: pd.DataFrame, filter_count: int = 2):
    '''
    Counts how many notes were played simultaneously by looking at notes ahead played in same time interval as initial note
    '''
    overlapping_indices = []
    out_record = record.copy()

    for idx_current, row_current in out_record.iterrows():
        # list of note incdices that are played with the current note
        notes = [idx_current]

        for idx_next, row_next in out_record.iloc[idx_current+1:].iterrows():
            # gather all notes that start before the current note ends
            if row_current['end'] > row_next['start']:
                notes.append(idx_next)   
            else:
                break

        overlapping_indices.append(notes.copy())

    # addinng this information to df, filtering to cases that have at least N notes played at the same time
    out_record['overlapping_indices'] = overlapping_indices
    out_record['len_overlapp'] = out_record.apply(lambda x: len(x['overlapping_indices']), axis=1)
    out_record = out_record[out_record['len_overlapp'] >= filter_count]

    return out_record


def count_notes_fixed_interval(record: pd.DataFrame, interval_length: float = .5):
    '''
    Counts how many notes were played simultaneously in a specified time intervals
    '''
    max_duration = int(record['end'].max())
    
    # creating interval points like [(t1, t2), (t2, t3), ...]
    time_interval = np.arange(0, max_duration, interval_length).tolist()
    interval1, interval2 = time_interval[:-1], time_interval[1:]

    # counting notes in a specified time intervals
    occurences = [
        {
            'start': t1, 'end': t2,
            'notes_played': record[(record['start'] > t1) & (record['end'] <= t2)].shape[0]
        } 
        for t1, t2 in zip(interval1, interval2)
    ]
    
    # we will actually return a new dataframe to represent the time dimension of intervals
    occurences = pd.DataFrame(occurences, )
    occurences['minute'] = np.floor(occurences['start']/60).astype(int)
    return occurences