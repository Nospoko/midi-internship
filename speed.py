import pandas as pd

def record_length(record: pd.DataFrame) -> int:
    
    return record.end.max() - record.start.min()

def get_intervals_by_note(record : pd.DataFrame, duration: int = 1) -> pd.DataFrame:
    start_times = record['start'].unique()
    
    starts = []
    ends = []
    for i in start_times:
        starts += [i]
        ends += [i+duration]

    return pd.IntervalIndex.from_arrays(starts, ends, closed='left')

def get_intervals(record: pd.DataFrame, duration: int = 1) -> pd.DataFrame:
    record_interval = pd.Interval(0, record.end.max(), closed="right")
    intervals = pd.interval_range(start=record_interval.left, end=record_interval.right, freq=duration, closed="left",)
    return intervals

def notes_per_interval(record:pd.DataFrame, duration:int=1):
    intervals = get_intervals(record, duration)
    times = record[['start','pitch']].groupby('start').count()
    nps = times.groupby(pd.cut(times.index, intervals)).sum() / duration
    nps['count'] = nps['pitch']
    nps = nps.drop('pitch', axis=1)
    return nps

def count_notes_in_interval(record: pd.DataFrame, interval: pd.Interval) -> int:
    return record.loc[(record['start'] >= interval.left) & (record['end'] < interval.right)]['pitch'].count()
    
def count_notes_by_intervals(record: pd.DataFrame, duration: int) -> pd.DataFrame:
    intervals = get_intervals_by_note(record, duration)
    
    rv = {interval : count_notes_in_interval(record, interval) for interval in intervals}
    rv = pd.DataFrame(list(rv.items()), columns=['interval', 'count'])
    return rv

        