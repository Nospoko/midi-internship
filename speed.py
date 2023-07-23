from typing import Tuple

import pandas as pd


def record_length(record: pd.DataFrame) -> float:
    """
    Returns the record length in seconds.

    Parameters
    ----------
    record : pd.DataFrame
        Dataframe containing the information about the record

    Returns
    -------
    float
        Length in seconds
    """
    return record.end.max() - record.start.min()


def get_intervals_by_note(record: pd.DataFrame, duration: int = 1) -> pd.IntervalIndex:
    """
    Creates and returns an IntervalIndex for each time a note is played.

    Parameters
    ----------
    record : pd.DataFrame
        DataFrame containing the data about the notes played
    duration : int, optional
        Duration of the intervals, by default 1

    Returns
    -------
    pd.IntervalIndex
    """
    start_times = record["start"].unique()

    starts = []
    ends = []
    for i in start_times:
        starts += [i]
        ends += [i + duration]

    return pd.IntervalIndex.from_arrays(starts, ends, closed="left")


def get_intervals(record: pd.DataFrame, duration: int = 1) -> pd.IntervalIndex:
    """
    Creates and returns an IntervalIndex starting from the first note.

    Parameters
    ----------
    record : pd.DataFrame
        DataFrame containing data about the notes played
    duration : int, optional
        The duration of the intervals, by default 1

    Returns
    -------
    pd.IntervalIndex
    """
    record_interval = pd.Interval(0, record.end.max(), closed="left")
    intervals = pd.interval_range(
        start=record_interval.left,
        end=record_interval.right,
        freq=duration,
        closed="left",
    )
    return intervals


def notes_per_interval(record: pd.DataFrame, duration: int = 1) -> pd.DataFrame:
    """
    Creates and returns a DataFrame containing the number of notes played in each interval.

    Parameters
    ----------
    record : pd.DataFrame
        DataFrame contining data about the notes played
    duration : int, optional
        The duration of the intervals in which we want to count the notes, by default 1

    Returns
    -------
    pd.DataFrame
        DataFrame containing the number of notes played in each interval
    """
    intervals = get_intervals(record, duration)
    times = record[["start", "pitch"]].groupby("start").count()
    nps = times.groupby(pd.cut(times.index, intervals)).sum() / duration
    nps["count"] = nps["pitch"]
    nps = nps.drop("pitch", axis=1)
    return nps


def count_notes_in_interval(record: pd.DataFrame, interval: pd.Interval) -> int:
    """
    Counts the number of notes played in a given interval.

    Parameters
    ----------
    record : pd.DataFrame
        DataFrame containing data about the notes played
    interval : pd.Interval
        Interval in question

    Returns
    -------
    int
        Number of notes played in the given interval
    """
    return record.loc[(record["start"] >= interval.left) & (record["end"] < interval.right)]["pitch"].count()


def count_notes_by_intervals(record: pd.DataFrame, duration: int) -> pd.DataFrame:
    """
    For a given record and duration, counts the number of notes played in every interval of given duration that starts on a note.

    Parameters
    ----------
    record : pd.DataFrame
        DataFrame containing data about the notes played
    duration : int
        The duration of the intervals

    Returns
    -------
    pd.DataFrame
        DataFrame whose index is a pd.Interval object with the column 'notes_played' contains the number of notes played in that interval.
    """
    intervals = get_intervals_by_note(record, duration)

    rv = {interval: count_notes_in_interval(record, interval) for interval in intervals}
    rv = pd.DataFrame(list(rv.items()), columns=["interval", "notes_played"]).set_index("interval")
    return rv


def fastest_interval(record: pd.DataFrame, duration: int) -> Tuple[pd.Interval, int]:
    """
    For a given record and interval duration, returns the interval which has the most notes played.

    Parameters
    ----------
    record : pd.DataFrame
        DataFrame containing data about notes played
    duration : int
        The duration of the interval

    Returns
    -------
    Tuple[pd.Interval, int]
        A tuple (interval, count) where:
        * interval represents the interval in which the notes are played the fastest
        * count represents the number of notes played in said interval
    """
    interval_speed = count_notes_by_intervals(record, duration)
    rv = interval_speed[interval_speed.notes_played == interval_speed.notes_played.max()].iloc[0]
    return rv.name, rv.notes_played
