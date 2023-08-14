from datasets import load_dataset
from pandas import DataFrame
import pandas as pd
from datetime import timedelta
from matplotlib import pyplot as plt
from tqdm import tqdm
from music21.chord import Chord as ms21Chord
from music21.note import Note as ms21Note
from collections import deque
from collections import Counter


def extract_chords(notes_df: DataFrame, threshold: float = 0.03) -> DataFrame:
    """Extracts chords from MIDI note DataFrame based on time threshold."""

    threshold_timedelta = timedelta(seconds=threshold)

    notes_df["start"] = notes_df["start"].apply(lambda x: timedelta(seconds=x))

    notes_df["end"] = notes_df["end"].apply(lambda x: timedelta(seconds=x))

    founded_chords = []
    chord = []
    start_time = timedelta(seconds=0)
    times = []

    for ind, note in notes_df.iterrows():
        if not chord:
            chord.append(note["pitch"])
            start_time = note["start"]
            continue

        elif note["start"] - start_time <= threshold_timedelta:
            chord.append(note["pitch"])
        else:
            if len(chord) > 2:
                founded_chords.append(chord)
                times.append(note["start"])
            chord = [note["pitch"]]
            start_time = note["start"]

    chords_df = DataFrame({"start": times, "chord": founded_chords})

    return chords_df


def chord_speed_task(threshold: float = 0.03, time_unit: int = 60):  # Task 2 1
    """Visualizes chord speed over time for MIDI records."""

    dataset = \
        load_dataset("roszcz/internship-midi-data-science", split="train")
    current_record = 1
    for record in tqdm(dataset):
        notes_df = DataFrame(record["notes"])
        chords_df = extract_chords(notes_df, threshold)

        chords_df.set_index("start", inplace=True)

        resampled_chords = chords_df.resample(f"{time_unit}S").count()

        notes_times = []
        for note in resampled_chords.index:
            notes_times.append(note.total_seconds() / time_unit)

        chords_df = DataFrame({"time": notes_times,
                               "count": resampled_chords["chord"]})
        speed = list(map(lambda x: x, chords_df["count"]))

        plt.subplot(2, 3, current_record)
        plt.plot(notes_times, speed, lw=0.7)
        plt.title(f"Record {current_record - 1}")
        plt.xlabel(f"Time interval {time_unit} seconds")
        plt.ylabel(f"Chords per {time_unit} seconds")
        plt.suptitle(f"Chords per {time_unit} seconds,\
                      with threshold {threshold}")

        current_record += 1
    plt.tight_layout()
    plt.show()


def convert_chord_ms21Note(chord):
    """Converts a chord to music21 note objects."""
    return list(map(lambda x: ms21Note(x), chord))


def most_repeated_chord_task(threshold: float = 0.01,
                             piece_interval: int = 900):
    """Finds the most repeated chord within
    a specified time interval for MIDI records"""

    dataset = load_dataset("roszcz/maestro-v1", split="train")

    piece_interval = pd.Timedelta(seconds=piece_interval)
    chord_to_number = {}
    number_to_chord = {}
    most_common_chord = []
    current_record = 0

    for record in tqdm(dataset):
        chords = extract_chords(DataFrame(record["notes"]), threshold)

        if chords.empty:
            current_record += 1
            continue

        chords["chord"] = chords["chord"].apply(convert_chord_ms21Note)
        chords["chord_name"] = chords["chord"].apply(
            lambda x: ms21Chord(x).pitchedCommonName
        )

        chords = chords.loc[~chords["chord_name"].
                            str.contains("octave", case=False)]

        if chords.empty:
            current_record += 1
            continue

        for chord_name in chords["chord_name"]:
            if chord_name not in chord_to_number.keys():
                chord_to_number[chord_name] = len(chord_to_number)
                number_to_chord[len(number_to_chord)] = chord_name

        chords["name_number"] = chords["chord_name"].apply(
            lambda x: chord_to_number[x])
        chords = chords.drop("chord_name", axis=1)

        chords.reset_index(inplace=True, drop=True)
        chords.set_index("start", inplace=True, drop=False)

        start_time = chords.iloc[0]["start"]
        window_name_number = deque(
            chords[start_time:
                   start_time + piece_interval]["name_number"]
        )

        counter = Counter(window_name_number)

        most_common_chord_number = counter.most_common(1)[0][0]
        most_common_chord_count = counter.most_common(1)[0][1]

        i = 0
        while len(window_name_number) > 0:
            if most_common_chord_count < counter.most_common(1)[0][1]:
                most_common_chord_number = counter.most_common(1)[0][0]
                most_common_chord_count = counter.most_common(1)[0][1]

            left_chord_name_number = window_name_number.popleft()

            counter[left_chord_name_number] -= 1

            if len(window_name_number) + i >= len(chords["name_number"]):
                continue
            current_chord = chords.iloc[len(window_name_number) + i]

            window_name_number.append(current_chord["name_number"])
            counter[current_chord["name_number"]] += 1
            i += 1

        current_record += 1

        most_common_chord.append(
            {
                "record_number": current_record,
                "chord_name": number_to_chord[most_common_chord_number],
                "chord_count": most_common_chord_count,
                "title": record["title"],
                "composer": record["composer"],
                "year": record["year"],
            }
        )

    most_common_chord_df = DataFrame(most_common_chord)
    most_common_chord_df.to_csv("most_common_chords.csv")


if __name__ == "__main__":
    chord_speed_task()
    most_repeated_chord_task()
    pass
