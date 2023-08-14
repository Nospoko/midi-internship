import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
from datetime import timedelta
from tqdm import tqdm


from datasets import load_dataset


# Constants representing the state of a note
PRESSED = 0
RELEASED = 1


def pressed_notes_by_time(notes: DataFrame) -> DataFrame:
    """
    Calculates the number of simultaneously pressed notes at each moment
    based on the input notes DataFrame.
     Parameters:
        notes (DataFrame): A DataFrame containing MIDI note
    Returns:
        DataFrame with time index and the count of
            simultaneously pressed notes.
    """

    # Create a list of tuples representing the start and end states of notes
    notes_start_states = \
        [(start_time, PRESSED) for start_time in notes["start"]]
    notes_end_states = [(end_time, RELEASED) for end_time in notes["end"]]

    # Concatenate the start and end states and sort them based on time
    notes_states = notes_start_states + notes_end_states
    notes_states.sort(key=lambda x: x[0])

    # Variables to keep track of the number of pressed notes at each moment
    now_pressed_notes_count = 0
    notes_times = [0]
    notes_count = [0]

    # Compute the number of pressed notes at each time point
    for note in notes_states:
        if note[1] == PRESSED:
            now_pressed_notes_count += 1
        else:
            now_pressed_notes_count -= 1

        if notes_times[-1] == note[0]:
            notes_times[-1] = note[0]
            notes_count[-1] = now_pressed_notes_count
        else:
            notes_times.append(note[0])
            notes_count.append(now_pressed_notes_count)

    # Create a DataFrame with time index
    # and the count of simultaneously pressed notes
    simultaneously_notes_df = DataFrame(
        {"time": notes_times, "notes_count": notes_count}
    )
    # Convert the 'time' column to datetime, where 1.0 is 1 second
    simultaneously_notes_df["time"] = simultaneously_notes_df["time"].apply(
        lambda x: timedelta(seconds=x)
    )
    # Set the 'time' column as the index
    simultaneously_notes_df.set_index("time", inplace=True)

    return simultaneously_notes_df


def simultaneously_notes_task(time_unit=1):
    """
    Generates and displays a set of subplots, each containing
    plot of the number of simultaneously pressed notes
    over time for different MIDI records in the dataset.
    """

    dataset = \
        load_dataset("roszcz/internship-midi-data-science", split="train")
    current_record = 1

    for record in tqdm(dataset):
        # Extract the MIDI note data from
        notes = DataFrame(record["notes"])

        # Calculate the number of simultaneously pressed notes
        simultaneously_notes_df = pressed_notes_by_time(notes)

        # Initialize an empty list to store the time values after resampling
        notes_times = []

        simultaneously_notes_df = simultaneously_notes_df.resample(
            f"{time_unit}S"
        ).max()

        # Convert the timestamps to seconds
        # and adjust them according to the 'time_unit'
        for note in simultaneously_notes_df.index:
            notes_times.append(note.total_seconds() / time_unit)

        # Replace NaN values in 'notes_count' column
        simultaneously_notes_df["notes_count"] = simultaneously_notes_df[
            "notes_count"
        ].replace(np.nan, 0)

        # Create a new subplot
        plt.subplot(2, 3, current_record)
        plt.plot(notes_times, simultaneously_notes_df["notes_count"], lw=0.5)
        plt.fill_between(
            notes_times,
            simultaneously_notes_df["notes_count"].min(),
            simultaneously_notes_df["notes_count"],
            alpha=0.5,
        )
        plt.title(f"Record {current_record - 1}")
        plt.xlabel(f"Time per {time_unit} seconds")
        plt.ylabel("Simultaneously pressed notes")

        # Move to the next subplot in the grid
        current_record += 1

    # Display the subplots containing the visualizations

    plt.tight_layout()
    plt.show()


def resample_notes_df_by_start_time(notes: DataFrame,
                                    time_unit=1) -> DataFrame:
    notes = DataFrame({"start": notes["start"], "count": 1})
    notes["start"] = notes["start"].apply(lambda x: timedelta(seconds=x))
    notes.set_index("start", inplace=True)
    notes = notes.resample(f"{time_unit}S").count()
    return notes


def speed_notes_task(time_unit=1):
    """Main function for creating a chart of time vs. speed"""
    dataset = load_dataset("roszcz/internship-midi-data-science",
                           split="train")

    current_record = 1

    for record in tqdm(dataset):
        notes = DataFrame(record["notes"])
        notes = DataFrame({"start": notes["start"], "count": 1})

        notes["start"] = notes["start"].apply(lambda x: timedelta(seconds=x))

        notes.set_index("start", inplace=True)
        notes = notes.resample(f"{time_unit}S").count()
        notes_times = []

        for note in notes.index:
            notes_times.append(note.total_seconds() / time_unit)
        speed = list(map(lambda x: x / 60, notes["count"]))

        # Create a new subplot
        plt.subplot(2, 3, current_record)
        plt.plot(notes_times, speed, lw=0.8)
        plt.title(f"Record {current_record - 1}")
        plt.xlabel(f"Time interval {time_unit} seconds")
        plt.ylabel(f"Notes per {time_unit} seconds")

        # Move to the next subplot in the grid
        current_record += 1

    # Show plots
    plt.tight_layout()
    plt.show()


def find_fastest_interval(notes: DataFrame, time_unit=1, window_size=15):
    """
    Finds the fastest note interval within a rolling window of specified size.
    """

    # Resample the note data using the specified time unit
    notes = resample_notes_df_by_start_time(notes, time_unit)

    # Calculate the rolling mean of note counts with the given window size
    rolling_mean = (
        notes["count"]
        .rolling(
            window=window_size,
            center=True,
        )
        .mean()
    )

    return rolling_mean


def find_fastest_interval_task(time_unit=1, window_size=15):
    dataset = load_dataset("roszcz/maestro-v1", split="train")

    fastest_interval_ind = None
    fastest_interval_value = 0

    fastests_intervals = []
    current_record = 0

    for record in tqdm(dataset):
        notes = DataFrame(record["notes"])

        record_fastest_interval = find_fastest_interval(notes)

        # Find the index with the maximum rolling mean
        record_ind_max = record_fastest_interval.idxmax()
        record_value_max = record_fastest_interval[record_ind_max]

        fastests_intervals.append(
            {
                "record_number": current_record,
                "ind": record_ind_max,
                "value": record_value_max,
            }
        )

        if record_value_max > fastest_interval_value:
            fastest_interval_value = record_value_max
            fastest_interval_ind = record_ind_max

        current_record += 1
        fastests_intervals_df = DataFrame(fastests_intervals)
        fastests_intervals_df.to_csv("fastests_records.csv")

    # Print maximum of rolling mean and index
    print("Index with Maximum Rolling Mean:", fastest_interval_ind)
    print("The Maximum Rolling Mean:", fastest_interval_value)


if __name__ == "__main__":
    find_fastest_interval_task()
