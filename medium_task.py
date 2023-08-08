import math
from pandas import DataFrame
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter


def make_pitch_base_unigrams(df: DataFrame) -> list:
    """Function to create pitch difference based unigrams"""
    unigrams = []

    for ind, note in df.iterrows():
        if ind + 1 == len(df["pitch"]):
            break

        pitch_current = df.iloc[ind]["pitch"]
        pitch_next = df.iloc[ind + 1]["pitch"]

        unigrams.append(pitch_next - pitch_current)

    return unigrams


def make_pitch_base_n_grams(unigrams: list, n: int) -> list:
    """Function to create pitch difference based n-grams"""

    n_grams = []

    for i in range(len(unigrams) - n):
        n_grams.append(unigrams[i: i + n])  # Can be optimized with dqueue

    return n_grams


def pitch_based_n_grams_task():
    """Main function for analyzing n-grams based on pitch difference tasks"""

    dataset = load_dataset("roszcz/maestro-v1", split="train")

    for record in dataset:
        df = DataFrame(record["notes"])
        two_grams_counter = Counter()
        three_grams_counter = Counter()
        four_grams_counter = Counter()

        unigrams = make_pitch_base_unigrams(df)

        two_grams = make_pitch_base_n_grams(unigrams, 2)
        three_grams = make_pitch_base_n_grams(unigrams, 3)
        four_grams = make_pitch_base_n_grams(unigrams, 4)

        for sub_n_grams in two_grams:
            two_grams_counter[str(sub_n_grams)] += 1
        for sub_n_grams in three_grams:
            three_grams_counter[str(sub_n_grams)] += 1
        for sub_n_grams in four_grams:
            four_grams_counter[str(sub_n_grams)] += 1

        print(f"Composer: {record['composer']} - title {record['title']}")
        most_popular_2_grams = two_grams_counter.most_common(1)[0]
        most_popular_3_grams = three_grams_counter.most_common(1)[0]
        most_popular_4_grams = four_grams_counter.most_common(1)[0]
        print(
            f"The most popular 2-grams is {most_popular_2_grams[0]} "
            f"repeated {most_popular_2_grams[1]} times"
        )
        print(
            f"The most popular 3-grams is {most_popular_3_grams[0]} "
            f"repeated {most_popular_3_grams[1]} times"
        )
        print(
            f"The most popular 4-grams is {most_popular_4_grams[0]} "
            f"repeated {most_popular_4_grams[1]} times"
        )


def get_next_note(df: DataFrame, ind: int, threshold: int):
    """Get next note in threshold interval with highest pitch"""

    upper_envelope_note = df.iloc[ind + 1]
    curr_ind = ind + 1

    while True:
        if (
            curr_ind + 1 < len(df["start"])
            and df.iloc[curr_ind + 1]["start"]
            <= upper_envelope_note["start"] + threshold
        ):
            if df.iloc[curr_ind + 1]["pitch"] > upper_envelope_note["pitch"]:
                upper_envelope_note = df.iloc[curr_ind + 1]
            curr_ind += 1
        else:
            break

    return upper_envelope_note, curr_ind


def make_combined_unigrams(df: DataFrame) -> list:
    """
    Function to create unigrams based on pitch difference
    and relative onsets
    """

    unigrams = []

    # Decimal places
    accuracy = 1
    # Trashhold for simultaneous notes
    threshold = 0.03

    for ind, note in tqdm(df.iterrows()):
        if ind + 1 >= len(df["start"]):
            break

        upper_envelope_note, curr_ind = get_next_note(df, ind, threshold)

        if df.iloc[ind]["start"] == upper_envelope_note["start"]:
            upper_envelope_note, curr_ind = \
                get_next_note(df, curr_ind, threshold)

        if curr_ind + 1 >= len(df["start"]):
            break

        upper_envelope_note_next, curr_ind = \
            get_next_note(df, curr_ind, threshold)

        pitch_i = df.iloc[ind]["pitch"]
        pitch_i_1 = upper_envelope_note["pitch"]

        onset_i = df.iloc[ind]["start"]
        onset_i_1 = upper_envelope_note["start"]
        onset_i_2 = upper_envelope_note_next["start"]

        interval_i = pitch_i_1 - pitch_i
        ration_i = (onset_i_2 - onset_i_1) / (onset_i_1 - onset_i)
        ration_i = round(ration_i, accuracy)

        if math.modf(interval_i)[0] > 0:
            print(df.iloc[ind]["start"])
            print(upper_envelope_note)
            exit()

        unigrams.append(interval_i)
        unigrams.append(ration_i)

    return unigrams


def make_combined_n_grams(unigrams: list, n: int) -> list:
    """
    Main function for analyzing n-grams based on pitch difference
    and relative onsets
    """

    n_grams = []

    for i in range(0, len(unigrams) - n, 2):
        n_grams.append(unigrams[i: i + n])  # Can be optimized with dqueue

    return n_grams


def combined_n_grams_task():
    """Main function for analyzing n-grams relative duration-based tasks"""

    dataset = load_dataset("roszcz/maestro-v1", split="train")

    for record in dataset:
        df = DataFrame(record["notes"])
        unigrams = make_combined_unigrams(df)
        two_grams_counter = Counter()
        three_grams_counter = Counter()
        four_grams_counter = Counter()

        two_grams = make_combined_n_grams(unigrams, 2)
        three_grams = make_combined_n_grams(unigrams, 3)
        four_grams = make_combined_n_grams(unigrams, 4)

        for sub_n_grams in two_grams:
            two_grams_counter[str(sub_n_grams)] += 1
        for sub_n_grams in three_grams:
            three_grams_counter[str(sub_n_grams)] += 1
        for sub_n_grams in four_grams:
            four_grams_counter[str(sub_n_grams)] += 1

        print(
            f"Composer: {record['composer']} - title {record['title']}"
            f" year - {record['year']}"
        )
        most_popular_2_grams = two_grams_counter.most_common(1)[0]
        most_popular_3_grams = three_grams_counter.most_common(1)[0]
        most_popular_4_grams = four_grams_counter.most_common(1)[0]
        print(
            f"The most popular 2-grams is {most_popular_2_grams[0]} "
            f"repeated {most_popular_2_grams[1]} times"
        )
        print(
            f"The most popular 3-grams is {most_popular_3_grams[0]} "
            f"repeated {most_popular_3_grams[1]} times"
        )
        print(
            f"The most popular 4-grams is {most_popular_4_grams[0]} "
            f"repeated {most_popular_4_grams[1]} times"
        )
        

if __name__ == "__main__":
    pass
