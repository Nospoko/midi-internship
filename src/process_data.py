from itertools import combinations

import datasets
import pandas as pd
import fortepyan as ff
from tqdm import tqdm
import Levenshtein as Lv
from datasets import load_dataset

from helpers import play_midi_piece


def play_rec_no(number, dataset=None):
    if dataset is None:
        dataset = load_dataset("roszcz/maestro-v1", split="train+test+validation")
    record = dataset[number]
    piece = ff.MidiPiece.from_huggingface(record)
    play_midi_piece(piece)


def find_inconsistent_pieces(dataset):
    multiples = find_multiples(dataset)
    inconsistent = multiples.loc[~multiples["are similar"]][["index_1", "index_2", "title"]]
    return inconsistent


def notes_to_string(notes):
    """
    Converts a list of pitch values to a string of symbols.

    Args:
        notes (list): A list of pitch values (integers).

    Returns:
        str: A string containing the symbols based on the modulo 12 of the pitch values.
    """
    symbols = {0: "c", 1: "1", 2: "d", 3: "2", 4: "e", 5: "f", 6: "3", 7: "g", 8: "4", 9: "a", 10: "5", 11: "b"}
    string = str()
    for note in notes["pitch"]:
        string += symbols[note % 12]
    return string


def find_multiples(dataset: datasets.Dataset):
    """
    Find identically named data and check if it is consistent.

    This function takes a dataset containing musical compositions, calculates the similarity
    between the identically named data, and identifies pairs of similar compositions.

    Args:
        dataset (Dataset): A dataset containing musical compositions data. Each
                        dictionary should have keys: "notes" (list), "composer" (str), and
                        "title" (str).

    Returns:
        pandas.DataFrame: A DataFrame containing pairs of similar compositions along with
                        their indexes and titles. Columns: "are similar" (bool), "index_1" (int),
                        "index_2" (int), "title" (str).
    """
    multiples = []
    data = pd.DataFrame(dataset)
    data["full_name"] = data["composer"] + ", " + data["title"]
    data["count"] = data.groupby("full_name")["full_name"].transform("count")
    groups = data.groupby("full_name").groups
    for indexes_to_check in tqdm(groups.values()):
        group = []
        if len(indexes_to_check.tolist()) == 1:
            continue
        for index in indexes_to_check:
            record = dataset[index]
            string = notes_to_string(record["notes"])
            group.append([string, index, record["composer"] + ", " + record["title"]])
        for notes_first, notes_second in combinations(group, 2):
            similar = are_similar(notes_first[0], notes_second[0])
            multiples.append([similar, notes_first[1], notes_second[1], notes_first[2]])

    multiples = pd.DataFrame(multiples, columns=["are similar", "index_1", "index_2", "title"])
    return multiples


def are_similar(notes_first: str, notes_second: str, threshold=0.45):
    """

    This function compares two sequences of musical notes and determines whether they are similar
    based on the Levenshtein distance between the sequences.

    """
    length = max(len(notes_first), len(notes_second))
    if Lv.distance(notes_first, notes_second) <= length * threshold:
        return True
    else:
        return False


def main():
    dataset = load_dataset("roszcz/maestro-v1", split="train+test+validation")

    find_inconsistent_pieces(dataset).to_csv("inconsistent-data.csv")
    find_multiples(dataset).to_csv("multiples.csv")
    # play_rec_no(912, dataset)


if __name__ == "__main__":
    main()
