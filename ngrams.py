import numpy as np
import pandas as pd
from datasets import load_dataset


def find_n_gram_pitch_octave_apart(record, n):
    pitches = record["notes"]["pitch"]
    # treat pitches an octave apart as the same token
    pitches = [p % 12 for p in pitches]
    n_grams = {}
    for i in range(len(pitches) - n):
        n_gram = tuple(pitches[i : i + n])
        if n_gram not in n_grams:
            n_grams[n_gram] = 0
        n_grams[n_gram] += 1

    # sort n_grams by frequency
    n_grams = {k: v for k, v in sorted(n_grams.items(), key=lambda item: item[1], reverse=True)}
    return n_grams


def find_n_gram_all_keys(record, n):
    pitches = record["notes"]["pitch"]
    n_grams = {}
    for i in range(len(pitches) - n):
        n_gram = tuple(pitches[i : i + n])
        if n_gram not in n_grams:
            n_grams[n_gram] = 0
        n_grams[n_gram] += 1

    # sort n_grams by frequency
    n_grams = {k: v for k, v in sorted(n_grams.items(), key=lambda item: item[1], reverse=True)}
    return n_grams


def find_n_gram_distance_based(record, n):
    pitches, start_times = record["notes"]["pitch"], record["notes"]["start"]
    # find the distance between each note
    start_times = np.diff(start_times)
    # dimensions of start_times and pitches are off by one
    pitches = pitches[:-1]
    pitch_distance = pd.DataFrame({"pitch": pitches, "distance": start_times})
    # sort by distance
    pitch_distance = pitch_distance.sort_values(by=["distance"])
    # find n-grams
    n_grams = {}
    for i in range(len(pitch_distance) - n):
        n_gram = tuple(pitch_distance["pitch"][i : i + n])
        if n_gram not in n_grams:
            n_grams[n_gram] = 0
        n_grams[n_gram] += 1

    # sort n_grams by frequency
    n_grams = {k: v for k, v in sorted(n_grams.items(), key=lambda item: item[1], reverse=True)}
    return n_grams


def print_top_n_grams(n_grams, n, how_many=5):
    print(f"Top {how_many} {n}-grams:")
    for i, (n_gram, count) in enumerate(n_grams.items()):
        if i >= how_many:
            break
        print(f"{n_gram}: {count}")
    print()


if __name__ == "__main__":
    dataset = load_dataset("roszcz/internship-midi-data-science", split="train")

    n_values = [2, 3, 4]
    for record in dataset:
        print(f"Record id = {record['record_id']}")
        print("\npitches an octave apart treated as the same token")
        for n in n_values:
            n_grams = find_n_gram_pitch_octave_apart(record, n)
            print_top_n_grams(n_grams, n)

        print("\nall keys on the piano treated as different tokens")
        for n in n_values:
            n_grams = find_n_gram_all_keys(record, n)
            print_top_n_grams(n_grams, n)

        print("\nnotes ordered by distance between them")
        for n in n_values:
            n_grams = find_n_gram_distance_based(record, n)
            print_top_n_grams(n_grams, n)
