from datasets import load_dataset
from pandas import DataFrame
from matplotlib import pyplot as plt
import numpy as np
from fastdtw import fastdtw
import random
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance


def fastdtw_distance(record_1: dict, record_2: dict) -> float:
    """Function to calculate distance between two records using FastDTW"""

    pitch_sequence_1 = record_1["notes"]["pitch"]
    pitch_sequence_2 = record_2["notes"]["pitch"]

    pitch_sequence_1 = np.array(pitch_sequence_1).reshape(-1, 1)
    pitch_sequence_2 = np.array(pitch_sequence_2).reshape(-1, 1)

    distance, _ = fastdtw(pitch_sequence_1, pitch_sequence_2)

    return distance


def similarity_task_fastdtw():
    """Function for similarity comparison using FastDTW"""

    dataset = load_dataset("roszcz/maestro-v1", split="train")

    # Choose a random record
    random_record_number = random.randint(0, len(dataset) - 1)
    random_record = dataset[random_record_number]
    random_record_name = (f"{random_record['composer']}"
                          f"{random_record['title']}"
                          f" - {random_record['year']}")

    similarity_list = []

    current_record = 0
    for record in tqdm(dataset):
        # if len(similarity_list) > 100: break

        distance = fastdtw_distance(random_record, record)
        if distance > 100:
            similarity_list.append([distance, current_record])
        current_record += 1

    # Sort the similarity list based on distance
    similarity_list.sort()

    random_record_pitch = np.array(
        dataset[random_record_number]["notes"]["pitch"]
    ).reshape(-1, 1)

    for i in range(4):
        current_record = dataset[similarity_list[i][1]]
        current_record_pitch = current_record["notes"]["pitch"]
        current_record_name = (f"{current_record['composer']}"
                               f"{current_record['title']} - "
                               f"{current_record['year']}")

        plt.subplot(2, 2, i + 1)
        plt.plot(random_record_pitch, label=random_record_name, lw=0.9)
        plt.plot(current_record_pitch, label=current_record_name, lw=0.9)
        plt.title(f"Distance - {similarity_list[i][0]}")

        plt.xlabel("Time Step")
        plt.ylabel("Pitch")
        plt.legend()

    distance_list = [item[0] for item in similarity_list]
    record_number_list = [item[1] for item in similarity_list]
    name_list = list(
        map(
            lambda x: (f"{dataset[x]['composer']}"
                       f"{dataset[x]['title']} - {dataset[x]['year']}"),
            record_number_list,
        )
    )

    df = DataFrame(
        {
            "name": name_list,
            "distance": distance_list,
            "record_number": record_number_list,
        }
    )
    df.to_csv(f"similarity_list_for_record_{random_record_number}_dtw.csv")

    plt.suptitle(f"4 Most Similar Compositions to {random_record_name}")
    plt.tight_layout()
    plt.show()


def similarity_task_levenshtein():
    """Function for similarity comparison using Levenshtein distance"""
    dataset = load_dataset("roszcz/maestro-v1", split="train")

    # Choose a random record
    random_record_number = random.randint(0, len(dataset) - 1)
    random_record = dataset[random_record_number]
    random_record_name = (f"{random_record['composer']}"
                          f"{random_record['title']}"
                          f" - {random_record['year']}")

    similarity_list = []

    current_record = 0
    for record in tqdm(dataset):

        pitch_sequence_1 = random_record["notes"]["pitch"]
        pitch_sequence_2 = record["notes"]["pitch"]

        distance = levenshtein_distance(pitch_sequence_1, pitch_sequence_2)

        # Normalize the distance
        distance = distance / max(len(pitch_sequence_1), len(pitch_sequence_2))

        if distance > 0:
            similarity_list.append([distance, current_record])
        current_record += 1

    similarity_list.sort()

    random_record_pitch = dataset[random_record_number]["notes"]["pitch"]

    for i in range(4):
        current_record = dataset[similarity_list[i][1]]
        current_record_pitch = current_record["notes"]["pitch"]
        current_record_name = (f"{current_record['composer']}"
                               f"{current_record['title']} - "
                               f"{current_record['year']}")

        plt.subplot(2, 2, i + 1)

        plt.plot(random_record_pitch, label=random_record_name, lw=0.9)
        plt.plot(current_record_pitch, label=current_record_name, lw=0.9)
        plt.title(f"Distance - {similarity_list[i][0]}")

        plt.xlabel("Time Step")
        plt.ylabel("Pitch")
        plt.legend()

    plt.suptitle(f"4 Most Similar Compositions to {random_record_name}")
    plt.tight_layout()
    plt.show()

    distance_list = [item[0] for item in similarity_list]
    record_number_list = [item[1] for item in similarity_list]

    name_list = []

    name_list = list(
        map(
            lambda x: (f"{dataset[x]['composer']}"
                       f"{dataset[x]['title']} - {dataset[x]['year']}"),
            record_number_list,
        )
    )  # can be optimized

    df = DataFrame(
        {
            "name": name_list,
            "distance": distance_list,
            "record_number": record_number_list,
        }
    )
    df.to_csv(f"similarity_list_for_record_{random_record_number}"
              f"_levenshtein.csv")


if __name__ == "__main__":
    # similarity_task_levenshtein()
    similarity_task_fastdtw()
    pass
