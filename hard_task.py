from pydub.generators import Sine
import numpy as np
from datasets import load_dataset
import pandas as pd
from time import time
from pydub import AudioSegment, playback
#%%
gallop = np.array([4, 4, 4, 0, 4, 7, 7, 0, 4, 7, 11, 7, 4, 7, 9, 9, 9, 7, 5, 4, 0, 2, 11, 7])
#%%
def scale_array_with_distance_from_median(arr):
    return (arr - np.median(arr)).astype(int)
#%%
def generate_transition_matrix(sequence, num_states=12):
    # Initialize transition matrix with zeros
    transition_matrix = np.zeros((num_states, num_states))

    # Count occurrences of each transition
    for i in range(len(sequence) - 1):
        current_state = sequence[i]
        next_state = sequence[i + 1]
        transition_matrix[current_state, next_state] += 1

    # Calculate probabilities of each transition
    for i in range(num_states):
        total_occurrences = np.sum(transition_matrix[i])
        if total_occurrences > 0:
            transition_matrix[i] /= total_occurrences

    return transition_matrix
#%%
#supports stride, very efficient computationally
def stochastic_distance(transition_matrix,sequence):
    return np.linalg.norm((transition_matrix - generate_transition_matrix(sequence)),'fro')
#%%
#does not support stride,better at handling structure of the sequence
def diff_based_distance(sequence1, sequence2,order=1):
    return np.linalg.norm(np.abs(np.diff(sequence1,n=order) - np.diff(sequence2,n=order)))
#%%
def find_most_similar_sequence(input_sequence, pitches, stride):

    # Initialize the most similar sequence and its distance
    column_length = len(pitches)
    input_length = len(input_sequence)
    maximum_start = column_length - input_length
    most_similar_sequence = None
    most_similar_sequence_distance = float("inf")
    start_index = 0
    scaled_input = scale_array_with_distance_from_median(input_sequence)
    transition_matrix = generate_transition_matrix(scaled_input)
    # Slide the window across the column
    for i in range(0, maximum_start, stride):
        # Get the subsequence of pitches
        subsequence = pitches[i:int(i + input_length)]
        scaled_subsequence = scale_array_with_distance_from_median(pitches[i:int(i + input_length)])

        # Calculate the distance between the subsequence and the sequence
        distance = int(stochastic_distance(transition_matrix, scaled_subsequence))

        # Update the most similar sequence if necessary
        if distance <= most_similar_sequence_distance:
            most_similar_sequence = subsequence
            most_similar_sequence_distance = distance
            start_index = i

    return most_similar_sequence, start_index
#%%
def find_the_most_similar_sequence_in_dataset(sequence,path_to_dataset,stride):
    with load_dataset(path_to_dataset, split="train") as dataset:
         for i in range(len(dataset)):
            record = dataset[i]
            df = (pd.DataFrame(record["notes"])).sort_values(by='start')
            df['pitch'] = df['pitch'] % 12
            df['duration'] = df['end'] - df['start']
            pitches = df['pitch'].to_numpy()
            most_similar_sequence, start_index = find_most_similar_sequence(sequence,pitches,stride)
            if most_similar_sequence is not None:
                return most_similar_sequence, start_index, df
#%%
#function to locate the pitches in the dataframe from the set start to the set end and return them as a numpy array %12
def locate_pitches(df,start=0,end=float('inf')):
    return df.loc[(df['start'] >= start) & (df['end'] <= end),'pitch'].to_numpy() % 12
#%%
sequence, start_index, df = find_the_most_similar_sequence_in_dataset(gallop, 'roszcz/maestro-v1', 8)