import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

import speed
import chords

def fastest_15_seconds(record):
    notes, unit, binSize = speed.binned_number_of_notes(record, binSizeOverride=15)
    return notes["notes"].max()

# load the dataset
test_dataset = load_dataset("roszcz/maestro-v1", split="train")

# find the fastest 15 seconds of music
mostNotes = 0
for record in test_dataset:
    notes = fastest_15_seconds(record)
    if notes > mostNotes:
        mostNotes = notes
        fastestRecord = record

print("Fastest 15 seconds of music is", mostNotes, "notes long")
print("In piece: " + record["composer"] + " - " + record["title"] + "(" + str(record["year"]) + ")")
    
# Finding the most common chord
idx = None
mostCommonChord = 0
chord_interval = 0.05

for i in range(len(test_dataset)):
    record = test_dataset[i]
    df = pd.DataFrame(record["notes"])
    chords = chords.chordDetection(df, chord_interval)
    sorted_chords = chords.chord_count(chords)
    if sorted_chords.iloc[0]["count"] > mostCommonChord:
        idx = (record["composer"] + " - " + record["title"] + " (" + str(record["year"]) + ")")
        mostCommonChord = sorted_chords.iloc[0]["count"]
        chord = sorted_chords.iloc[0]["pitches"]

print(f"{idx} with most common chord {chord} repeated {mostCommonChord} times")

#Output:
# Fastest 15 seconds of music is 608 notes long
# In piece: Wolfgang Amadeus Mozart - Twelve Variations, K179 (189a)(2004)
#
# Modest Mussorgsky - Pictures at an E(2006) with most common chord (55.0, 43.0, 31.0) repeated 38 times