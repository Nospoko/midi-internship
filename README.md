# Data-Science Internship @ Piano For AI: MIDI Analytics

## MIDI

Don't worry if MIDI format is something strange to you. It is a very powerful protocol, and there are myriad
things you can do with it. We are doing something very specific in relation to piano music. To get a better
understanding about how to work with MIDI in this project, see [MIDI](./MIDI.md).

## MIDI Data

Here's a Hugging Face Dataset that should be used for algorithm development.

```python
from datasets import load_dataset

dataset = load_dataset("roszcz/internship-midi-data-science", split="train")
```

This dataset contains sample data from the [Piano For AI](https://pianofor.ai) project.
Each record in the dataset is a 1-2h recording of a practice session from various musicians.

Most of the tasks below require only the `notes` column, which for every record holds a list of events describing the pianists
actions on the keyboard (which note was played, with what force/velocity, for how long, when).

To read this data into a pandas data frame:

```python
import pandas as pd

record = dataset[0]
df = pd.DataFrame(record["notes"])
print(df.head())
```

To do this with our internal MIDI library:

```python
import fortepyan as ff

record = dataset[2]
piece = ff.MidiPiece.from_huggingface(record)
print(piece.df.head())
```

# Data Science Internship Challenge

# Objectives

Difficulty level: easy

#### Speed

1. For a given record, create a chart of time vs. speed.
    - If the record is longer than 120 seconds, use minutes as the time unit.
    - Use "notes played per second" as the speed unit.
2. Create a chart showing the number of notes pressed at the same time. Experiment with different thresholds.

#### Chords

1. For a given record, create a chart of time vs. the number of chords played.
2. Based on chord detection developed for 1., crate a table with the number of occurances of chords.
    - Try to use [chorder](https://github.com/joshuachang2311/chorder) to assign names to detected notes.

Using solutions you developed, review this dataset: https://huggingface.co/datasets/roszcz/maestro-v1, and find:

1. A piece with the fastest 15 seconds of music
2. A piece where a single chord is repeated the most (each piece will have a different chord)

---

Difficulty level: medium

#### N-grams

See https://en.wikipedia.org/wiki/N-gram for theoretical background. Use only pitch for tokenization.

1. For a given record, find the most popular 2-grams, 3-grams, and 4-grams.
    - Use 12 tokens corresponding to the 12 tones of an octave. Treat pitches an octave apart as the same token.
    - Use 88 tokens corresponding to the 88 keys of the piano.
2. Find n-grams based on note distance instead of pitch
    - Note distance is the time between start for note `a` and the start of the next note `b` (it's not duration).
    - Experiment with different numbers of tokens used to quantize the note distance. Choose one value and make a case for it.

---

Difficulty level: hard

#### Sequence Similarity

1. Given a sequence of notes, find similar sequences inside all available records. Sort by similarity.
    - Propose at least 2 different similarity metrics to compare sequences.

---

## Solution presentation

Fork this repository to your account and work on your solutions there. When you are ready open a Pull Request to this repository. Use the Pull Request description to present the results.
We will provide feedback and a code review in an iterative process - you can update your code after opening the PR as much as you need.

Here are some basic guidelines:

1. Make your presentation clear.
2. Use matplotlib to create charts.
3. Write code that is PEP8 compliant, readable, and sparks joy.
4. Join the [Piano For AI discord channel](https://discord.gg/67bHMBZTaT) to ask questions or discuss the project.
5. Don't feel obliged to solve everything, even a single algorithm is enough to open a PR.
6. Results from your presentation should be easy to reproduce with the code you are commiting.
7. Do not commit matplotlib figures as files - instead, embed images within your Pull Request descriptions.

## Development environment setup

Use python 3.9+

```sh
pip install -r requirements
```

### Code Style

This repository uses pre-commit hooks with forced python formatting ([black](https://github.com/psf/black),
[flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit` the files altered / added within the commit will be checked and corrected.
`black` and `isort` can modify files locally - if that happens you have to `git add` them again.
You might also be prompted to introduce some fixes manually.

To run the hooks against all files without running `git commit`:

```sh
pre-commit run --all-files
```
