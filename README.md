# Data-Science Internship @ Nospoko: MIDI Analytics

## Setup

Use python 3.9+

```sh
pip install -r requirements
```

## Midi Data

There's a Hugging Face Dataset that can be used for development.

```python
from datasets import load_dataset

dataset = load_dataset("roszcz/internship-midi-data-science", split="train")
```

This dataset contains sample data from the [Piano For AI](https://pianofor.ai) project.
Each record holds information about pianist-instrument interactions.
Most of the tasks below require only the `notes` data, which is a list of events describing the pianists
actions on the keyboard (which key was played, with what force/velocity, for how long).
To read this data into a pandas data frame:

```python
import pandas as pd

record = dataset[0]
df = pd.DataFrame(record)
```

## Data Science Tasks

1. Use matplotlib to create charts.
2. Do not commit any matplotlib figures - use Pull Requests to describe progress and show charts there.

### Performance Metrics

#### Speed

1. For a given record, create a chart of time vs speed.
    - If the record is longer than 120 seconds, use minutes as the time unit.
    - Use "notes played per second" as the speed unit.

#### Chords

1. For a given record, create a chart of time vs number of chords played.
2. Based on chord detection developed for 1., crate a table with number of occurances of chords.
    - Try to use [chorder](https://github.com/joshuachang2311/chorder) to assign names to detected notes.

### Language-like Properties

#### N-grams

1. For a given record, find the most popular 2-grams, 3-grams, and 4-grams.
    - Use 12 tokens corresponding to the 12 tones of an octave. Treat pitches an octave apart as the same token.
    - Use 88 tokens corresponding to the 88 keys of the piano.
2. Find n-grams based on note distance instead of pitch
    - Note distance is the time between start for note `a` and the start of the next note `b` (it's not duration).
    - Experiment with different numbers of tokens used to quantize the note distance. Choose one value and make a case for it.

#### Sequence Similarity

1. Given a sequence of notes, find similar sequences inside all available records. Sort by similarity.
    - Propose at least 2 different similarity metrics to compare sequences.

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
