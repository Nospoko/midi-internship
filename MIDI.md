# What is MIDI?

__Musical Instrument Digital Interface__ is a technical standard that enables electronic musical instruments, computers,
and other devices to communicate with each other. It allows these devices to transmit information about musical notation,
pitch, velocity, and control signals for parameters such as volume, vibrato, and audio panning.

### MIDI & Electric Pianos

In piano music, MIDI allows musicians to record performances in a structured, tabular, and computer-friendly format.
For the purpose of this project, we load this data into `pd.DataFrame`:

```java
    pitch     start       end  velocity
0      83  0.991667  3.190625        88
1      71  0.991667  3.203125        95
2      74  0.996875  3.238542        88
3      79  1.001042  3.238542        89
4      77  1.001042  3.245833        84
5      68  1.294792  1.400000        82
6      67  1.404167  1.504167        75
7      65  1.523958  2.816667        65
8      62  1.620833  1.785417        66
9      63  1.712500  2.816667        73
10     62  1.785417  2.816667        72
```

Every note played is represented as a row with information about the key pressed (pitch), loudness (velocity), and
time and duration of the note (with nanosecond precision)(not really).

### Pianoroll

In music production, it's a popular practice to represent MIDI recordings in form of "piano rolls", like this:

<img width="789" alt="image" src="https://github.com/Nospoko/midi-internship/assets/8056825/6b8b73f0-0080-433a-861a-32fd554c098d">

Here, every rectangle is a note, color hue depends on velocity. The background pattern follows the sequence of white and black keys on a piano. Vertical sticks also represent velocity, on a 0-127 scale.

### Animation

Here's the same fragment of a Chopin's etude with sound and movement:

https://github.com/Nospoko/midi-internship/assets/8056825/02ba5937-3aa1-4b64-9baf-373e7d6e2e6c

### Fortepyan

We developed our own library to work with piano recordings in MIDI format: [fortepyan](https://github.com/nospoko/fortepyan).
Here's the code to recreate everything from this document:

```python
import fortepyan as ff
from datasets import load_dataset
from matplotlib import pyplot as plt

# Maestro dataset is accessible through HuggingFace
maestro_test = load_dataset("roszcz/maestro-v1-sustain", split="test")

# Know thy data
chopin_idx = 74

# Load the MIDI file into our structure
piece = ff.MidiPiece.from_huggingface(maestro_test[chopin_idx])

# Preview the DataFrame
print(piece.df.head(10))

# Draw a pianoroll of the first 120 notes (whole piece has >2k)
ff.view.draw_pianoroll_with_velocities(piece[:120], title=title)
plt.show()

# Generate the animation (it's not efficient, your machine can get warm)
ff.view.make_piano_roll_video(piece[:120], movie_path='tmp/chopin.mp4')
```
