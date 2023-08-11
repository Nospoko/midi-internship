from class_and_setings import *
from Chords_1_1 import chords, df_copy
from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi import containers
from chorder.dechorder import Dechorder

"""Based on chord detection developed for 1.,
crate a table with the number of occurances of chords.
Try to use chorder to assign names to detected notes."""

if __name__ == "__main__":
    draw_chords_counter_plot(chords, top=100)
    df_copy['Octave'] = df_copy['pitch'] // 12
    df_copy['root_pc'] = df_copy['pitch'] % 12
    print(df_copy)

    mido_obj = mid_parser.MidiFile()
    beat_rseol = mido_obj.ticks_per_beat
    instrument = containers.Instrument(program=0, is_drum=False, name='example_track')
    mido_obj.instruments = [instrument]

    for index, row in df.iterrows():
        start = int(row['start'] * beat_rseol)
        end = int(row['end'] * beat_rseol)
        pitch = int(row['pitch'])
        velocity = int(row['velocity'])
        note = containers.Note(start=start, end=end, pitch=pitch, velocity=velocity)
        mido_obj.instruments[0].notes.append(note)

    mido_obj.dump('output.midi')

    mido_obj_re = mid_parser.MidiFile('output.midi')
    for note in mido_obj_re.instruments[0].notes:
        print(note)
        break
    print(len(mido_obj_re.instruments[0].notes))
    print(Dechorder.get_chord_quality(mido_obj_re.instruments[0].notes))
"""Here are my attempts to solve the chord assignment problem. I tried to use my code for chord assignment,
 but I realized that not all detected lists of notes are actually chords. So I am just stuck here.
  Here's what has been done:

*   Tried dividing the dataset into octaves to separate the bass line from the melodic line - **Failure**
 (chords can be played in high or low octave levels depending on the recording).
*   Attempted to add additional conditions related to timing to the chord recognition function - **Failure** 
(chords can be played with the use of arpeggio technique, so they can be played faster or slower or even simultanously).
* Tried using Dechorder to create a MIDI file and checked how the dedicated library works - **Failure** 
(after creating a MIDI file based on the loaded dataset, Dechorder did not identify anything in the MIDI file).
* Attempted to use unsupervised clustering methods (KNearestNeighbors) to classify notes as chords - Failure 
(chords can contain 3 notes as well as 5 groups of notes, so this method did not work well)."""

"""code with attempt to generate chords form df"""
# for note in mido_obj_re.instruments[0].notes:
#   a = Dechorder.get_chord_quality(note)
#   print(a)
#   break
