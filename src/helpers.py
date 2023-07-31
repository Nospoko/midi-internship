import os

import pygame
import numpy as np
import fortepyan as ff
import matplotlib.pyplot as plt
import miditoolkit.midi.containers as ct
from miditoolkit.midi.parser import MidiFile
from pretty_midi.pretty_midi import PrettyMIDI


def plot(x, y, xlabel, ylabel, title=None):
    """
    Create and save a bar plot using the given data.

    Parameters:
        x (list): A list of values for the x-axis.
        y (list): A list of values for the y-axis, representing the bar heights.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.

    Returns:
        None
    """
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))
    # Use pastel colors for the bars
    pastel_colors = ["#b3e2cd", "#fdcdac", "#cbd5e8", "#f4cae4", "#e6f5c9"]
    plt.bar(x, y, width=0.8, align="center", color=pastel_colors, alpha=0.8)
    plt.xlabel(xlabel, fontsize=14, color="#e6e6e6", fontfamily="monospace")
    plt.ylabel(ylabel, fontsize=14, color="#e6e6e6", fontfamily="monospace")
    plt.title(title, fontsize=18, color="#e6e6e6", fontfamily="monospace")
    plt.grid(True, color="gray", alpha=0.3)
    plt.tick_params(axis="x", labelsize=12, color="#e6e6e6")
    plt.tick_params(axis="y", labelsize=12, color="#e6e6e6")
    plt.savefig("./plots/" + title.replace(" ", "-").replace(",", "") + "-" + ylabel.lower() + "-vs-" + xlabel.lower() + ".png")
    plt.close()  # close the figure window


def midipiece_to_midifile(piece: ff.MidiPiece):
    """
    Converts a MidiPiece object to a MidiFile object.

    Parameters:
        piece (MidiPiece): A MidiPiece object containing notes.

    Returns:
        MidiFile: A MidiFile object representing the MIDI data.
    """

    def start_to_ticks(row, piece: PrettyMIDI):
        return piece.time_to_tick(row["start"])

    def end_to_ticks(row, piece: PrettyMIDI):
        return piece.time_to_tick(row["end"])

    mido_obj = MidiFile(ticks_per_beat=480)
    track = ct.Instrument(program=0, is_drum=False, name="track")
    mido_obj.instruments = [track]
    # print(piece.df.head())
    midi_data = piece.df
    piece = piece.to_midi()

    midi_data["start"] = midi_data.apply(start_to_ticks, axis=1, args=(piece,))
    midi_data["end"] = midi_data.apply(end_to_ticks, axis=1, args=(piece,))
    notes_midi = np.array(midi_data[["velocity", "pitch", "start", "end"]])

    for note in notes_midi:
        mido_obj.instruments[0].notes.append(ct.Note(*note))
    # I don't know how to do a better re-initialization than this
    # mido_obj.dump("res.mid")
    # mido_obj = MidiFile("res.mid", ticks_per_beat=480)
    return mido_obj


def play_midi_piece(midi_piece):
    piece = midipiece_to_midifile(midi_piece)
    piece.dump("piece.mid")
    play_midi_file("piece.mid")
    if os.path.isfile("piece.mid"):
        os.remove("piece.mid")


def play_midi_file(file_path):
    """
    Play a MIDI file using pygame.mixer.music.

    Parameters:
        file_path (str): The file path to the MIDI file that needs to be played.

    Returns:
        None

    Raises:
        pygame.error: If there is an error loading or playing the MIDI file.
    """
    pygame.init()
    pygame.mixer.init()

    try:
        pygame.mixer.music.load(file_path)
        print(f"Playing {file_path}")
        pygame.mixer.music.play()

        # Let the music play for a while (you can adjust the duration as needed)
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(30)

        # Stop playing the music
        pygame.mixer.music.stop()

    except pygame.error as error:
        print("Error loading or playing MIDI file:", str(error))

    pygame.quit()
