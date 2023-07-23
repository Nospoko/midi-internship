import argparse
from multiprocessing import Pool

import fortepyan as ff
from matplotlib import pyplot as plt

from data import MyDataset
from speed import fastest_interval

parser = argparse.ArgumentParser()
parser.add_argument(
    "--duration",
    type=float,
    required=True,
)
parser.add_argument(
    "--dataset-uri",
    type=str,
    required=True,
)
parser.add_argument("--fig-path", type=str)
parser.add_argument("--vid-path", type=str)
args = parser.parse_args()


def f(x):
    return fastest_interval(x, args.duration)


dataset = MyDataset(args.dataset_uri)

with Pool() as pool:
    res = pool.map(f, dataset)

record, interval, count = max(res, key=lambda x: x[2])

piece = record[(record.start >= interval.left) & (record.start < interval.right)]
print(f"Number of notes played in the fastest 15 seconds is {len(piece)}")


start = piece.start.min()
piece.start = piece.start - start
piece.end = piece.end - start

# we need to encapsulate in dict because of the way the method handles parameters
piece = {"notes": piece}

# Load the MIDI file into our structure
piece = ff.MidiPiece.from_huggingface(piece)


# Draw a pianoroll of the first
fig = ff.view.draw_pianoroll_with_velocities(piece, title="Fastest 15 seconds of music")
plt.show()
fig.savefig(args.fig_path)


# Generate the animation (it's not efficient, your machine can get warm)
if args.vid_path is not None:
    ff.view.make_piano_roll_video(piece, movie_path=args.vid_path)
