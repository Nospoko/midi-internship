import pygame
import matplotlib.pyplot as plt


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
