import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def recordLength(record):
    """
    Returns the length of a record in seconds.
    """
    return max(record["notes"]["end"]) - min(record["notes"]["start"])


def binSize_unit(record):
    """
    Returns the bin size and time unit that should be used.
    """
    if recordLength(record) > 120:
        return (60, "minutes")
    else:
        return (1, "seconds")
    

def binned_number_of_notes(record, binSizeOverride=None):
    """
    Returns a dataframe with the number of notes in each bin.,
    units to use for graphs as well as the bin size.
    """
    binSize, unit = binSize_unit(record)
    if binSizeOverride is not None:
        binSize = binSizeOverride
        unit = "custom"
    
    recordDF = pd.DataFrame(record["notes"])

    bins = range(recordDF["start"].min().astype(int), recordDF["end"].max().astype(int), binSize)
    notes = recordDF.groupby(pd.cut(recordDF["start"], bins=bins))["velocity"].count().reset_index(name="count")

    notes.start = np.arange(1, len(notes) + 1)
    notes.columns = [unit, "notes"]

    return notes, unit, binSize


def plotRecord(record):
    """
    Plots the number of notes/s for each bin as a line chart.
    """
    notes, unit, binSize = binned_number_of_notes(record)
    plt.plot(notes[unit], notes["notes"]/binSize)
    plt.xlabel(unit)
    plt.ylabel("Notes per Second")
    plt.title("Speed vs Time")
    plt.show()
    

def plotAllRecords(dataset):
    """
    hard coded to plot records from 0 to 5
    in the training set
    """
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    for i, record in enumerate(dataset):
        notes, unit, binSize = binned_number_of_notes(record)
        axs[i//3, i%3].plot(notes[unit], notes["notes"]/binSize)
        axs[i//3, i%3].set_xlabel(unit)
        axs[i//3, i%3].set_ylabel("Notes per Second")
        axs[i//3, i%3].set_title("Speed vs Time for record " + str(record["record_id"]))
    plt.show()


def simultaneousNotes(record, threshold=0.1, fingers=10):
    """
    returns a list of lists of simultaneous notes being played
    """
    start = np.array(record["notes"]["start"])
    pitch = record["notes"]["pitch"]

    simultaneous = []
    checking = 0
    for i, time in enumerate(start):
        if i == checking:
            appendN = np.count_nonzero((start[i+1: i+fingers] - time) < threshold)
            # count true values in appendList
            simultaneous.append(pitch[i: i+appendN+1])
            checking += appendN + 1
        else:
            pass

    return simultaneous


def plotSimultaneousForRecord(record, threshold=0.1, fingers=10):
    """
    plots the number of simultaneous notes being played at each time
    """
    simultaneous = simultaneousNotes(record, threshold, fingers)
    
    x = np.arange(0, len(simultaneous))
    y = [len(i) for i in simultaneous]
    plt.scatter(x, y)
    plt.xlabel("Time")
    plt.ylabel("Number of Notes pressed")
    plt.title("Number of Notes pressed vs Time")
    plt.show()


def plotAllSimultaneous(dataset):
    """
    hard coded to plot records from 0 to 5 in the training set
    """
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    for i, record in enumerate(dataset):
        simultaneous = simultaneousNotes(record)
        x = np.arange(0, len(simultaneous))
        y = [len(i) for i in simultaneous]
        axs[i//3, i%3].scatter(x, y)
        axs[i//3, i%3].set_xlabel("Time [s]]")
        axs[i//3, i%3].set_ylabel("Number of Notes pressed")
        axs[i//3, i%3].set_title("Number of Notes pressed vs Time for record " + str(record["record_id"]))
    plt.show()

        