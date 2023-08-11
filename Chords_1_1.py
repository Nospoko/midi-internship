from class_and_setings import *


"""For a given record, create a chart of time vs. the number of chords played."""
if __name__ == "__main__":
    """ Data exploration with created functions. The column 'chord' indicates
     whether the functions recognized chords in each line. More information
     can be found in the function's docstring"""
    chords, chords_index, df_copy = find_chords(df, pitch_threshold)
    print("List of chords list:")
    print(chords[:5])
    print("\nListo of chords index:")
    print(chords_index[:5])
    print("\nDataFrame with added column to identyfication chords presence:")
    print(df_copy)
    draw_speed_plot(df, get_speed_calculation(df), y_name='chords')