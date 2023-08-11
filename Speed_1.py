from class_and_setings import *


""" For a given record, create a chart of time vs. speed.
If the record is longer than 120 seconds, use minutes as the time unit.
Use "notes played per second" as the speed unit."""
if __name__ == "__main__":
    for record in dataset:
        df_speed = pd.DataFrame(record['notes'])
        bits_calculation = get_speed_calculation(df_speed)
        draw_speed_plot(df_speed, bits_calculation)
        break  # this break is needed if you want to show only 1 plot
