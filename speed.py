df['time_intervals'] = df['end'] - df['start']
df['speed'] = 1 / df['time_intervals']
if df['end'].max() > 120:
    df['start'] /= 60
    df['end'] /= 60
    time_unit = "minutes"
else:
    time_unit = "seconds"


plt.plot(df['start'], df['speed'])
plt.xlabel(f"Time ({time_unit})")
plt.ylabel("Speed (notes per second)")
plt.title("Time vs. Speed")
plt.show()