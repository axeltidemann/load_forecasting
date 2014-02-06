# Code demonstrating the use of Dataset (actually data.bchydro.Dataset).

from datetime import timedelta as dt

import numpy as np
import matplotlib.pyplot as plt

import sg.data.bchydro as bc
import sg.src.spclean as cln

# 7-day periods, selected with one day overlap (step length 6 days)
duration = 7
step = 6

# Create the dataset, specifying period and step length as datetime.timedelta
dataset = bc.Dataset(period=dt(days=duration), step_length=dt(days=step))

# Plot the first 5 periods sequentially with overlap
for period in (0, 1, 2, 3, 4):
    period_start_hrs = period * step * 24
    period_end_hrs = period_start_hrs + duration * 24
    x = np.arange(period_start_hrs, period_end_hrs)
    y = dataset.get_period(period)
    plt.plot(x, y)
plt.title("A sequence of 7-day periods selected with 1 day overlap.")

# Plot the same sequence using the original time series directly
plt.figure()
plt.plot(dataset.series[0:4*step*24+duration*24])
plt.title("Same data plotted by manually selecting a slice from the time series")

# Plot a random sequence
plt.figure()
(data, period_number) = dataset.get_random_period(True)
plt.plot(data)
plt.title("Period number %d (randomly selected).\n\nThis period starts at " \
              "index %d in the original time series." % \
              (period_number, dataset.index_of(period_number)))

# Show all figures
plt.show()
