# Short demonstration of the utilities to load BCHydro data
import sys
import os
from datetime import timedelta as dt

import matplotlib.pyplot as plt

import sg.data.bchydro as bc

if __name__ == "__main__":
    # Option 1: load the entire dataset as a timeseries
    timeseries = bc.load()
    filtered = [x if x > 10 else 4000 for x in timeseries]
    plt.plot(filtered, '-')
    plt.title("The entire BC Hydro dataset")
    # Option 2: load the using the Dataset class
    dataset = bc.Dataset(period=dt(days=30), step_length=dt(days=7))
    plt.figure()
    plt.plot(dataset.get_random_period(), '-')
    plt.title("A randomly selected 30-day period from the BC Hydro dataset")
    plt.show()
