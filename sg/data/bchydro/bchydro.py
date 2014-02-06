#!/usr/bin/python

#import time
import os
import sys
import sqlite3
import datetime

import pandas as pd
import numpy as np

from sg.globals import SG_DATA_PATH
import sg.data

PATH_TO_BCH_DB = os.path.join(SG_DATA_PATH, "bchydro", "bchydro.db")
PATH_TO_BCH_HDF5_FILE = os.path.join(SG_DATA_PATH, "bchydro", "bc_weather.blosc-9.h5")

class Dataset(sg.data.Dataset):
    def __init__(self, period, step_length=None):
        """Loads the BCHydro time series and sets up for extraction of random
        slices of length 'period', 'step_length' apart. See class Dataset for
        more info."""
        sg.data.Dataset.__init__(self, load(), period, step_length)

class DatasetNonzero(sg.data.Dataset):
    def __init__(self, period, step_length=None):
        """Loads the BCHydro time series and sets up for extraction of random
        slices of length 'period', 'step_length' apart. See class Dataset for
        more info.

        Removes zero-valued observations by setting to the last recorded
        non-zero value."""
        sg.data.Dataset.__init__(self, load_remove_zeros(),
                                 period, step_length)

def load_remove_zeros(dbpath=PATH_TO_BCH_DB):
    """Read the load data from the given database. Return a Pandas time series
    containing the data, with zero-valued elements set to the last recorded
    non-zero value."""
    dataset = load(dbpath)
    return sg.data.remove_outlier_set_previous(dataset)

def temperature(hdf5path=PATH_TO_BCH_HDF5_FILE):
    h5store = pd.HDFStore(hdf5path, 'r')
    return h5store['bc_weather_mean']

def load(dbpath=PATH_TO_BCH_DB):
    """Read the load data from the given database. Return a Pandas time series
    containing the data."""
    with sqlite3.connect(dbpath, detect_types=sqlite3.PARSE_DECLTYPES|
                             sqlite3.PARSE_COLNAMES) as conn:
        crs = conn.cursor()
        crs.execute("SELECT Timestamp as 'stamp [timestamp]', "\
                    "MWh as 'load [float]' "\
                    "FROM loads " \
                    "ORDER BY Timestamp ASC")
        stamps, loads = zip(*crs.fetchall())
    return pd.Series(loads, index=stamps, dtype=float).asfreq('H')

def remove_holidays(loads):
    """Remove statutory holidays from dataset by replacing with the mean of the
    same weekday from the previous and the following weeks."""
    from holiday_parser import bc_holidays_for_year
    start = loads.first_valid_index()
    end = loads.last_valid_index()
    years = range(start.year, end.year + 1)
    one_week = datetime.timedelta(days=7)
    one_day = datetime.timedelta(days=1)
    for year in years:
        holidays = bc_holidays_for_year(year)
        for (_, when) in holidays:
            before = pd.date_range(start=when - one_week,
                                   end=when - one_week + one_day,
                                   freq=loads.index.freq)[:-1]
            after = pd.date_range(start=when + one_week,
                                   end=when + one_week + one_day,
                                   freq=loads.index.freq)[:-1]
            now = pd.date_range(start=when, end=when + one_day,
                                freq=loads.index.freq)[:-1]
            if before[0] >= start and after[-1] <= end:
                loads[now] = (loads[before].values + loads[after].values) / 2

def store_file(path):
    """Open a file, call LoadBCHydroStream to read and parse its contents, and
    finally store the data in the database."""
    with open(path, "r") as f:
        (timestamps, loads) = load_stream(f)
    storeindatabase(PATH_TO_BCH_DB, timestamps, loads)

def load_stream(stream):
    """Read a stream of lines containing dates, times and load values separated
    by semicolons. The date format is <alpha month> <numeric day>, <numeric
    year>. The time format is hour of day 1-24. Return a list of load values."""
    prev_timestamp = None
    loads = []
    timestamps = []
    for line in stream:
        (timestamp, load) = parseline(line)
        check_time_difference(prev_timestamp, timestamp)
        prev_timestamp = timestamp
        loads.append(load)
        timestamps.append(timestamp)
    return (timestamps, loads)

def check_time_difference(prev_timestamp, timestamp):
    """Check that the difference between previous and current timestamp is one
    hour."""
    if prev_timestamp is None:
        return
    if prev_timestamp + datetime.timedelta(hours=1) != timestamp:
        str_prev = prev_timestamp.strftime("%Y-%m-%d:%H")
        str_cur = timestamp.strftime("%Y-%m-%d:%H")
        raise RuntimeError("Time step error (0-based hours): '" +
                           str_prev + "' is not 1 hour before '"
                           + str_cur + "'.")
                           
def parseline(line, separator = ";"):
    """Expects a string with three (semicolon-)separated elements: a date
    (e.g. 'April 1, 2004', an hour of day (1-24), and a numeric value for that
    day. Returns a datetime and the value."""
    try:
        (datestr, hr, value) = line.split(separator)
    except ValueError as e:
        raise ValueError("Unable to split line into three items "
                         "(date, hr, value) using '" + separator +
                         "' as separator. Error message when "
                         "unpacking was: '" + e.args[0] + "'.")
    try:
        datetimestr = datestr + " " + str(int(hr)-1)
        d = datetime.datetime.strptime(datetimestr, "%B %d, %Y %H")
    except ValueError:
        raise ValueError("Unable to parse date and hour from string into "
                         "datetime. String was '" + datetimestr + ".")
    return (d, float(value))

def store_in_database(dbpath, timestamps, loads, stream=sys.stdout):
    """Creates a connection to the database used to store BC Hydro load
    data."""
    try:
        with sqlite3.connect(dbpath,
                             detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            crs = conn.cursor()
            crs.execute("DELETE FROM loads")
            crs.executemany("INSERT INTO loads VALUES (?, ?)",
                            zip(timestamps, loads))
        print >>stream, "Stored %d fresh timestamps and loads " \
            "in database %s." % (len(loads), dbpath)
    except sqlite3.Warning as w:
        print >>sys.stderr, "Warning raised while trying to write loads and " \
            "timestamps to database. Message was: " + w.args[0]
    except sqlite3.Error as e:
        raise RuntimeError("Failed to write loads and timestamps to " \
                           "database. Error message was: " + e.args[0])


if __name__ == '__main__':
    from unittest import main
    main(module='test_'+__file__[:-3])
